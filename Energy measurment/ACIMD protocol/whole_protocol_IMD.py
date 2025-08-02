from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import json
import base64
import hashlib

from scipy.linalg import hadamard
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import RPi.GPIO as GPIO
import time

import socket

IMD_ip = "192.168.43.42"
programmer_ip = "192.168.43.49"
PORT = 8713

def create_client_listen(host):
    # Creates connection to the server (host)

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((host, PORT))

    return client_socket

def create_server_sender():
    # Creating a server (host)

    # Create a socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to an IP and port
    server_socket.bind((programmer_ip, PORT))
    server_socket.listen()

    print(f"Server listening on {programmer_ip}:{PORT}")
    return server_socket    

def generate_session_key(key_bit_length):
    """Generate a secure session key."""
    bytes = int(key_bit_length / 8)
    if bytes % 2 != 0:
        raise ValueError("Invalid key lengths: it must be devidable by 2.")
    session_key = os.urandom(32)  # Generate a 256-bit key
    #print(f"Generated Session Key: {base64.b64encode(session_key).decode()}")
    return session_key

def encrypt_data(session_key, data):
    """Encrypt a dictionary of data using the session key."""
    # Convert dictionary to JSON string
    json_data = json.dumps(data).encode()

    # Add padding to the data
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(json_data) + padder.finalize()

    # Generate a random IV (Initialization Vector)
    iv = os.urandom(16)

    # Create AES Cipher in CBC mode
    cipher = Cipher(algorithms.AES(session_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Return the IV and encrypted data, both encoded as base64
    return {
        "iv": base64.b64encode(iv).decode(),
        "ciphertext": base64.b64encode(encrypted_data).decode()
    }

def decrypt_data(session_key, encrypted_data):
    """Decrypt data using the session key."""
    # Decode base64-encoded IV and ciphertext
    iv = base64.b64decode(encrypted_data["iv"])
    ciphertext = base64.b64decode(encrypted_data["ciphertext"])

    # Create AES Cipher in CBC mode
    cipher = Cipher(algorithms.AES(session_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove padding from the decrypted data
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()

    # Convert JSON string back to dictionary
    return json.loads(data.decode())

def dynamic_quantizer(data, levels=256):
    """
    Perform dynamic quantization on the input data.
    
    Parameters:
        data (numpy array): The input data to be quantized.
        levels (int): The number of quantization levels (default is 256).
    
    Returns:
        quantized_data (numpy array): The quantized data.
        quantization_mapping (dict): Mapping of quantized values to original range intervals.
    """
    # Find the minimum and maximum of the data
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Calculate the step size
    step_size = (data_max - data_min) / (levels - 1)
    
    # Quantize the data
    quantized_data = np.round((data - data_min) / step_size).astype(int)
    
    # Create a mapping of levels to intervals
    quantization_mapping = {
        level: (data_min + level * step_size, data_min + (level + 1) * step_size)
        for level in range(levels)
    }
    
    return quantized_data, quantization_mapping

def remove_dc_component(signal):
    """
    Removes the DC component (mean) from the signal.
    
    Parameters:
        signal (numpy array): The input ECG signal.
    
    Returns:
        numpy array: Signal with the DC component removed.
    """
    return signal - np.mean(signal)

def bandpass_filter(signal, fs, lowcut=0.67, highcut=45.0, order=4):
    """
    Apply a Butterworth band-pass filter to the signal.
    
    Parameters:
        signal (numpy array): The input ECG signal.
        fs (float): Sampling frequency of the signal.
        lowcut (float): Lower cutoff frequency of the band-pass filter.
        highcut (float): Upper cutoff frequency of the band-pass filter.
        order (int): The order of the Butterworth filter.
    
    Returns:
        numpy array: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    # Apply the filter using filtfilt (zero-phase filtering)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def slice_ecg_signal(signal, fs, window_length=2):
    """
    Slice the ECG signal into non-overlapping windows of specified length.
    
    Parameters:
        signal (numpy array): The input ECG signal.
        fs (int): Sampling frequency of the signal (Hz).
        window_length (int): Length of each window in seconds.
    
    Returns:
        list: A list of ECG signal windows.
    """
    samples_per_window = int(window_length * fs)
    num_windows = len(signal) // samples_per_window
    windows = [signal[i * samples_per_window:(i + 1) * samples_per_window]
               for i in range(num_windows)]
    return windows

def apply_walsh_hadamard(signal_window):
    """
    Apply the Walsh-Hadamard transform to a signal window.
    
    Parameters:
        signal_window (numpy array): The input signal window.
    
    Returns:
        numpy array: The Walsh-Hadamard transformed signal.
    """
    # Pad the window to the nearest power of 2 if necessary
    n = len(signal_window)
    next_power_of_2 = 2 ** int(np.ceil(np.log2(n)))
    padded_window = np.pad(signal_window, (0, next_power_of_2 - n), mode='constant')
    
    # Generate the Hadamard matrix
    hadamard_matrix = hadamard(next_power_of_2)
    
    # Apply the Walsh-Hadamard transform
    transformed_signal = hadamard_matrix @ padded_window
    return transformed_signal

def ecg_sig_process(ecg_signal, number_of_windows, fs):
    '''
    Returns: walsh hadamard transfomred quantisized matrix
    '''
    window_length = 2
    one_matrix_length_sig = ecg_signal[:number_of_windows * window_length * fs]
    
    dc_removed_sig = remove_dc_component(one_matrix_length_sig)
    filtered_sig = bandpass_filter(dc_removed_sig, fs)
    windows = slice_ecg_signal(filtered_sig, fs)
    transformed_windows = [apply_walsh_hadamard(win) for win in windows]
    quantized_transformed_matrix = [dynamic_quantizer(win)[0] for win in transformed_windows]
    quantized_transformed_matrix = np.array(quantized_transformed_matrix)
    
    return quantized_transformed_matrix

def compute_correlation(X_i, X_e):
    """
    Compute the correlation between two sets of ECG vectors.

    Parameters:
        X_I (numpy array): Matrix representing the i-th set of vectors (e.g., from IMD).
        X_R (numpy array): Matrix representing the reference set of vectors (e.g., from Programmer).

    Returns:
        float: The correlation value between the two sets.
    """
    # Ensure inputs are numpy arrays
    X_i = np.array(X_i)
    X_e = np.array(X_e)
    
    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(X_i.flatten(), X_e.flatten())
    
    # Extract the correlation coefficient (off-diagonal element)
    correlation_value = correlation_matrix[0, 1]
    
    return correlation_value

def check_proximity(correlation_value, alpha):
    """
    Check if the correlation is within the acceptable range (threshold alpha).

    Parameters:
        correlation_value (float): Correlation value between the two signals.
        alpha (float): Threshold for the correlation value.

    Returns:
        bool: True if within the neighborhood area (inside NA), False otherwise.
    """
    if abs(correlation_value) < alpha:
        res = 1
    else:
        res = 0
    return res

def hash_input(data):
    """Hashes the input using SHA-256 and returns the hexadecimal digest."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif not isinstance(data, bytes):
        data = str(data).encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()

def concat_and_stringify(data1, data2):
    """Converts two inputs to strings, concatenates them, and returns the result."""
    return str(data1) + str(data2)

def dumper(data_dict):
    return json.dumps(data_dict)

def postman(data):
    return json.loads(data)

###########################################
#   Initialization
###########################################

fs = 200
alpha = 0.11
time_start_record = 3
IDi = "IMD_123"

# set the protocol mode!
mode = "normal"

mode_seconds = {
    "normal": 6,
    "emergency": 4
}

ecg_length_seconds = mode_seconds[mode]
number_of_windows = int(ecg_length_seconds / 2)

valied_key_lengths = [128, 192, 256]
number_of_energy_measurment = 100

# Connecting to the programmer
programmer_socket = create_client_listen(programmer_ip)
print("Connected to programmer")

# Pin configuration
signal_pin = 26  # GPIO pin connected to the Arduino

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
GPIO.setup(signal_pin, GPIO.OUT)

GPIO.output(signal_pin, GPIO.HIGH)
# print("Plug the arduino in!")
# time.sleep(10)

###########################
#   Key length : 128
###########################

# set this!
key_length_bits = valied_key_lengths[0]

#############################
#   Energy capturing step: Normal mode
#############################
each_round_time = []

for _ in range(number_of_energy_measurment):
    try:
        start_time = time.time()
        # Start recording
        #print("Turning ON signal...")
        GPIO.output(signal_pin, GPIO.LOW)
        
        ######################
        #   MSG #1: wake up
        ######################

        # Step 0: send initial msg
        msg1_data = postman(programmer_socket.recv(1024).decode('utf-8'))
        IDe = msg1_data["IDe"]
        print(f"Msg 1 recived: {msg1_data}")

        ######################
        #   MSG #2: Share key
        ######################

        # Step 1: Generate session key
        session_key = generate_session_key(key_length_bits)
        msg2_data = {
            "Ks" : session_key,
            "ts" : time_start_record,
            "IDi" : IDi
        }
        msg2_json = dumper(msg2_data)
        programmer_socket.sendall(msg2_json.encode('utf-8'))
        print(f"Msg 2 sent: {msg2_data}")

        ######################
        #   Waiting: recording ecg
        ######################

        ecg_file = f"{ecg_length_seconds}seconds_signal.json"
        with open(ecg_file, "r") as f:
            signals = json.load(f)
        ecg_sig_IMD = signals["ylead"]

        time.sleep(ecg_length_seconds)

        ######################
        #   MSG #3: Nounce
        ######################
        
        Ni = 10
        msg3_data = {
            "Ni" : Ni,
        }
        msg3_json = dumper(msg3_data)
        programmer_socket.sendall(msg3_json.encode('utf-8'))
        print(f"Msg 3 sent: {msg3_data}")
        
        ######################
        #   MSG #4: m1!
        ######################

        m1_data = postman(programmer_socket.recv(1024).decode('utf-8'))
        print(f"Msg 4: m1 recived: {m1_data}")

        ######################
        #   MSG #4: m2!
        ######################

        # Step 4: Decrypt the data
        decrypted_m1 = decrypt_data(session_key, m1_data)
        #print("\nDecrypted Data:", decrypted_m1)

        Ne = decrypted_m1["Ne"]
        Ni = decrypted_m1["Ni"]
        ecg_signal_programmer = decrypted_m1["beta"]

        x_matrix_e = ecg_sig_process(ecg_signal_programmer, number_of_windows, fs)
        x_matrix_i = ecg_sig_process(ecg_sig_IMD, number_of_windows, fs)

        corr_value = compute_correlation(x_matrix_i, x_matrix_e)
        result = check_proximity(corr_value, alpha)

        m2_data = {
            "Ni" : Ni,
            "Ne" : Ne,
            "hashed_result" : hashlib(concat_and_stringify(ecg_signal_programmer, result))
        }
        
        m2_json = dumper(encrypt_data(session_key, m2_data))
        programmer_socket.sendall(m2_json.encode('utf-8'))
        print(f"Msg 5: m2 sent: {m2_json}")

        # Stop recording
        #print("Turning OFF signal...")
        GPIO.output(signal_pin, GPIO.HIGH)
        
        end_time = time.time()
        execution_time = end_time - start_time
        each_round_time.append(execution_time)
        time.sleep(0.5)
                
    except Exception as error:
        print(error)
    #finally:
    #    GPIO.cleanup()  # Cleanup GPIO settings
    #    print("GPIO cleaned up.")

with open(f"./rounds_time_{ecg_length_seconds}seconds_{key_length_bits}bitkey.json", "w") as f:
    json.dump(each_round_time, f)

print("Finished")
print("Pull out the sd card!")
