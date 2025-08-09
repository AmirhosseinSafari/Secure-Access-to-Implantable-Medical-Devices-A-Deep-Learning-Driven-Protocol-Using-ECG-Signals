import os
import json
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import biosignalsnotebooks as bsnb
from scipy.signal import detrend
import seaborn as sns
import neurokit2 as nk
from io import BytesIO
import tflite_runtime.interpreter as tflite

import RPi.GPIO as GPIO
import time

import struct
import socket

HOST = "192.168.42.49"
programmer_ip = "192.168.41.122"
Rpeak_method = "pantompkins1985"
PORT = 8713

bpf = 3

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
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f"Programmer listening on {HOST}:{PORT}")
    return server_socket  

def dumper(data_dict):
    return json.dumps(data_dict)

def postman(data):
    return json.loads(data)

def send_data(sock, data):
    """Send an message with a length prefix."""
    message_length = len(data)
    sock.sendall(struct.pack("!I", message_length) + data.encode('utf-8'))  # Send length + message

def recv_ecg(sock):
    """Receive a complete encrypted message."""
    raw_length = sock.recv(4)  # Read 4-byte message length
    if not raw_length:
        return None  # Connection closed

    message_length = struct.unpack("!I", raw_length)[0]  # Unpack message length
    received_data = b""
    
    while len(received_data) < message_length:  # Keep receiving until full message arrives
        chunk = sock.recv(min(1024, message_length - len(received_data)))
        if not chunk:
            raise ConnectionError("Connection lost while receiving data.")
        received_data += chunk

    return received_data  # Return full ecg (message)

def process_ecg(unfiltered_ecg, fs):
   # ECG Filtering (Bandpass between 5 and 15 Hz)
   filtered_signal = bsnb.detect._ecg_band_pass_filter(unfiltered_ecg, fs)
   signals, info = nk.ecg_peaks(filtered_signal, sampling_rate=fs, method=Rpeak_method)
   rpeaks = info["ECG_R_Peaks"]

   return rpeaks, filtered_signal

# Normalizing method
def normalize(signal):
    a, b = -1, 1
    c = b - a
    aux = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    norm_ecg = c * aux + a
    return norm_ecg

# Calculates the mean distance between all peaks for each user
def peak_distance(r_peaks):
    dist = []
    for i in range(len(r_peaks)):
        if r_peaks[i] == r_peaks[-1]:
            break
        distance = r_peaks[i + 1] - r_peaks[i]
        if i == 0:
            dist.append(distance)
            continue
        if distance > np.mean(dist) + np.std(dist) * 2:
            continue
        else:
            dist.append(distance)
    return np.mean(dist)

def electrocardiomatrix(distance, r_peaks, filtered_ecg, sampling_rate, recording_signal_length):
    '''
    Creating bpf based EKMs in recording signal length
    '''
    # Initilization
    init_seg = int(0.2 * distance)
    fin_seg = int(1.5 * distance)
    peaks_window = bpf-1
    one_EKM_signal_size = recording_signal_length * sampling_rate

    # Definging the lower and upper bound of the 6-second EKM
    EKM_counter = 0
    lower_bound = one_EKM_signal_size * (EKM_counter)
    upper_bound = one_EKM_signal_size * (EKM_counter + 1)

    # Getting r peaks of one EKM (bpf in signal boundary length)
    r_peaks_all_slice = r_peaks[(r_peaks >= lower_bound) & (r_peaks <= upper_bound)]
    
    # Checking if there are enough r_peaks in the signal or not
    defficient_peaks_flag = False
    if len(r_peaks_all_slice) >= peaks_window:
        r_peaks_one_EKM = r_peaks_all_slice[0:peaks_window]
    else:
        defficient_peaks_flag = True

    # Returning if the EKM have not enough R peaks
    if defficient_peaks_flag == True:
        ekm = "Not enough peaks"
        return ekm, r_peaks_all_slice, [lower_bound, upper_bound]
    
    # Getting the segments
    all_segments = []
    for peak in r_peaks_one_EKM:
        segment = filtered_ecg[peak - init_seg : peak + fin_seg]
        all_segments.append(segment)

    norm_all_segments = normalize(all_segments)
    
    return norm_all_segments

def electrocardiomatrix_no_1(filtered_ecg, sampling_rate, window_size):
    init_window = 0
    window_signal_sample_size = window_size * fs
    each_line_ekm_size = 1 # seconds
    each_line_ekm_sample_signal_size = each_line_ekm_size * fs
    all_segments = []

    for ekm_line in range(window_size):
      segment = filtered_ecg[init_window + (ekm_line * each_line_ekm_sample_signal_size): \
                  init_window + ((ekm_line+1) * each_line_ekm_sample_signal_size)]
      all_segments.append(segment)

    ecm = normalize(all_segments)

    return ecm

###########################################
#   Initialization
###########################################

fs = 200
fig_width_px = 33
fig_height_px = 21
# user_id = 139
time_start_record = "now"

# set the protocol mode!
mode = "3_sec"

mode_seconds = {
    "normal": 6,
    "emergency": 4,
    "3_sec": 3
}

ecg_length_seconds = mode_seconds[mode]
window_size = ecg_length_seconds # seconds

number_of_energy_measurment = 100

# Pin configuration
signal_pin = 26  # GPIO pin connected to the Arduino

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
GPIO.setup(signal_pin, GPIO.OUT)

GPIO.output(signal_pin, GPIO.HIGH)
print("Plug the arduino in!")
time.sleep(10)

# Connecting to the programmer
programmer_socket = create_client_listen(programmer_ip)
print("Connected to programmer")

#############################
#   Energy capturing step: Normal mode
#############################
each_round_time = []

for _ in range(number_of_energy_measurment):
    try:
        start_time = time.perf_counter()
        # Start recording
        #print("Turning ON signal...")
        GPIO.output(signal_pin, GPIO.LOW)

        ######################
        #   MSG #1: wake up
        ######################

        # Step 0: send initial msg
        msg1_data = postman(programmer_socket.recv(1024).decode('utf-8'))
        # print(f"Msg 1 recived: {msg1_data}")

        ######################
        #   MSG #2: Sending the start recording time
        ######################

        msg2_data = {
            "time" : time_start_record
        }
        msg2_json = dumper(msg2_data)
        programmer_socket.sendall(msg2_json.encode('utf-8'))
        # print(f"Msg 2 sent: {msg2_data}")

        ######################
        #   Waiting: recording ecg
        ######################

        ecg_file = f"{ecg_length_seconds}seconds_signal.json"
        with open(ecg_file, "r") as f:
            signals = json.load(f)
        ecg_sig_IMD = signals["ylead"]

        time.sleep(ecg_length_seconds)

        ######################
        #   MSG #3: Getting the ecg of programmer
        ######################

        m3_json = recv_ecg(programmer_socket)
        m3_data = postman(m3_json.decode('utf-8'))
        # print(f"Msg 3 recived: ecg programmer length: {len(m3_json)}")
        
        ecg_sig_programmer = m3_data["ecg signal"]

        # Delay check
        peaks_IMD, filtered_ecg_IMD = process_ecg(ecg_sig_IMD , fs)
        peaks, filtered_ecg = process_ecg(ecg_sig_programmer , fs)
        distance = peak_distance(peaks)

        # print(f"peaks_IMD: {peaks_IMD}")
        # print(f"peaks programmer: {peaks}")
        # print(f"distance: {distance}")
        
        # temp_peaks_IMD = peaks_IMD[2:]
        # temp_peaks_programmer = peaks[1:]
        # delayed_flage = False

        # for i in range(len(temp_peaks_IMD)):
        #     if abs(temp_peaks_IMD[i] - temp_peaks_programmer[i]) > 0.15*distance:
        #         delayed_flage = True
        
        # print(f"Distance bounding result: {not delayed_flage}")

        # if not delayed_flage:

        # ECM creation process 
        detrend_signal = detrend(filtered_ecg)
        norm_ecg = normalize(detrend_signal)
        # print("Normalized the programmer signal.")

        ecm = electrocardiomatrix(distance, peaks, norm_ecg, fs, window_size)
        # print("Created the ECM (bpf).")

        fig = plt.figure(num=1, clear=True, figsize=(fig_width_px / 80, fig_height_px / 80))
        ax = fig.add_subplot()
        ax.axis('off')

        sns.heatmap(ecm, xticklabels=False, yticklabels=False, cbar=False)
        
        # Save to buffer instead of disk
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Use PIL to open image from buffer
        buf.seek(0)
        image = Image.open(buf)

        # Proceed with vectorization
        image = image.convert('RGB')
        image = image.resize((31, 20))
        image_array = np.array(image).astype('float32') / 255.0

        # print(f"image_array shape {image_array.shape}")
        # Add batch dimension
        input_data = np.expand_dims(image_array, axis=0)
        # print(f"input_data shape {input_data.shape}")

        # Load the TFLite model and allocate tensors.
        interpreter_auth = tflite.Interpreter(model_path=f'omniRI_{bpf}bpf_{window_size}sec.tflite')
        interpreter_auth.allocate_tensors()

        # Get input and output details.
        input_details = interpreter_auth.get_input_details()
        output_details = interpreter_auth.get_output_details()

        # Set the input tensor.
        interpreter_auth.set_tensor(input_details[0]['index'], input_data)

        # Run inference.
        interpreter_auth.invoke()

        # Retrieve and process the output.
        output_data = interpreter_auth.get_tensor(output_details[0]['index'])
        
        pred_user_id = np.argmax(output_data)
        # print(f"Result of authentication: {pred_user_id}")

        if pred_user_id == pred_user_id:
            # Programmer sbf
            sbf_programmer = electrocardiomatrix_no_1(norm_ecg, fs, window_size)
            
            fig = plt.figure(num=1, clear=True, figsize=(fig_width_px / 80, fig_height_px / 80))
            ax = fig.add_subplot()
            ax.axis('off')

            sns.heatmap(sbf_programmer, xticklabels=False, yticklabels=False, cbar=False)

            # Save to buffer instead of disk
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()

            # Use PIL to open image from buffer
            buf.seek(0)
            image_programmer = Image.open(buf)

            # Proceed with vectorization
            image_programmer = image_programmer.convert('RGB')
            image_programmer = image_programmer.resize((31, 20))
            image_array_programmer = np.array(image_programmer).astype('float32') / 255.0

            input_data_programmer = np.expand_dims(image_array_programmer, axis=0)

            # IMD sbf
            sbf_IMD = electrocardiomatrix_no_1(norm_ecg, fs, window_size)
            
            fig = plt.figure(num=1, clear=True, figsize=(fig_width_px / 80, fig_height_px / 80))
            ax = fig.add_subplot()
            ax.axis('off')

            sns.heatmap(sbf_IMD, xticklabels=False, yticklabels=False, cbar=False)

            # Save to buffer instead of disk
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()

            # Use PIL to open image from buffer
            buf.seek(0)
            image_IMD = Image.open(buf)

            # Proceed with vectorization
            image_IMD = image_IMD.convert('RGB')
            image_IMD = image_IMD.resize((31, 20))
            image_array_IMD = np.array(image_IMD).astype('float32') / 255.0

            input_data_IMD = np.expand_dims(image_array_IMD, axis=0)

            # Load the TFLite model and allocate tensors.
            interpreter_auth = tflite.Interpreter(model_path=f'omni_{window_size}sbf.tflite')
            interpreter_auth.allocate_tensors()

            # Get input details
            input_details = interpreter_auth.get_input_details()

            # Set input tensors
            interpreter_auth.set_tensor(input_details[0]['index'], input_data_programmer)
            interpreter_auth.set_tensor(input_details[1]['index'], input_data_IMD)

            # Run inference
            interpreter_auth.invoke()

            # Get output details
            output_details = interpreter_auth.get_output_details()
            output_data = interpreter_auth.get_tensor(output_details[0]['index'])
            # print(f"Result of replay attack check: {output_data}")

            if output_data[0] > 0.9 :
                result = "Pass"

        else:
            result = "Fail"

        ######################
        #   MSG #4: Sending the result (pass or fail)
        ######################
        # result = "pass"
        msg4_data = {
            "result" : result
        }
        msg4_json = dumper(msg4_data)
        programmer_socket.sendall(msg4_json.encode('utf-8'))

        # Stop recording
        #print("Turning OFF signal...")
        GPIO.output(signal_pin, GPIO.HIGH)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        each_round_time.append(execution_time)

        time.sleep(1)
    
    except Exception as e:
        print(e)
        

with open(f"./rounds_time_emergency_Auth_{ecg_length_seconds}seconds_{bpf}bpf_Replay_{ecg_length_seconds}sbf.json", "w") as f:
    json.dump(each_round_time, f)

print("Finished")
print("Pull out the sd card!")