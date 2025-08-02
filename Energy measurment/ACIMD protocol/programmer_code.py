import json
import socket
import struct

import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import os
from cryptography.hazmat.backends import default_backend

HOST = "192.168.42.49"
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
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f"Programmer listening on {HOST}:{PORT}")
    return server_socket  

def dumper(data_dict):
    return json.dumps(data_dict)

def postman(data):
    return json.loads(data)

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

def send_encrypted(sock, encrypted_data):
    """Send an encrypted message with a length prefix."""
    message_length = len(encrypted_data)
    sock.sendall(struct.pack("!I", message_length) + encrypted_data.encode('utf-8'))  # Send length + message

def key_to_str(key):
    """Convert a bytes key to a Base64-encoded string."""
    return base64.b64encode(key).decode('utf-8')

def str_to_key(key_str):
    """Convert a Base64-encoded string back to bytes."""
    return base64.b64decode(key_str)

def recv_encrypted(sock):
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

    return received_data  # Return full encrypted message

##############################
#       Initialization
##############################

number_of_energy_measurment = 100

# set the protocol mode!
mode = "normal"

mode_seconds = {
    "normal": 6,
    "emergency": 4
}
ecg_length_seconds = mode_seconds[mode]

# connecting to the IMD
programmer_socket = create_server_sender()
IMD_socket, IMD_address = programmer_socket.accept()
print(f"IMD connected: {IMD_address}")

for _ in range(number_of_energy_measurment):
    print("#####################################")
    print(f"Number of message {_}")
    ##############################
    #       MSG #1: wake up
    ##############################

    msg1_data = {
        "wake_up" : "Wake up!",
        "IDe" : "programmer_123"
    }

    msg1_json = dumper(msg1_data)
    IMD_socket.sendall(msg1_json.encode('utf-8'))
    print(f"Msg 1 sent: {msg1_data}")
    print(f"len msg 1: {len(msg1_data)}")

    ##############################
    #     MSG #2: Share key
    ##############################

    msg2_data = postman(IMD_socket.recv(1024).decode('utf-8'))
    session_key = str_to_key(msg2_data["Ks"])
    IDi = msg2_data["IDi"]
    print(f"Msg 2 recived: {msg2_data}")
    print(f"len Msg 2: {len(msg2_data)}")

    ##############################
    #     MSG #3: Nounce
    ##############################

    msg3_data = postman(IMD_socket.recv(1024).decode('utf-8'))
    Ni = msg3_data["Ni"]
    print(f"Msg 3 recived: {msg3_data}")
    print(f"len Msg 3 sent: {len(msg3_data)}")

    ##############################
    #     MSG #4: m1!
    ##############################

    Ne = Ni + 1

    ecg_file = f"{ecg_length_seconds}seconds_signal.json"
    with open(ecg_file, "r") as f:
        signals = json.load(f)
    ecg_sig_programmer = signals["xlead"]

    beta = ecg_sig_programmer

    m1_data = {
        "Ne" : Ne,
        "Ni" : Ni,
        "IDi" : IDi,
        "beta" : str(beta)
    }

    m1_json = dumper(encrypt_data(session_key, m1_data))

    # IMD_socket.sendall(m1_json.encode('utf-8'))
    send_encrypted(IMD_socket, m1_json)
    print(f"Msg 4: m1 sent: {str(m1_json)[:20]} ... {str(m1_json)[-20:]}")
    print(f"len Msg 4 sent: {len(m1_json)}")

    ##############################
    #     MSG #5: m2!
    ##############################

    # msg5_data = postman(IMD_socket.recv(1024).decode('utf-8'))
    msg5_json = recv_encrypted(IMD_socket)
    print(f"Msg 5 recived")

    IMD_socket.sendall("ok".encode('utf-8'))