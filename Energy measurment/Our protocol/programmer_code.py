import json
import socket
import struct

HOST = "192.168.41.122"
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

##############################
#       Initialization
##############################

number_of_energy_measurment = 100

# set the protocol mode!
mode = "3_sec"

mode_seconds = {
    "normal": 6,
    "emergency": 4,
    "3_sec": 3
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
    }

    msg1_json = dumper(msg1_data)
    IMD_socket.sendall(msg1_json.encode('utf-8'))
    print(f"Msg 1 sent: {msg1_data}")
    print(f"len msg 1: {len(msg1_data)}")

    ##############################
    #     MSG #2: Getting the start recording time
    ##############################

    msg2_data = postman(IMD_socket.recv(1024).decode('utf-8'))
    start_recording_time = msg2_data["time"]
    print(f"Msg 2 recived: {msg2_data}")
    print(f"len Msg 2: {len(msg2_data)}")

    ##############################
    #     MSG #3: sending the recorded ECG
    ##############################

    ecg_file = f"{ecg_length_seconds}seconds_signal.json"
    with open(ecg_file, "r") as f:
        signals = json.load(f)
    ecg_sig_programmer = signals["xlead"]

    msg3_data = {
        "ecg signal" : ecg_sig_programmer
    }

    msg3_json = dumper(msg3_data)

    # IMD_socket.sendall(m1_json.encode('utf-8'))
    send_data(IMD_socket, msg3_json)
    print(f"Msg 3: ecg signal sent: {str(msg3_json)[:20]} ... {str(msg3_json)[-20:]}")
    print(f"len Msg 3 sent: {len(msg3_json)}")

    ##############################
    #     MSG #4: Pass or fail!
    ##############################

    # msg5_data = postman(IMD_socket.recv(1024).decode('utf-8'))
    msg4_data = postman(IMD_socket.recv(1024).decode('utf-8'))
    print(f"Msg 4 recived")