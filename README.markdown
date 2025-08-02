# Secure Access to Implantable Medical Devices: A Deep Learning-Driven Protocol Using ECG Signals

## Table of Contents
- [Introduction to IoMTs and Security Importance](#introduction-to-iomts-and-security-importance)
- [Uniqueness of ECG Signals](#uniqueness-of-ecg-signals)
- [Using ECG Signals to Improve IMD Privacy](#using-ecg-signals-to-improve-imd-privacy)
- [ECG Dataset](#ecg-dataset)
- [Code Structure and Execution Guide](#code-structure-and-execution-guide)
  - [ECM Creation](#ecm-creation)
  - [SBF Creation](#sbf-creation)
  - [Authentication](#authentication)
  - [Replay Attack Check](#replay-attack-check)
  - [Energy Measurement](#energy-measurement)
- [Requirements](#requirements)
  - [Python Dependencies](#python-dependencies)
  - [Raspberry Pi Setup](#raspberry-pi-setup)
  - [Arduino Setup](#arduino-setup)
- [Attacks](#attacks)


## Introduction to IoMTs and Security Importance
The Internet of Medical Things (IoMTs) refers to a network of interconnected medical devices, such as implantable medical devices (IMDs), wearables, and health monitoring systems, that collect, transmit, and process sensitive health data. These devices, including pacemakers, insulin pumps, and biosensors, are critical for patient care but are vulnerable to cyber threats due to their connectivity. Unauthorized access to IoMTs can lead to data breaches, device manipulation, or even life-threatening consequences, such as altering a pacemaker’s settings. Ensuring robust security and privacy in IoMTs is paramount to protect patient data, maintain trust in healthcare systems, and comply with regulations like HIPAA and GDPR.

## Uniqueness of ECG Signals
Electrocardiogram (ECG) signals, which measure the electrical activity of the heart, are inherently unique to each individual, much like fingerprints. Their distinct patterns, including P-waves, QRS complexes, and T-waves, vary due to physiological differences, making them ideal for biometric authentication. Unlike traditional passwords or PINs, ECG signals are difficult to replicate or steal, as they are generated in real-time by the body. This uniqueness makes ECG signals a promising tool for enhancing the security of IMDs by providing a biometric-based access control mechanism.

![Figure 1](./Assets/ECG-waveform.jpg)  
*Caption: Example of an ECG signal showing P-wave, QRS complex, and T-wave in addition to length of ECM and SBF.*

## Using ECG Signals to Improve IMD Privacy
Our project leverages the unique properties of ECG signals to enhance the privacy of IMD owners. We developed a novel authentication protocol that uses ECG signals as a biometric key to secure communication between IMDs and external devices, such as programmers or monitoring systems. By integrating ECG-based authentication, only authorized users with matching ECG signatures can access or control the IMD, preventing unauthorized access. The system employs advanced signal processing and machine learning techniques to extract and compare ECG features, ensuring high accuracy and low false acceptance rates.

![Figure 2](./Assets/both_protocol.jpg)  
*Caption: Proposed ECG-based authentication protocol for IMDs.*

## ECG Dataset
The dataset used in this project is the E-HOL-03-0202-003 from the Telemetric and Holter ECG Warehouse at the University of Rochester. It was collected using a SpaceLab-Burdick digital Holter recorder with a pseudo-orthogonal lead setup (X, Y, Z electrodes), where X-lead signals are associated with the programmer and Y-lead signals with the IMD, both synchronized. The dataset includes 24-hour ECG recordings from 199 healthy individuals (4 excluded due to insufficient file size or poor quality). Participants had no cardiovascular diseases, high blood pressure (>150/90 mmHg), medication use, chronic illnesses, or pregnancy, confirmed by physical examinations, echocardiography, and exercise testing.

## Code Structure and Execution Guide
The project is organized into several directories, each containing scripts for specific functionalities. Below is the tree structure of the project and instructions for running the code to create ECMs (ECG Characteristic Matrices) and SBFs (Signal-Based Features), perform authentication, check for replay attacks, and measure energy consumption.

```
└── ECG_Authenticate
    ├── Assets
    │   ├── both_protocol.jpg
    │   ├── Conv_model_allusers.jpg
    │   ├── ECG-waveform.jpg
    │   ├── ECM_SBF.jpg
    │   ├── energy device scheme.jpg
    │   ├── Omni_Mono_uRI.jpg
    │   ├── power consuption device setup.jpg
    │   └── Trusted.jpg
    ├── Attacks
    │   └── attacks.ipynb
    ├── Authentication
    │   ├── auth_Mono_uRI.py
    │   ├── auth_Omni_RI.ipynb
    │   └── auth_Omni_uRI.py
    ├── ECM creation
    │   ├── bpf_EKMs.py
    │   └── user_ekms_functions.py
    ├── Energy measurment
    │   ├── ACIMD protocol
    │   │   ├── programmer_code.py
    │   │   └── whole_protocol_IMD.py
    │   ├── ADC-4CH-PinReading-of-Rasp.ino
    │   └── Our protocol
    │       └── programmer_code.py
    ├── README.markdown
    ├── Replay attack check
    │   ├── replay_check_mono.py
    │   └── replay_check_Omni.ipynb
    ├── requirements.txt
    └── SBF creation
        ├── sbf_EKMs.py
        └── user_ekms_functions.py
```

### ECM Creation

![!Figure](./Assets/ECM_SBF.jpg)

To create the ECG Characteristic Matrices (ECMs) for each user in the dataset, follow these steps:

1. **Navigate to the ECM creation directory**: This contains `bpf_EKMs.py` and `user_ekms_functions.py`.
2. **Configure `bpf_EKMs.py`**:
   - Open `bpf_EKMs.py` and specify the beats per frame (BPF), which is the number of heartbeats in an ECM, and the duration (in seconds) of the ECG signal segment from which these heartbeats are extracted.
   - Example configuration in `bpf_EKMs.py`:
     ```python
     bpf = 5  # Number of heartbeats per ECM
     duration = 6  # Duration of ECG segment in seconds
     ```
3. **Run the script**:
   - Execute `bpf_EKMs.py` to process the ECG dataset. The script uses functions from `user_ekms_functions.py` to generate ECMs for each user.
   - The output will be stored in a user-specific directory, typically structured as `ECM creation/output/[user_id]/`, containing the ECMs for each user in the dataset.

### SBF Creation
To create the Signal-Based Features (SBFs) for each user in the dataset, follow these steps:

1. **Navigate to the SBF creation directory**: This contains `sbf_EKMs.py` and `user_ekms_functions.py`.
2. **Configure `sbf_EKMs.py`**:
   - Open `sbf_EKMs.py` and specify the time duration (in seconds) for each SBF. This determines the length of the ECG segment used to extract features.
   - Example configuration in `sbf_EKMs.py`:
     ```python
     sbf_duration = 5  # Duration of ECG segment for SBF in seconds
     ```
3. **Run the script**:
   - Execute `sbf_EKMs.py` to process the ECG dataset. The script uses functions from `user_ekms_functions.py` to generate SBFs for each user.
   - The output will be stored in a user-specific directory, typically structured as `SBF creation/output/[user_id]/`, containing the SBFs for each user in the dataset.

### Authentication
The `Authentication` directory contains scripts for implementing ECG-based authentication protocols using ECMs extracted from users' ECG signals:

- **`auth_Omni_uRI.py`**: This script implements the Omni_uRI version of the authentication protocol, which authenticates users based on their ECMs. It unzips the ECM dataset, creates pairs of synchronized X-lead (programmer) and Y-lead (IMD) ECMs, and uses a Siamese neural network with 10-fold cross-validation to classify pairs as true (same user) or false (different users). The script vectorizes ECM images, trains the model, and saves performance metrics (accuracy, AUC-ROC, AUPR, confusion matrix) to a JSON file.

![Figure 3](./Assets/Omni_Mono_uRI.jpg)
*Caption: Proposed ECG-based authentication Siamese model: Omni_uRI, All users without trusting the IMD!.*

- **Other scripts** (`auth_Mono_uRI.py`, `auth_Omni_RI.ipynb`): These scripts operate similarly, implementing different variants of the authentication protocol (e.g., Mono_uRI for single-user focus or Omni_RI for alternative configurations). They follow the same process of dataset unzipping, ECM pairing, vectorization, and model training with 10-fold cross-validation.

![Figure 4](./Assets/Conv_model_allusers.jpg)
*Caption: Proposed ECG-based authentication Convolutional model: Omni_RI, All users with trusting the IMD!.*

To run these scripts:

1. Ensure the ECM dataset is available in the specified path (e.g., `../7 seconds_5 bpf EKM dataset_with 6000 EKMs length signal_padded`).
2. Execute the desired script (e.g., `python auth_Omni_uRI.py`). The script will generate and save authentication results in a JSON file.

### Replay Attack Check
The `Replay attack check` directory contains scripts to detect replay attacks, where an intruder attempts to use previously recorded ECG signals to gain unauthorized access:

- **`replay_check_mono.py`**: This script checks for replay attacks using SBF-based EKMs for each user (Mono approach). It assumes Y-lead EKMs represent the IMD's ECG signals and uses them as the programmer's signal. The script selects 1000 EKMs per user, creates true (same-user) and false (different-time) EKM pairs, and trains a Siamese neural network with 10-fold cross-validation to detect replay attempts. Results, including accuracy, AUC-ROC, AUPR, F1-score, precision, recall, and confusion matrix, are saved to JSON files, along with model weights and training history.
- **`replay_check_Omni.ipynb`**: This notebook implements a similar replay attack detection but for an Omni approach, considering multiple users simultaneously. It follows the same workflow as `replay_check_mono.py`.

To run `replay_check_mono.py`:

1. Ensure the SBF dataset is available (e.g., `../sbf no 1_6sbf`).
2. Execute `python replay_check_mono.py`. The script will process each user’s EKMs, train the model, and save results to the specified directories (`Results/ReplayCheck/`).

### Energy Measurement
The `Energy measurment` directory includes scripts and an Arduino sketch to evaluate the energy efficiency of the proposed protocol and a baseline ACIMD protocol:

- **Energy Measurement Device**: We implemented the authentication protocols on a Raspberry Pi 3 Model B v1.2 and developed a dedicated device to measure energy consumption. The device uses an Arduino to record the current (I) and voltage (V) of the Raspberry Pi during protocol execution. The Arduino operates independently from its power source to avoid additional power overhead. Communication between the Arduino and Raspberry Pi occurs via a GPIO pin, with the Raspberry Pi controlling when measurements are recorded. A pull-down resistor configuration minimizes power consumption during communication. The hardware components include an Arduino Uno, an Arduino shield, an LM358 IC, an ADC 1115 module, a memory module, resistors (0.1 Ω, 1 kΩ, 10 kΩ), and capacitors (1 μF, 10 μF).

![Figure 10](./Assets/energy%20device%20scheme.jpg)  
*Caption: Circuit diagram of the energy measurement device.*

![Figure 15](./Assets/power%20consuption%20device%20setup.jpg)  
*Caption: Setup of the power consumption measurement.*

- **`ADC-4CH-PinReading-of-Rasp.ino`**: This Arduino sketch records the current, voltage, and calculated power consumption of the Raspberry Pi. It uses the Adafruit ADS1015 ADC to read analog inputs, stores data in CSV format on an SD card, and responds to a GPIO signal from the Raspberry Pi to start/stop recording.

- **Protocol Implementation**: The `ACIMD protocol` and `Our protocol` subdirectories contain scripts (`programmer_code.py`, `whole_protocol_IMD.py`) that implement the authentication protocols using socket programming. The programmer and IMD communicate, while the Arduino-based energy measurement device records energy usage concurrently.

To run the energy measurement:

1. Set up the Arduino with the specified hardware components and upload `ADC-4CH-PinReading-of-Rasp.ino`.
2. Run the protocol scripts (e.g., `python programmer_code.py` in `Our protocol`) on the Raspberry Pi.
3. The Arduino will record energy data to an SD card when triggered by the Raspberry Pi’s GPIO signal.

## Requirements:
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
tensorflow>=2.12.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.8.0
pyserial>=3.5
pywt>=1.4.0
```

**Installation Command**:
Run the following command on the Raspberry Pi to install the required Python packages:
```bash
pip3 install -r requirements.txt
```

**Notes**:
- Ensure `pip3` corresponds to Python 3.9 or later, as commonly used on Raspbian.
- The `requirements.txt` file assumes compatibility with the Raspberry Pi environment. Verify package versions if using a different Python version.
- Standard Python libraries (`json`, `zipfile`, `os`, `pathlib`, `socket`, `time`) are included in Python and do not require installation.

### Raspberry Pi Setup
The project uses a Raspberry Pi 3 Model B v1.2 to run the authentication protocols and communicate with the Arduino for energy measurements. We utilized TensorFlow Lite to obtain trained models, which are optimized for deployment on the resource-constrained Raspberry Pi environment. The following steps ensure the Raspberry Pi is properly configured:

1. **Operating System**:
   - Use Raspbian (Debian-based OS, latest version recommended, e.g., Raspberry Pi OS Bullseye or Bookworm).
   - Update the system:
     ```bash
     sudo apt update
     sudo apt upgrade
     ```

2. **System Dependencies**:
   - Install dependencies for Python development and numerical computations:
     ```bash
     sudo apt install python3-pip python3-dev libatlas-base-dev
     ```
   - `libatlas-base-dev` is required for optimized numerical operations with `numpy` and `tensorflow`.

3. **Python Environment**:
   - Ensure Python 3.9 or later is installed (default on recent Raspbian versions).
   - Verify Python version:
     ```bash
     python3 --version
     ```
   - Install `pip3` if not already present:
     ```bash
     sudo apt install python3-pip
     ```

4. **TensorFlow Lite Installation**:
   - Install TensorFlow Lite for running the trained models on the Raspberry Pi:
     ```bash
     pip3 install tflite-runtime
     ```
   - Note: The full `tensorflow` package is included in `requirements.txt` for training and evaluation scripts, but `tflite-runtime` is used for deploying the optimized models on the Raspberry Pi.
   - Ensure trained models are converted to TensorFlow Lite format (`.tflite`) using TensorFlow’s converter before deployment. This is typically done on a more powerful machine before transferring the `.tflite` models to the Raspberry Pi.

5. **GPIO Configuration**:
   - Enable GPIO support for communication with the Arduino (used in `programmer_code.py` and `whole_protocol_IMD.py`):
     ```bash
     sudo raspi-config
     ```
     - Navigate to "Interfacing Options" > "GPIO" and enable it.
   - Install the `pyserial` package (included in `requirements.txt`) for serial communication with the Arduino.

6. **Storage**:
   - Ensure sufficient storage for the ECG dataset (E-HOL-03-0202-003), output files (ECMs, SBFs, JSON results), and TensorFlow Lite models. A 16GB or larger SD card is recommended.

### Arduino Setup
The energy measurement device uses an Arduino Uno to record current, voltage, and power consumption, controlled by the `ADC-4CH-PinReading-of-Rasp.ino` sketch. The following hardware and software components are required:

**Hardware Requirements**:
- Arduino Uno
- Arduino shield (for connecting components)
- LM358 IC (for signal amplification)
- Adafruit ADS1115 ADC module (for precise analog readings)
- SD card module (for data storage)
- Resistors: 0.1 Ω, 1 kΩ, 10 kΩ
- Capacitors: 1 μF, 10 μF
- Jumper wires and breadboard for connections

**Software Requirements**:
- **Arduino IDE**: Download and install the latest version from [https://www.arduino.cc/en/software](https://www.arduino.cc/en/software).
- **Arduino Libraries**:
  - **Adafruit_ADS1X15**: For interfacing with the ADS1115 ADC module.
  - **SD**: For writing data to the SD card.
  - **SPI**: For communication with the SD card module.
  - Install these libraries via the Arduino IDE Library Manager:
    1. Open Arduino IDE.
    2. Go to `Sketch` > `Include Library` > `Manage Libraries`.
    3. Search for and install `Adafruit ADS1X15` and `SD` (SPI is included with the Arduino IDE).

**Setup Instructions**:
1. **Assemble the Circuit**:
   - Connect the ADS1115 ADC module to the Arduino Uno (refer to the circuit diagram in `Assets/energy%20device%20scheme.jpg`).
   - Attach the SD card module, LM358 IC, resistors, and capacitors as described in the README.
   - Ensure a pull-down resistor configuration for the GPIO pin to minimize power consumption.

2. **Upload the Sketch**:
   - Open `ADC-4CH-PinReading-of-Rasp.ino` in the Arduino IDE.
   - Connect the Arduino Uno to your computer via USB.
   - Select the correct board (`Arduino Uno`) and port in the Arduino IDE.
   - Upload the sketch:
     ```bash
     # In Arduino IDE: Click the "Upload" button
     ```

3. **Connect to Raspberry Pi**:
   - Connect the Arduino to the Raspberry Pi via a GPIO pin for communication.
   - Ensure the Raspberry Pi’s `pyserial` package is installed to send start/stop signals to the Arduino.

4. **SD Card**:
   - Insert an SD card into the SD card module to store energy measurement data in CSV format.

**Notes**:
- Verify the Arduino’s serial port settings (baud rate) match those in the Python scripts (`programmer_code.py`, `whole_protocol_IMD.py`) for proper communication.
- Test the circuit setup using the diagram in `Assets/energy%20device%20scheme.jpg` and the physical setup in `Assets/power%20consuption%20device%20setup.jpg`.

## Attacks
The `attacks.ipynb` notebook evaluates the robustness of the authentication models against replay attacks by calculating the Attack Success Rate (ASR) for different model configurations. A replay attack occurs when an unauthorized entity attempts to gain access using previously recorded legitimate physiological signals. The notebook performs this analysis for both Omni-RI and Mono-RI models with varying signal durations and beats per frame (BPF). 

The ASR is calculated as the percentage of attempts where a sample is misclassified as another user (not the true user). The notebook provides both an overall ASR and a per-user ASR, highlighting the vulnerability of individual users to such attacks.