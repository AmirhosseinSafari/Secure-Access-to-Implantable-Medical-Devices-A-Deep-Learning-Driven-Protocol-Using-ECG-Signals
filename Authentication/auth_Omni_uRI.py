import os
from datetime import datetime
import random
import zipfile

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import json
import re

bpf = 5
seconds = 7

# Define the dataset path and output directories
base_dataset_path = f"../{seconds} seconds_{bpf} bpf EKM dataset_with 6000 EKMs length signal_padded"
dataset_path = f"{base_dataset_path}/EKMs_{seconds}secondsBoundary_{bpf}bpf.zip"
unzip_dir = f"../users_zip_files_{seconds}sec_{bpf}bpf_padded"
users_ekm_dir = f"../users_EKM_files_{seconds}sec_{bpf}bpf_padded"

# Unzip the main dataset file
print("Unzipping the EKM dataset")
if not os.path.exists(unzip_dir):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

# Create directory for user EKM files if it doesn't exist
os.makedirs(users_ekm_dir, exist_ok=True)

# Unzip individual user EKM zip files
if not os.path.exists(users_ekm_dir):
    user_zip_path = os.path.join(unzip_dir, "Users EKM zip")
    for _file in os.listdir(user_zip_path):
        file_name, _ = os.path.splitext(_file)
        file_dir = os.path.join(users_ekm_dir, file_name)
        
        os.makedirs(file_dir, exist_ok=True)
        
        with zipfile.ZipFile(os.path.join(user_zip_path, _file), 'r') as zip_ref:
            zip_ref.extractall(file_dir)

#########################################
# Healthy EKMs check
# - Moving window checking

path = f"{users_ekm_dir}/2005/x_lead"
EKMs = os.listdir(path)

def are_images_identical(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    return np.array_equal(np.array(img1), np.array(img2))

path_EKM1 = path + "/" + EKMs[0]
path_EKM2 = path + "/" + EKMs[1]

result = are_images_identical(path_EKM1, path_EKM2)
print("Images are identical:" if result else "Images are different.")

#########################################
# Creating dictionary of list of each user's EKMs
## Gathering same time EKMs

# Getting users id
users_id = []
path = users_ekm_dir
dirs = os.listdir(path)
for dir in dirs:
    user_id = dir.split("_")[-1]
    users_id.append(user_id)

print(f"users amount: {len(users_id)}")

def loading_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

# Function to extract the last integer from a string
def extract_last_int(s):
    return int(re.search(r'(\d+)$', s).group(1))

def sort_based_on_last_int(_list):
    # Sort the list based on the last integer in each string
    sorted_list = sorted(_list, key=extract_last_int)
    return sorted_list

#########################################
# Loading the Same time EKMs
# Save the dictionary to a JSON file
with open(f'{base_dataset_path}/sametime_EKMs_{bpf}bpf_{seconds}seconds.json', 'r') as f:
    same_time_EKMs_users = json.load(f)

## Adding path of EKMs of all users
# Making the path of same time EKMs
for user_id in same_time_EKMs_users.keys():
    for index, ekms_tuple in enumerate(same_time_EKMs_users[user_id]):
        y_lead_ekm = f"{users_ekm_dir}/{user_id}/y_lead/{bpf}bpf-ekm-bpf_recording_signal_length_ekm_dataset-{user_id}-{ekms_tuple[0]}.png"
        x_lead_ekm = f"{users_ekm_dir}/{user_id}/x_lead/{bpf}bpf-ekm-bpf_recording_signal_length_ekm_dataset-{user_id}-{ekms_tuple[1]}.png"

        same_time_EKMs_users[user_id][index] = (y_lead_ekm, x_lead_ekm)

#########################################
# Creating X and y
X = []
y = []

#########################################
## True labels

for user_id in same_time_EKMs_users.keys():
    X += same_time_EKMs_users[user_id]
    y += [1 for _ in range(len(same_time_EKMs_users[user_id]))]

print("Some true labels:")
print(X[:5])

## False labels

other_users_count = 50
true_labels_per_user = 1000
len_of_false_lbls_others_each = int(true_labels_per_user / other_users_count)

for user_id in same_time_EKMs_users.keys():
    other_users = random.sample([u for u in users_id if u != user_id], min(other_users_count, len(users_id) - 1))
    
    if len(same_time_EKMs_users[user_id]) >= true_labels_per_user:
        false_storage_ylead = []   
        for other_user in other_users:
            ylead_others = [tuple[0] for tuple in same_time_EKMs_users[other_user][:len_of_false_lbls_others_each]]
            false_storage_ylead += ylead_others
        
        false_storage_xlead = [tuple[1] for tuple in same_time_EKMs_users[user_id][:true_labels_per_user]]

        false_tuples = list(zip(false_storage_ylead, false_storage_xlead))
        X += false_tuples
        y += [0] * len(false_tuples)
    else:
        len_true_labels_user = len(same_time_EKMs_users[user_id])
        len_lbls_others_each = int(len_true_labels_user / other_users_count)

        false_storage_ylead = []
        for other_user in other_users:
            ylead_others = [tuple[0] for tuple in same_time_EKMs_users[other_user][:len_lbls_others_each]]
            false_storage_ylead += ylead_others

        false_storage_xlead = [tuple[1] for tuple in same_time_EKMs_users[user_id][:(len_lbls_others_each*other_users_count)]]

        false_tuples = list(zip(false_storage_ylead, false_storage_xlead))
        X += false_tuples
        y += [0] * len(false_tuples)

print("Some false labels:")
print(X[-5:])

X = np.array(X)
y = np.array(y)

# Vectorizing
def vertorizing_png_imges(address):
  # Load the PNG image
  image = Image.open(address)

  # Convert the image to RGB mode
  image = image.convert('RGB')

  # Resize the image to match the input size expected by the CNN
  desired_width = 31
  desired_height = 20
  image = image.resize((desired_width, desired_height))

  # Convert the image to a NumPy array
  image_array = np.array(image)

  # Reshape the array to match the input shape expected by the CNN
  # image_array = image_array.reshape((1, desired_height, desired_width, 3))

  # Normalize the array
  image_array = image_array.astype('float32') / 255.0

  return image_array

# from IPython.display import clear_output

def progress_bar(index, total_length, name_of_list):
    bar_length = 50

    # Calculate the percentage of completion
    percent_complete = (index / total_length) * 100

    # Clear the current cell's output
    # clear_output(wait=True)

    print(name_of_list)

    # Print the progress bar
    print("[", end="")
    completed_blocks = int(bar_length * (percent_complete / 100))
    print("*" * completed_blocks, end="")
    print("-" * (bar_length - completed_blocks), end="]\n")

    # Print the progress in the format: index/total_length
    print(f"{index}/{total_length}")

def vectorizing_list_of_ekms(ekm_list, name_of_list):
    # Vectorize a list of EKMs and return it
    num_ekms = len(ekm_list)
    vectorized_ekms = np.empty((num_ekms, 20, 31, 3), dtype=np.float32)

    for _, ekm_path in enumerate(ekm_list):
        veced_ekm = vertorizing_png_imges(ekm_path)
        vectorized_ekms[_, :] = veced_ekm
        if _ % 1000 == 0:
            progress_bar(_, num_ekms, name_of_list)

    return vectorized_ekms

# Vectorizing EKMs
X_xlead = [ekm[1] for ekm in X]
X_xlead = np.array(X_xlead)
X_xlead = vectorizing_list_of_ekms(X_xlead, "No.1 => x lead")

X_ylead = [ekm[0] for ekm in X]
X_ylead = np.array(X_ylead)
X_ylead = vectorizing_list_of_ekms(X_ylead, "No.2 => y lead")


## 10 fold 
# Model and prepration of data for fitting them to model

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
import json

# Create the Siamese sub-network
def create_siamese_subnetwork(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    return Model(inputs, x)

# Siamese Network
def create_siamese_network(input_shape):
    input_xlead = Input(shape=input_shape, name="x_lead_input")
    input_ylead = Input(shape=input_shape, name="y_lead_input")

    # Shared subnetwork
    shared_network = create_siamese_subnetwork(input_shape)

    # Encode x_lead and y_lead using the shared network
    encoded_xlead = shared_network(input_xlead)
    encoded_ylead = shared_network(input_ylead)

    # Concatenate the two encoded leads
    concatenated = tf.keras.layers.concatenate([encoded_xlead, encoded_ylead])

    # Final classification layer for binary output
    outputs = Dense(1, activation='sigmoid', name="classification_output")(concatenated)

    return Model(inputs=[input_xlead, input_ylead], outputs=outputs)

# Metrics calculation helper function
def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    return accuracy, auc_roc, aupr, conf_matrix

# Hyperparameters
input_shape = (20, 31, 3)
num_classes = 2
n_splits = 10

print(f"X_xlead {X_xlead.shape}")
print(f"X_ylead {X_ylead.shape}")
print(f"y {y.shape}")

# Check if the results file exists
results_file = f"siamese_binary_allUsers_10fold_results_{bpf}bpf_{seconds}seconds_siamese.json"
fold_results = {}

# Load the results from the previous run (if any)
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        fold_results = json.load(f)

next_fold = len(fold_results.keys())

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_xlead, y)):
    # Skip already completed folds
    if fold < next_fold:
        continue

    print(f"Training Fold {fold + 1}/{n_splits}...")
    
    # Split data
    X_xlead_train, X_xlead_test = X_xlead[train_idx], X_xlead[test_idx]
    X_ylead_train, X_ylead_test = X_ylead[train_idx], X_ylead[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create and compile the model
    model = create_siamese_network(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        [X_xlead_train, X_ylead_train],
        y_train,
        batch_size=32,
        epochs=100,
        verbose=1,
        validation_split=0.2
    )

    # Evaluate the model
    y_prob = model.predict([X_xlead_test, X_ylead_test]).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    accuracy, auc_roc, aupr, conf_matrix = calculate_metrics(y_test, y_pred, y_prob)

    # Save results
    fold_results[fold] = {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "aupr": aupr,
        "confusion_matrix": conf_matrix
    }
    print(f"Fold {fold + 1} Results - Accuracy: {accuracy}, AUC-ROC: {auc_roc}, AUPR: {aupr}")

    # Save all fold results to a JSON file
    with open(results_file, "w") as f:
        json.dump(fold_results, f, indent=4)
    print("Results saved to siamese_binary_fold_results.json")