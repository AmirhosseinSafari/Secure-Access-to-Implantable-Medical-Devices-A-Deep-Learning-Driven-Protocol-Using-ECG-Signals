# # Purpose
# Creating a network which checks the replay attack 
# with sbf EKMs for each of the users (Mono)
# Assumptions:
# - y_lead are the EKMs from IMDs ECG signals.
# - So we use y_lead as the programmer's signal
# - 1000 EKMs for each user

######################################################
# Imports and installations

# Imports and installations
import os
from datetime import datetime
import random
import zipfile

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import json
import re

######################################################
# Unzipping the dataset
sbf = 6
phase_name = "Mono"

# Define the dataset path and output directories
base_dataset_path = f"../sbf no 1_6sbf"
dataset_path_zip = f"{base_dataset_path}/EKMs_{sbf}sbf.zip"
unzip_dir = f"../users_zip_files_{sbf}sbf"
users_ekm_dir = f"../users_EKM_files_{sbf}sbf"
result_base_storage_path = "/home/sadeghi/Amirhossein/panTompkins/Results"
result_predictions_storage_path = f"{result_base_storage_path}/ReplayCheck/predictions"
result_historys_storage_path = f"{result_base_storage_path}/ReplayCheck/historys"
result_models_storage_path = f"{result_base_storage_path}/ReplayCheck/models"
dataset_path = users_ekm_dir

# Create the hierarchy if it doesn't exist
os.makedirs(result_predictions_storage_path, exist_ok=True)
os.makedirs(result_historys_storage_path, exist_ok=True)
os.makedirs(result_models_storage_path, exist_ok=True)

# Unzip the main dataset file
print("Unzipping the EKM dataset")
if not os.path.exists(unzip_dir):
    with zipfile.ZipFile(dataset_path_zip, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

# Unzip individual user EKM zip files
if not os.path.exists(users_ekm_dir):
    # Create directory for user EKM files if it doesn't exist
    os.makedirs(users_ekm_dir, exist_ok=True)

    user_zip_path = os.path.join(unzip_dir, "Users EKM zip")
    for _file in os.listdir(user_zip_path):
        file_name, _ = os.path.splitext(_file)
        file_dir = os.path.join(users_ekm_dir, file_name)
        
        os.makedirs(file_dir, exist_ok=True)
        
        with zipfile.ZipFile(os.path.join(user_zip_path, _file), 'r') as zip_ref:
            zip_ref.extractall(file_dir)

######################################################
# # Healthy EKMs check
# - Moving window checking

path = f"{dataset_path}/2005/x_lead"
EKMs = os.listdir(path)

def are_images_identical(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    return np.array_equal(np.array(img1), np.array(img2))

path_EKM1 = path + "/" + EKMs[0]
path_EKM2 = path + "/" + EKMs[1]

result = are_images_identical(path_EKM1, path_EKM2)
print("Images are identical:" if result else "Images are different.")

######################################################
# Creating dictionary of list of each user's EKMs

# Getting users id
users_id = []
dirs = os.listdir(dataset_path)
for _dir in dirs:
    user_id = _dir.split("_")[-1]
    users_id.append(user_id)

print(f"users amount: {len(users_id)}")

def loading_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

# Creating dict of counters for each randomly-chosen users
# of ecg 200 dataset
# Also creating dict of users' EKMs
ecg_200_users_EKM_amount_dict_ylead = {}
ecg_200_users_EKMs_dict_ylead = {}

for user in users_id:
  ecg_200_users_EKM_amount_dict_ylead[user] = 0
  ecg_200_users_EKMs_dict_ylead[user] = []

# Counting each user's EKMs in dataset and collecting EKMs of him/her
users_files = os.listdir(users_ekm_dir)

for _files in users_files:
    user_id = _files.split("_")[-1]
    if user_id in users_id:
      ecg_200_users_EKM_amount_dict_ylead[user_id] = len(os.listdir(f"{dataset_path}/{user_id}/y_lead"))
      ecg_200_users_EKMs_dict_ylead[user_id] = os.listdir(f"{dataset_path}/{user_id}/y_lead")

# Creating dict of counters for each randomly-chosen users
# of ecg 200 dataset
# Also creating dict of users' EKMs
ecg_200_users_EKM_amount_dict_xlead = {}
ecg_200_users_EKMs_dict_xlead = {}

for user in users_id:
  ecg_200_users_EKM_amount_dict_xlead[user] = 0
  ecg_200_users_EKMs_dict_xlead[user] = []

# Counting each user's EKMs in dataset and collecting EKMs of him/her
users_files = os.listdir(dataset_path)

for _files in users_files:
    user_id = _files.split("_")[-1]
    if user_id in users_id:
      ecg_200_users_EKM_amount_dict_xlead[user_id] = len(os.listdir(f"{dataset_path}/{user_id}/x_lead"))
      ecg_200_users_EKMs_dict_xlead[user_id] = os.listdir(f"{dataset_path}/{user_id}/x_lead")

# Creating dict of counters for each randomly-chosen users
# of ecg 200 dataset
# Also creating dict of users' EKMs
ecg_200_users_EKM_amount_dict_zlead = {}
ecg_200_users_EKMs_dict_zlead = {}

for user in users_id:
  ecg_200_users_EKM_amount_dict_zlead[user] = 0
  ecg_200_users_EKMs_dict_zlead[user] = []

# Counting each user's EKMs in dataset and collecting EKMs of him/her
users_files = os.listdir(dataset_path)

for _files in users_files:
    user_id = _files.split("_")[-1]
    if user_id in users_id:
      ecg_200_users_EKM_amount_dict_zlead[user_id] = len(os.listdir(f"{dataset_path}/{user_id}/z_lead"))
      ecg_200_users_EKMs_dict_zlead[user_id] = os.listdir(f"{dataset_path}/{user_id}/z_lead")

print(f"min ecg_200_users_EKM_amount_dict_xlead values: {min(ecg_200_users_EKM_amount_dict_xlead.values())}")
print(f"min ecg_200_users_EKM_amount_dict_ylead values: {min(ecg_200_users_EKM_amount_dict_ylead.values())}")
print(f"min ecg_200_users_EKM_amount_dict_zlead values: {min(ecg_200_users_EKM_amount_dict_zlead.values())}")

#########################################
# # Selecting one random user and EKMs of him and different users
# Note: Selecting **1000** EKMs for the user

# Function to extract the last number in the string
def extract_last_number(file_name):
    match = re.search(r'(\d+)-(\d+)\.png$', file_name)
    if match:
        return int(match.group(2))
    return None

def one_vs_all_EKMs(user_id):
    X  = []
    y = []

    N_each_user_ekms_amount = 1000

    # True labels
    true_tuples = []
    user_id_ekms_ylead = os.listdir(f"{dataset_path}/{user_id}/y_lead/")[:N_each_user_ekms_amount]
    user_id_ekms_xlead = os.listdir(f"{dataset_path}/{user_id}/x_lead/")[:N_each_user_ekms_amount]
    
    for ekm_index in range(N_each_user_ekms_amount):
        true_tuple = (f"{dataset_path}/{user_id}/y_lead/{user_id_ekms_ylead[ekm_index]}", f"{dataset_path}/{user_id}/x_lead/{user_id_ekms_xlead[ekm_index]}")
        true_tuples.append(true_tuple)

    X = X + true_tuples
    y = y + [1 for _ in range(len(true_tuples))]

    ## False labels
    false_tuples = []
    
    user_id_ekms_ylead = os.listdir(f"{dataset_path}/{user_id}/y_lead/")
    user_id_ekms_xlead = os.listdir(f"{dataset_path}/{user_id}/x_lead/")

    # Sort the array based on the last serial number
    user_id_ekms_ylead = sorted(user_id_ekms_ylead, key=extract_last_number)
    user_id_ekms_xlead = sorted(user_id_ekms_xlead, key=extract_last_number)

    for _ in range(N_each_user_ekms_amount):
        xlead_rand_ekm = random.randint(0, ecg_200_users_EKM_amount_dict_xlead[str(user_id)] - 2)
        ylead_rand_ekm = random.randint(xlead_rand_ekm + 1, ecg_200_users_EKM_amount_dict_ylead[str(user_id)] - 1)

        false_tuple = (f"{dataset_path}/{user_id}/y_lead/{user_id_ekms_ylead[ylead_rand_ekm]}", f"{dataset_path}/{user_id}/x_lead/{user_id_ekms_xlead[xlead_rand_ekm]}")
        false_tuples.append(false_tuple)

    X = X + false_tuples
    y = y + [0 for _ in range(len(false_tuples))]

    return X, y

#########################################
# Vectorizing the EKMs

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

from IPython.display import clear_output

def progress_bar(index, total_length, name_of_list):
    bar_length = 50

    # Calculate the percentage of completion
    percent_complete = (index / total_length) * 100

    # Clear the current cell's output
    clear_output(wait=True)

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

#########################################
# 10 fold validation
from sklearn.model_selection import StratifiedKFold
# Hyperparameters
input_shape = (20, 31, 3)
num_classes = 2
n_splits = 10

# Model and prepration of data for fitting them to model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, 
    confusion_matrix, f1_score, precision_score, recall_score, log_loss
)
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

# Custom layer to calculate the Euclidean distance between the outputs
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def build_siamese_model(input_shape, embeddingDim=54):
  # Specify the inputs for the feature extractor network
  inputs = Input(input_shape)

  # Defining network layers
  x = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  # x = Dropout(0.7)(x)

  # Add more convolutional layer
  x = Conv2D(64, (3, 3), activation='relu')(x)
  # x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.5)(x)

  x = Flatten()(x)
  x = Dense(128, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(x)
  x = Dense(64, activation='relu')(x)
  x = Dense(32, activation='relu')(x)
  x = Dense(16, activation='relu')(x)
  # x = Dense(8, activation='softmax')(x)
  x = Dense(8, activation='relu')(x)
  outputs = Dense(embeddingDim)(x)

  # build the model
  model = Model(inputs, outputs)

  return model

# Siamese Network
def create_siamese_network(input_shape):
    input_xlead = Input(shape=input_shape, name="x_lead_input")
    input_ylead = Input(shape=input_shape, name="y_lead_input")

    # Shared subnetwork
    shared_network = build_siamese_model(input_shape)

    # Encode x_lead and y_lead using the shared network
    encoded_xlead = shared_network(input_xlead)
    encoded_ylead = shared_network(input_ylead)

    # Use the Lambda layer to create a custom layer for the Euclidean distance
    distance = Lambda(euclidean_distance, output_shape=(1,))([encoded_xlead, encoded_ylead])

    # Creating last layer as dense layer with 1 neuron and sigmoid activation function
    outputs = Dense(1, activation="sigmoid")(distance)

    return Model(inputs=[input_xlead, input_ylead], outputs=outputs)

# Metrics calculation helper function with loss calculation
def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    loss = log_loss(y_true, y_prob)  # Loss calculation

    return accuracy, auc_roc, aupr, f1, precision, recall, loss, conf_matrix

def results_saver(history, y_pred, y_test, model, fold):
    # Saving the model
    model_name = f"{sbf}sbf_replayCheck_{phase_name}_fold{fold}.keras"
    model_saving_path = f"{result_models_storage_path}/{model_name}"
    model.save(model_saving_path)

    # Saving hisroty
    histoy_saving = f"{result_historys_storage_path}/{sbf}sbf_replayCheck_{phase_name}_fold{fold}.json"
    history_dict = history.history
    with open(histoy_saving, 'w') as f:
        json.dump(history_dict, f)

    # Convert y_test and y_pred to lists (if they are numpy arrays or other formats)
    y_test_list = y_test.tolist()  # Convert to list if it's not already
    y_pred_list = y_pred.tolist()  # Convert to list if it's not already
    data = {
        "y_test": y_test_list,
        "y_pred": y_pred_list
    }

    # Saving y_test, y_pred
    predictions_path = f"{result_predictions_storage_path}/y_test_and_y_pred_{sbf}sbf_replayCheck_{phase_name}_fold{fold}.json"
    with open(predictions_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Result Data saved!")

def fold_10_model(X_xlead, X_ylead, y, saving_results_flag):

    # Check if the results file exists
    n_splits = 10
    
    # Metrics to store performance
    folds_evaluation = {}

    for index in range(n_splits):
        folds_evaluation[index] = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_xlead, y)):

        print(f"Training Fold {fold + 1}/{n_splits}...")
        
        # Split data
        X_xlead_train, X_xlead_test = X_xlead[train_idx], X_xlead[test_idx]
        X_ylead_train, X_ylead_test = X_ylead[train_idx], X_ylead[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create and compile the model
        model = create_siamese_network(input_shape)
        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Set up early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=10,          # Stop training after 15 epochs of no improvement
            restore_best_weights=True  # Restore the best weights after training
        )

        # Train the model with early stopping
        history = model.fit(
            [X_xlead_train, X_ylead_train],
            y_train,
            batch_size=32,
            epochs=60,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stopping]  # Add early stopping callback here
        )

        # Evaluate the model
        y_prob = model.predict([X_xlead_test, X_ylead_test]).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        # Calculate metrics
        accuracy, auc_roc, aupr, f1, precision, recall, loss, conf_matrix = calculate_metrics(y_test, y_pred, y_prob)

        # Save results
        folds_evaluation[fold] = {
            "accuracy": accuracy,
            "auc_roc": auc_roc,
            "loss": loss,
            "aupr": aupr,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix
        }
        print(f"Fold {fold + 1} Results - Accuracy: {accuracy}, AUC-ROC: {auc_roc}, AUPR: {aupr}")

        if saving_results_flag == True:
            results_saver(history, y_pred, y_test, model, fold)
    return folds_evaluation


#########################################
# Ove vs. all (for all the users)
users_fold_evaluations_dict = {}

if os.path.exists(f"./{sbf}sbf_replayCheck_1000EKMs_Mono_siamese.json"):
    with open(f"./{sbf}sbf_replayCheck_1000EKMs_Mono_siamese.json", "r") as f:
        users_fold_evaluations_dict = json.load(f)

def saving_to_json(user_id, folds_evaluation):
    users_fold_evaluations_dict[user_id] = folds_evaluation
    # Save to a text file
    with open(f"./{sbf}sbf_replayCheck_1000EKMs_Mono_siamese.json", "w") as _file:
        json.dump(users_fold_evaluations_dict, _file, indent=4)

# Doing it all! running the model on each undividual and rest
_users_id = [_ for _ in users_id if _ not in list(users_fold_evaluations_dict.keys())]

saving_results_flag = True
# _users_id = ["1027"]
for _user_id in _users_id:
    X, y = one_vs_all_EKMs(_user_id)

    # Vectorizing EKMs
    X_xlead = [ekm[1] for ekm in X]
    X_xlead = np.array(X_xlead)
    X_xlead = vectorizing_list_of_ekms(X_xlead, "No.1 => x lead")

    X_ylead = [ekm[0] for ekm in X]
    X_ylead = np.array(X_ylead)
    X_ylead = vectorizing_list_of_ekms(X_ylead, "No.2 => y lead")

    y = np.array(y)

    print(f"X_xlead {X_xlead.shape}")
    print(f"X_ylead {X_ylead.shape}")
    print(f"y {y.shape}")

    folds_evaluation = fold_10_model(X_xlead, X_ylead, y, saving_results_flag)

    saving_to_json(_user_id, folds_evaluation)
    saving_results_flag = False