########################################
#       Installation and Imports
########################################

from ishneholterlib import Holter
import biosignalsnotebooks as bsnb
from scipy.signal import detrend
import os
from user_ekms_functions import user_ekm_no_1_dataset
import multiprocessing
import logging

########################################
#           Initial variables
########################################
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')

dataset_path = "../../../ECG200"
users_files = os.listdir(dataset_path)
users_files.remove("clinicalData-selected")

# Initialization of dataset extracting processing
dataset_name = "main_ekm_dataset"
base_ekms_path = f'EKM_dataset'

lead_names_dict = {
    1: "x_lead",
    2: "y_lead",
    3: "z_lead"
}

if not os.path.isdir('./Users EKM zip'):
  os.mkdir("./Users EKM zip")

########################################
#       Reading files and playground!
########################################

# Getting .ecg files of users
users_ecg_files = []
for _file in users_files:
  f_extention = _file.split(".")[1]
  if f_extention == "ecg":
    users_ecg_files.append(_file)

# Removing users that their EKMs have been extracted
user_ekms_zip_files = os.listdir("./Users EKM zip")
list_of_ekm_extracted_users = []

for _file in user_ekms_zip_files:
  try:
    extention = _file.split(".")[1]
    if extention == "zip":
      user_id = _file.split(".")[0]
      list_of_ekm_extracted_users.append(user_id)
  except:
    pass

for usr in list_of_ekm_extracted_users:
  try:
    users_ecg_files.remove(usr + ".ecg")
  except:
    print(f"User No. {usr} already been removed.")

########################################
#            Multi processing
########################################

# Specify the number of processes in the pool
num_processes = multiprocessing.cpu_count()
# num_processes = 1

# Creating slices of users' ecg file for multiprocessing
# slices_size = num_processes
# number_of_complete_slices = len(users_ecg_files)//slices_size
# users_ecg_files_chunks = [users_ecg_files[_ * slices_size: (_+1) * slices_size] for _ in range(number_of_complete_slices)]

# if number_of_complete_slices * slices_size != len(users_ecg_files):
#     users_ecg_files_chunks.append(users_ecg_files[number_of_complete_slices * slices_size:])

# # temp
# users_ecg_files_chunks = ["9005.ecg", "2007.ecg", "6030.ecg", "9062.ecg"]
# # temp

def processing_ecg_files(users_ecg_files_chunk):
    with multiprocessing.Manager() as manager:
        # Create a shared counter
        shared_counter = manager.Value('i', 0)

        # Create a lock from the manager
        lock = manager.Lock()

        # Create a pool of processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Pass the shared counter, lock, and total number of elements to the worker function
            pool.starmap(user_ekm_no_1_dataset, [(user, shared_counter, lock, len(users_ecg_files_chunk)) for user in users_ecg_files_chunk])

# for users_ecg_files_chunk in users_ecg_files_chunks:
#     # print(users_ecg_files_chunk)
#     processing_ecg_files(users_ecg_files_chunk)
#     break

# # temp
# processing_ecg_files(users_ecg_files_chunks)
# # temp

processing_ecg_files(users_ecg_files)

# Print final progress
print("Processing complete.")