# Getting EKMs of each lead of a user from .ecg files
# Then saving the user's EKM dataset as zip

########################################
#       Installation and Imports
########################################

import shutil
import os
import numpy as np
from ishneholterlib import Holter
import matplotlib.pyplot as plt
import biosignalsnotebooks as bsnb
from scipy.signal import detrend
import seaborn as sns
import json
# import biosppy
import neurokit2 as nk
import logging

########################################
#           Initial variables
########################################

dataset_path = "../../../../dataset/ECG_200"
dataset_name = "bpf_recording_signal_length_ekm_dataset"
base_ekms_path = f'EKM_dataset'
base_rpeaks_path = f'Rpeaks_dataset'
base_all_rpeaks_path = f'All_Rpeaks_dataset'
base_rpeaks_failure_path = f'Rpeaks_failure_dataset'
base_distance = f'R_R_distance_dataset'
test_big_EKM_view_path = "./big_EKMs_view_test"

lead_names_dict = {
    1: "x_lead",
    2: "y_lead",
    3: "z_lead"
}

bpf = 5
recording_signal_length = 6
Rpeak_method = "pantompkins1985"

########################################
#               Functions
########################################

###################
#    Preprocess
###################
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

def process_ecg(unfiltered_ecg, fs):
   # ECG Filtering (Bandpass between 5 and 15 Hz)
   filtered_signal = bsnb.detect._ecg_band_pass_filter(unfiltered_ecg, fs)
   signals, info = nk.ecg_peaks(filtered_signal, sampling_rate=fs, method=Rpeak_method)
   rpeaks = info["ECG_R_Peaks"]

   return rpeaks, filtered_signal

###################
# Storing into files
###################

def write_dict_to_file(my_dict, file_path):
    """
    Writes a dictionary to a specified file in JSON format.
    """
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    try:
        with open(file_path, 'w') as file:
            json.dump(my_dict, file, indent=4, default=convert_ndarray)
    except Exception as e:
        print(f"An error occurred: {e}")

# Labeling is in this way that, prelast element of EKM's name is the user's id,
# and the last element is the number of the EKM for that user
def save_ecm(dataset_name, path, key, i):
    # Saving EKMs in format of {path}/_NumberOfbpfsInAEKM_bpf-ekm-{key=user id}-{i=serial Number}
    plt.savefig(f"{path}/{bpf}bpf-ekm-{dataset_name}-{key}-{str(i)}",bbox_inches='tight', pad_inches=0)

def big_EKM_view_test(norm_ecm, test_path, user_id, ekms_counter):
    # Creating dir if doesn't exist
    base_user_saving_path = f"{test_path}/{user_id}"
    if not os.path.isdir(base_user_saving_path):
      os.makedirs(base_user_saving_path)

    # Create a large figure for the heatmap
    plt.figure(figsize=(16, 12))
    # Plot heatmap with desired parameters
    sns.heatmap(norm_ecm, xticklabels=False, yticklabels=False, cbar=False)
    # Save the heatmap image to the specified path with high resolution
    saving_path = f"{base_user_saving_path}/{ekms_counter}"
    plt.savefig(saving_path, dpi=300, bbox_inches='tight', pad_inches=0)
    # Close the plot to free memory
    plt.close()

def save_distance_bpf(key, distance, dataset_name, r_r_distance_path):
    distance_info = {
       "user_id": key,
       "R-R_distance": distance
    }
    saving_path = f"{r_r_distance_path}/{bpf}bpf-R-R-distance-{dataset_name}-{key}"
    write_dict_to_file(distance_info, saving_path)

def save_rpeaks_bpf(rpeaks, ekms_counter, dataset_name, rpeaks_path, key):
    r_peak_dict = {
       "user_id": key,
       "rpeaks": rpeaks,
       "EKM number" : ekms_counter,
    }
    saving_path = f"{rpeaks_path}/{bpf}bpf-rpeaks-{dataset_name}-{key}-{ekms_counter}"
    write_dict_to_file(r_peak_dict, saving_path)

def save_all_r_peaks_bpf(key, peaks, dataset_name, all_rpeaks_path):
    all_rpeaks_info = {
       "user_id": key,
       "all_rpeaks": peaks
    }
    saving_path = f"{all_rpeaks_path}/{bpf}bpf-all-Rpeaks-{dataset_name}-{key}"
    write_dict_to_file(all_rpeaks_info, saving_path)

def save_rpeaks_failure_bpf_recording_signal_length(rpeaks, boundaris, dataset_name, path, key, i):
    # Saving the rpeaks coresponding to their EKMs
    r_peak_dict = {
        "rpeaks" : rpeaks,
        "boundaris" : boundaris
    }
    saving_path = f"{path}/{bpf}bpf-rpeaks-recording-signal-length-failure-{dataset_name}-{key}-{str(i)}"
    write_dict_to_file(r_peak_dict, saving_path)

###################
#  EKM creation
###################

def electrocardiomatrix_bpf_in_recording_signal_length(distance, r_peaks, filtered_ecg, EKM_counter, sampling_rate):
    '''
    Creating bpf based EKMs in recording signal length
    '''
    # Initilization
    init_seg = int(0.2 * distance)
    fin_seg = int(1.5 * distance)
    peaks_window = bpf-1
    one_EKM_signal_size = recording_signal_length * sampling_rate

    # Definging the lower and upper bound of the 6-second EKM
    lower_bound = one_EKM_signal_size * (EKM_counter)
    upper_bound = one_EKM_signal_size * (EKM_counter + 1)

    # Getting r peaks of one EKM (bpf in signal boundary length)
    r_peaks_one_EKM = r_peaks[(r_peaks >= lower_bound) & (r_peaks <= upper_bound)]
    
    # Checking if there are enough r_peaks in the signal or not
    defficient_peaks_flag = False
    if len(r_peaks_one_EKM) >= peaks_window:
        r_peaks_one_EKM = r_peaks_one_EKM[0:peaks_window]
    else:
        defficient_peaks_flag = True

    # Returning if the EKM have not enough R peaks
    if defficient_peaks_flag == True:
        ekm = "Not enough peaks"
        return ekm, r_peaks_one_EKM, [lower_bound, upper_bound]
    
    # Getting the segments
    all_segments = []
    for peak in r_peaks_one_EKM:
        segment = filtered_ecg[peak - init_seg : peak + fin_seg]
        all_segments.append(segment)

    norm_all_segments = normalize(all_segments)
    
    return norm_all_segments, r_peaks_one_EKM, [lower_bound, upper_bound]

def little_ekm_dataset(lead_data, 
                       sampling_rate, 
                       dataset_name, 
                       ekms_path, 
                       key, 
                       rpeaks_path, 
                       r_r_distance_path, 
                       all_rpeaks_path,
                       rpeaks_failure_path,
                       bpf):
  logging.info(f"Processing user {key}. Lead: {ekms_path.split('/')[1]}. ECG Preprocessing.")

  total_ecms = 3000

  # Getting the needed signal
  lead_data = lead_data[:total_ecms * sampling_rate * recording_signal_length * 2]

  # print("  .Preprocessing the signal")
  peaks, filtered_ecg = process_ecg(lead_data , sampling_rate)

  # print("  .Getting detrend_signal, norm_ecg, distance")
  detrend_signal = detrend(filtered_ecg)
  norm_ecg = normalize(detrend_signal)
  distance = peak_distance(peaks)

  # Saving the R-R distance of each lead of the user
  save_distance_bpf(key, distance, dataset_name, r_r_distance_path)

  # Saving the R-R distance of each lead of the user
  save_all_r_peaks_bpf(key, peaks, dataset_name, all_rpeaks_path)

  ekms_counter, init_window = 0, 0
  boundary_counter = 0
  #total_ecms = 3000

  fig_width_px = 33
  fig_height_px = 21

  window_size = recording_signal_length # seconds

  logging.info(f"Processing user {key}. Lead: {ekms_path.split('/')[1]}. EKMs extracting.")

  # print("  .Getting EKMs")
  while(ekms_counter<total_ecms):
    if (init_window >= len(norm_ecg) or  init_window + (sampling_rate * window_size) >= len(norm_ecg)): break
    # electrocardiomatrix_bpf_in_recording_signal_length(distance, r_peaks, filtered_ecg, EKM_counter, sampling_rate) 
    ecm, rpeaks, boundaris = electrocardiomatrix_bpf_in_recording_signal_length(distance, peaks, norm_ecg, boundary_counter, sampling_rate)
    if ecm is None: break
    if isinstance(ecm, str):
        if ecm == "Not enough peaks": 
            save_rpeaks_failure_bpf_recording_signal_length(rpeaks, boundaris, dataset_name, rpeaks_failure_path, key, boundary_counter)
            boundary_counter += 1
            init_window += (sampling_rate * window_size)
            continue
    distance = int(distance)
    norm_ecm = normalize(ecm)

    #big_EKM_view_test(norm_ecm, test_big_EKM_view_path, key, ekms_counter)

    fig = plt.figure(num=1, clear=True, figsize=(fig_width_px / 80, fig_height_px / 80))
    ax = fig.add_subplot()
    ax.axis('off')

    sns.heatmap(norm_ecm, xticklabels=False, yticklabels=False, cbar=False)
    # plt.tight_layout()

    save_ecm(dataset_name, ekms_path, key, ekms_counter)
    save_rpeaks_bpf(rpeaks, ekms_counter, dataset_name, rpeaks_path, key)

    ekms_counter += 1
    boundary_counter += 1
    init_window += (sampling_rate * window_size)

    if ekms_counter % 100 == 0:
      logging.info(f"Processing user {key}. Lead: {ekms_path.split('/')[1]}. {ekms_counter} EKMs Extracted.")


###################
# Directory managment
###################

def user_EKMs_dir_creator(user_id):
  # Removing previous EKM dir and creating new one
  try:
    shutil.rmtree(f"./{base_ekms_path}_{user_id}")
  except OSError as e:
    pass

  try:
    os.mkdir(f"./{base_ekms_path}_{user_id}")
    os.makedirs(f"./{base_ekms_path}_{user_id}/x_lead")
    os.makedirs(f"./{base_ekms_path}_{user_id}/y_lead")
    os.makedirs(f"./{base_ekms_path}_{user_id}/z_lead")
  except OSError as e:
    print(f"Error: {e}")

def user_r_peaks_of_EKMs_dir_creator(user_id):
  # Removing previous EKM dir and creating new one
  try:
    shutil.rmtree(f"./{base_rpeaks_path}_{user_id}")
  except OSError as e:
    pass

  try:
    os.mkdir(f"./{base_rpeaks_path}_{user_id}")
    os.makedirs(f"./{base_rpeaks_path}_{user_id}/x_lead")
    os.makedirs(f"./{base_rpeaks_path}_{user_id}/y_lead")
    os.makedirs(f"./{base_rpeaks_path}_{user_id}/z_lead")
  except OSError as e:
    print(f"Error: {e}")

def user_R_R_distance_dir_creator(user_id):
  # Removing previous EKM dir and creating new one
  try:
    shutil.rmtree(f"./{base_distance}_{user_id}")
  except OSError as e:
    pass

  try:
    os.mkdir(f"./{base_distance}_{user_id}")
    os.makedirs(f"./{base_distance}_{user_id}/x_lead")
    os.makedirs(f"./{base_distance}_{user_id}/y_lead")
    os.makedirs(f"./{base_distance}_{user_id}/z_lead")
  except OSError as e:
    print(f"Error: {e}")

def user_all_r_peaks_dir_creator(user_id):
  # Removing previous EKM dir and creating new one
  try:
    shutil.rmtree(f"./{base_all_rpeaks_path}_{user_id}")
  except OSError as e:
    pass

  try:
    os.mkdir(f"./{base_all_rpeaks_path}_{user_id}")
    os.makedirs(f"./{base_all_rpeaks_path}_{user_id}/x_lead")
    os.makedirs(f"./{base_all_rpeaks_path}_{user_id}/y_lead")
    os.makedirs(f"./{base_all_rpeaks_path}_{user_id}/z_lead")
  except OSError as e:
    print(f"Error: {e}")

def user_r_peaks_of_failure_EKMs_dir_creator(user_id):
  # Removing previous EKM dir and creating new one
  try:
    shutil.rmtree(f"./{base_rpeaks_failure_path}_{user_id}")
  except OSError as e:
    pass

  try:
    os.mkdir(f"./{base_rpeaks_failure_path}_{user_id}")
    os.makedirs(f"./{base_rpeaks_failure_path}_{user_id}/x_lead")
    os.makedirs(f"./{base_rpeaks_failure_path}_{user_id}/y_lead")
    os.makedirs(f"./{base_rpeaks_failure_path}_{user_id}/z_lead")
  except OSError as e:
    print(f"Error: {e}")

###################
#  Main process
###################

def user_ekm_dataset(ecg_file, shared_counter_, lock, total_elements):
    # print(f"\n{ecg_file}")

    ecg_file_path = dataset_path + "/" + ecg_file
    user_leads_all_data = Holter(ecg_file_path)
    user_leads_all_data.load_data()

    x_lead = user_leads_all_data.lead[0]
    y_lead = user_leads_all_data.lead[1]
    z_lead = user_leads_all_data.lead[2]

    user_leads_signals = [x_lead, y_lead, z_lead]
    user_id = ecg_file.split(".")[0]
    sampling_rate = user_leads_all_data.sr

    user_EKMs_dir_creator(user_id)
    user_r_peaks_of_EKMs_dir_creator(user_id)
    user_R_R_distance_dir_creator(user_id)
    user_all_r_peaks_dir_creator(user_id)
    user_r_peaks_of_failure_EKMs_dir_creator(user_id)

    # Log the current progress
    logging.info(f"Processing user {user_id}. Progress: {shared_counter_.value}/{total_elements}")

    for _, lead_data in enumerate(user_leads_signals):
        # name_of_file = ecg_file + ": " + lead_names_dict[_ + 1]
        # pretier_print("begin", int(user_id), name_of_file)

        logging.info(f"Processing user {user_id}. Lead: {lead_names_dict[_ + 1]}")

        lead_path = f"{base_ekms_path}_{user_id}/{lead_names_dict[_ + 1]}"
        rpeaks_path = f"{base_rpeaks_path}_{user_id}/{lead_names_dict[_ + 1]}"
        r_r_distance_path = f"{base_distance}_{user_id}/{lead_names_dict[_ + 1]}"
        all_rpeaks_path = f"{base_all_rpeaks_path}_{user_id}/{lead_names_dict[_ + 1]}"
        rpeaks_failure_path = f"{base_rpeaks_failure_path}_{user_id}/{lead_names_dict[_ + 1]}"

        little_ekm_dataset(lead_data.data, sampling_rate, dataset_name, lead_path, user_id, rpeaks_path, r_r_distance_path, all_rpeaks_path, rpeaks_failure_path, bpf)

        # pretier_print("end", int(user_id), ecg_file)

    # logging the zipping and transfering
    logging.info(f"Processing user {user_id}. Progress: zipping and transfering")

    shutil.make_archive(user_id, format='zip', root_dir=f'./EKM_dataset_{user_id}')
    source_file_path = f"./{user_id}.zip"
    destination_directory = f"./Users EKM zip/{user_id}.zip"
    shutil.move(source_file_path, destination_directory)

    # Update the shared counter to track progress
    with lock:
        shared_counter_.value += 1
        processed_elements = shared_counter_.value
        percentage_completion = (processed_elements / total_elements) * 100
        print(f"Processed {processed_elements}/{total_elements} elements ({percentage_completion:.2f}% complete)")
        # writing_output(f"Processed {processed_elements}/{total_elements} elements ({percentage_completion:.2f}% complete)")
