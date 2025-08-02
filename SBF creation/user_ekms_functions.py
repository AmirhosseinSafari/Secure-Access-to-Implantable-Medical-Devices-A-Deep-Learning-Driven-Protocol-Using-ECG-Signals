# Getting EKMs of each lead of a user from .ecg files
# Then saving the user's EKM dataset as zip

########################################
#       Installation and Imports
########################################

import shutil
import os
import numpy as np
from ishneholterlib import Holter
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import biosignalsnotebooks as bsnb
from scipy.signal import detrend
import seaborn as sns
import neurokit2 as nk
import logging

########################################
#           Initial variables
########################################

dataset_path = "../../../ECG200"
dataset_name = "no1_ekm_dataset"
base_ekms_path = f'EKM_dataset'
lead_names_dict = {
    1: "x_lead",
    2: "y_lead",
    3: "z_lead"
}

# bpf = 4
sbf = 6

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
#    signals, info = nk.ecg_peaks(filtered_signal, sampling_rate=fs, method=Rpeak_method)
#    rpeaks = info["ECG_R_Peaks"]

   return filtered_signal

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

############################################################################
#               No_1 EKMs          
############################################################################

def electrocardiomatrix_no_1(filtered_ecg, init_window, sampling_rate, window_size):
    fs = sampling_rate
    window_signal_sample_size = window_size * fs
    each_line_ekm_size = 1 # seconds
    each_line_ekm_sample_signal_size = each_line_ekm_size * fs
    all_segments = []

    for ekm_line in range(window_size):
      segment = filtered_ecg[init_window + (ekm_line * each_line_ekm_sample_signal_size): \
                  init_window + ((ekm_line+1) * each_line_ekm_sample_signal_size)]
      all_segments.append(segment)

    ecm = all_segments

    return ecm

# Labeling is in this way that, prelast element of EKM's name is the user's id,
# and the last element is the number of the EKM for that user
def save_ecm_no_1(dataset_name, path, key, i):
    # Saving EKMs in format of {path}/_NumberOfbpfsInAEKM_bpf-ekm-{key=user id}-{i=serial Number}
    plt.savefig(f"{path}/{sbf}sbf-ekm-{dataset_name}-{key}-{str(i)}",bbox_inches='tight', pad_inches=0)


def little_ekm_no_1_dataset(lead_data, sampling_rate, dataset_name, ekms_path, key, sbf):
    logging.info(f"Processing user {key}. Lead: {ekms_path.split('/')[1]}. ECG Preprocessing.")

    # print("  .Preprocessing the signal")
    filtered_ecg = process_ecg(lead_data , sampling_rate)

    # print("  .Getting detrend_signal, norm_ecg, distance")
    detrend_signal = detrend(filtered_ecg)
    norm_ecg = normalize(detrend_signal)
    # distance = peak_distance(peaks)

    ekms_counter, init_window = 0, 0
    total_ecms = 3000

    fig_width_px = 33
    fig_height_px = 21

    window_size = sbf # seconds
    init_window = 0
    
    logging.info(f"Processing user {key}. Lead: {ekms_path.split('/')[1]}. EKMs extracting.")

    # print("  .Getting EKMs")
    while(ekms_counter<total_ecms):
      if (init_window >= len(norm_ecg) or  init_window + (sampling_rate * window_size) >= len(norm_ecg)): break
      # electrocardiomatrix_no_1
      #   - Inputs: (filtered_ecg, init_window, sampling_rate, window_size)
      ecm = electrocardiomatrix_no_1(norm_ecg, init_window, sampling_rate, window_size)
      if ecm is None: break
    #   distance = int(distance)
      norm_ecm = normalize(ecm)

      fig = plt.figure(num=1, clear=True, figsize=(fig_width_px / 80, fig_height_px / 80))
      ax = fig.add_subplot()
      ax.axis('off')

      sns.heatmap(norm_ecm, xticklabels=False, yticklabels=False, cbar=False)
      # plt.tight_layout()

      save_ecm_no_1(dataset_name, ekms_path, key, ekms_counter)
      init_window += (sampling_rate * window_size)
      ekms_counter += 1
      # break

      if ekms_counter % 100 == 0:
        logging.info(f"Processing user {key}. Lead: {ekms_path.split('/')[1]}. {ekms_counter} EKMs Extracted.")


###################
#  Main process
###################

def user_ekm_no_1_dataset(ecg_file, shared_counter_, lock, total_elements):
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

    # Log the current progress
    logging.info(f"Processing user {user_id}. Progress: {shared_counter_.value}/{total_elements}")

    for _, lead_data in enumerate(user_leads_signals):
        # name_of_file = ecg_file + ": " + lead_names_dict[_ + 1]
        # pretier_print("begin", int(user_id), name_of_file)

        logging.info(f"Processing user {user_id}. Lead: {lead_names_dict[_ + 1]}")

        lead_path = f"{base_ekms_path}_{user_id}/{lead_names_dict[_ + 1]}"
        little_ekm_no_1_dataset(lead_data.data, sampling_rate, dataset_name, lead_path, user_id, sbf)

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
