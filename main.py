"""
ECG and HRV Analysis Script
---------------------------
This script processes raw ECG waveform data to extract R-peaks and compute 
time-domain and frequency-domain heart rate variability (HRV) features.

It combines two open-source Python packages — [NeuroKit2](https://github.com/neuropsychology/NeuroKit) 
and [HRV-analysis](https://aura-healthcare.github.io/hrv-analysis/) — to provide a comprehensive 
pipeline for ECG signal processing and HRV feature extraction.

Author: Kian Godhwani   
Date: 2025-04-01

Instructions:
- Replace placeholder `file_path` values with the actual path to your dataset.
- Ensure the dataset contains an 'EcgWaveform' column.

Required packages:
- pandas
- numpy
- neurokit2
- hrvanalysis

See README.md for detailed usage.
"""


import pandas as pd
import neurokit2 as nk


# Step 1: Import libraries
print("Step 1: Importing libraries...")
print("Imported neurokit2 and pandas successfully!")


# Step 2: Load your ECG dataset
print("Step 2: Loading the ECG dataset...")
file_path = "  # Replace with the path to your dataset
try:
   data = pd.read_csv(file_path)
   print(f"Loaded dataset successfully! Dataset contains {len(data)} rows.")
except FileNotFoundError:
   print(f"Error: File not found at {file_path}. Please check the file path.")
   exit()


# Step 3: Extract the ECG signal
print("Step 3: Extracting ECG signal...")
if 'EcgWaveform' in data.columns:
   ecg_signal = data['EcgWaveform']  # Replace with the column name for your ECG signal
   print("ECG signal extracted successfully!")
else:
   print("Error: Column 'EcgWaveform' not found in the dataset. Please check your column names.")
   print(f"Available columns: {data.columns.tolist()}")
   exit()


# Step 4: Process the ECG signal using the neurokit pipeline
print("Step 4: Processing the ECG signal...")
try:
   signals, rpeaks = nk.ecg_process(ecg_signal, sampling_rate=250, method='pantompkins1985')
   print("ECG signal processed successfully!")
   print(f"Number of R-peaks detected: {len(rpeaks['ECG_R_Peaks'])}")
except Exception as e:
   print(f"Error during ECG processing: {e}")
   exit()

#If the raw ECG file is too big, you can alternatively use the code below

import pandas as pd
import neurokit2 as nk
import numpy as np


# Step 1: Import libraries
print("Step 1: Importing libraries...")
print("Imported neurokit2 and pandas successfully!")


# Step 2: Load your ECG dataset
print("Step 2: Loading the ECG dataset...")
file_path =   # Replace with your dataset path


try:
   data = pd.read_csv(file_path)
   print(f"Loaded dataset successfully! Dataset contains {len(data)} rows.")
except FileNotFoundError:
   print(f"Error: File not found at {file_path}. Please check the file path.")
   exit()


# Step 3: Extract the ECG signal
print("Step 3: Extracting ECG signal...")
if 'EcgWaveform' in data.columns:
   ecg_signal = data['EcgWaveform'].values  # Convert to NumPy array for efficient slicing
   print("ECG signal extracted successfully!")
else:
   print("Error: Column 'EcgWaveform' not found in the dataset. Please check your column names.")
   print(f"Available columns: {data.columns.tolist()}")
   exit()


# Step 4: Process ECG in Two Halves to Avoid Memory Issues
print("Step 4: Processing the ECG signal in two halves...")
half_length = len(ecg_signal) // 2  # Find the middle index


rpeaks_combined = []  # List to store combined R-peaks


for i, segment in enumerate([ecg_signal[:half_length], ecg_signal[half_length:]]):
   print(f"Processing half {i + 1} of the ECG data...")


   try:
       signals, rpeaks = nk.ecg_process(segment, sampling_rate=250, method='pantompkins1985')
       rpeaks_adjusted = rpeaks["ECG_R_Peaks"] + (i * half_length)  # Adjust indices for second half
       rpeaks_combined.extend(rpeaks_adjusted)
       print(f"Half {i + 1}: {len(rpeaks_adjusted)} R-peaks detected.")
   except Exception as e:
       print(f"Error during ECG processing in half {i + 1}: {e}")
       exit()


# Step 5: Return full R-peaks array
print(f"Total R-peaks detected: {len(rpeaks_combined)}")


#Now use that array to calculate the HRV domains


import numpy as np
from hrvanalysis.preprocessing import interpolate_nan_values, get_nn_intervals
from hrvanalysis.extract_features import get_time_domain_features, get_frequency_domain_features


# Step 1: Extract the R-peaks indices from the dictionary
r_peaks_indices = rpeaks_combined  # Use the full list of R-peaks from both halves


# Step 2: Convert R-peak indices to R-R intervals (in milliseconds)
sampling_rate = 250  # Define your ECG sampling rate
rri = np.diff(r_peaks_indices) * (1000 / sampling_rate)  # Convert sample indices to milliseconds
print(f"Extracted {len(rri)} R-R intervals.")


# Step 3: Preprocess R-R intervals
print("Step 3: Preprocessing R-R intervals...")
rri_interpolated = interpolate_nan_values(rri, interpolation_method="linear")


# Step 4: Clean R-R intervals (remove ectopic beats, artifacts) to get NN intervals
nn_intervals = get_nn_intervals(
   rri_interpolated,
   low_rri=300,  # Set plausible RRI range in ms
   high_rri=2000,
   interpolation_method="linear",
   ectopic_beats_removal_method="kamath",
   verbose=True
)
print(f"Obtained {len(nn_intervals)} valid NN intervals after preprocessing.")


# Step 5: Compute Time-Domain HRV Features
print("Step 5: Computing time-domain HRV features...")
time_domain_features = get_time_domain_features(nn_intervals)
print("Time-Domain HRV Features:")
print(time_domain_features)


# Step 6: Compute Frequency-Domain HRV Features
print("Step 6: Computing frequency-domain HRV features...")
freq_domain_features = get_frequency_domain_features(
   nn_intervals,
   method="welch",  # Welch's FFT method for spectral estimation
   sampling_frequency=4,  # Common HRV resampling frequency
   interpolation_method="linear"
)
print("Frequency-Domain HRV Features:")
print(freq_domain_features)


#If getting too many NAN values for HRV analysis


import numpy as np
from hrvanalysis.preprocessing import interpolate_nan_values, get_nn_intervals
from hrvanalysis.extract_features import get_time_domain_features, get_frequency_domain_features


# Step 1: Extract the R-peaks indices from the dictionary
r_peaks_indices = rpeaks_combined


# Step 2: Convert R-peaks to R-R intervals (in milliseconds)
sampling_rate = 250
rri = np.diff(r_peaks_indices) * (1000 / sampling_rate)


# **Check if R-R intervals are empty or full of NaNs**
if len(rri) == 0 or np.isnan(rri).all():
   print("Error: R-R intervals are empty or contain only NaNs.")
   exit()


print(f"Extracted {len(rri)} R-R intervals.")


# Step 3: Preprocess R-R intervals
print("Step 3: Preprocessing R-R intervals...")
rri_interpolated = interpolate_nan_values(rri, interpolation_method="linear")


# **Check if interpolation created NaNs**
if np.isnan(rri_interpolated).all():
   print("Error: Interpolated R-R intervals contain only NaNs.")
   exit()


# Step 4: Clean R-R intervals to get NN intervals
nn_intervals = get_nn_intervals(
   rri_interpolated,
   low_rri=400,  # Adjusted threshold
   high_rri=1800,  # Adjusted threshold
   interpolation_method="linear",
   ectopic_beats_removal_method="kamath",
   verbose=True
)


# **Check if NN intervals are empty or NaN**
if len(nn_intervals) == 0 or np.isnan(nn_intervals).all():
   print("Error: NN intervals are empty or contain only NaNs. Cannot compute HRV.")
   exit()


# Remove NaNs from NN intervals before HRV analysis
nn_intervals = [x for x in nn_intervals if not np.isnan(x)]


# Step 5: Compute Time-Domain HRV Features
print("Step 5: Computing time-domain HRV features...")
time_domain_features = get_time_domain_features(nn_intervals)
print("Time-Domain HRV Features:")
print(time_domain_features)


# Step 6: Compute Frequency-Domain HRV Features
print("Step 6: Computing frequency-domain HRV features...")
freq_domain_features = get_frequency_domain_features(
   nn_intervals,
   method="welch",
   sampling_frequency=4,
   interpolation_method="linear"
)
print("Frequency-Domain HRV Features:")
print(freq_domain_features)
