# CWT code!

# Modify your Python script to accept window indices and calculate CWT features per window

### Calculate freq and Amp. using CWT of each bin in Static and Dynamic X, Y, Z, and VEDBA ###

import numpy as np
import pywt
import pandas as pd
import sys

# Load data
df = pd.read_csv('Pup_Features.csv', low_memory=False) #change back to Pup_Features if using merged data! Pup_Features_Raw for just BB
# Show all rows with at least one NaN value
rows_with_na = df[df.isna().any(axis=1)]
print(f"Number of rows with NA: {rows_with_na.shape[0]}")
#print(rows_with_na)
print(rows_with_na.head())

#df = df.dropna()

fs = 25
nyquist = 0.5 * fs
scales = np.logspace(1, np.log10(nyquist), num=10)
wavelet = 'morl'

def calculate_peak_amplitude_frequency(signal):
    cwtmatr, freqs = pywt.cwt(signal, scales, wavelet)
    dominant_freq_indices = np.argmax(np.abs(cwtmatr), axis=0)
    dominant_freqs = freqs[dominant_freq_indices]
    peak_amplitudes = np.max(np.abs(cwtmatr), axis=0)
    return np.mean(dominant_freqs), np.mean(peak_amplitudes)

# Prepare columns
columns = ['Static_X', 'Static_Y', 'Static_Z', 'Dynamic_X', 'Dynamic_Y', 'Dynamic_Z', 'VEDBA']
features = []

for bin_num, group in df.groupby('bin'):
    feature_row = {'bin': bin_num}
    for axis in columns:
        freq, amp = calculate_peak_amplitude_frequency(group[axis].values)
        feature_row[f'{axis}_Peak_Frequency'] = freq
        feature_row[f'{axis}_Peak_Amplitude'] = amp
    features.append(feature_row)

# Convert to DataFrame and merge back
features_df = pd.DataFrame(features)
df = df.merge(features_df, on='bin')

# Display all columns and the first 100 rows before saving
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't truncate wide rows
pd.set_option('display.max_rows', 100)       # show all rows

print(df.head(100))  # Or df.iloc[:100]

df.to_csv('BB_final_features_cwt.csv', index=False) #change to what acceleroemter is used!


## Do this for adults too!! ##

##############################################################
## option 2 : better? finds single highest frequency instead of averaging dominant peaks
##############################################################

import numpy as np
import pywt
import pandas as pd

# Load data
df = pd.read_csv('Pup_Features.csv', low_memory=False) #change back to Pup_Features if using merged data! Pup_Features_Raw for just BB
# Show all rows with at least one NaN value
rows_with_na = df[df.isna().any(axis=1)]
print(f"Number of rows with NA: {rows_with_na.shape[0]}")
print(rows_with_na.head())
df = df.dropna()
# Store original shape for comparison
original_rows = df.shape[0]
print(f"Original dataframe shape: {df.shape}")

fs = 25  # Sampling frequency in Hz
min_freq = 0.1  # Minimum frequency of interest (Hz)
max_freq = fs/2  # Nyquist frequency (Hz)

# Create scales that correspond to frequencies of interest
# For older versions of PyWavelets without the frequencies module
frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num=50)
# Direct scale calculation for Morlet wavelet
central_frequency = 0.8125  # Central frequency for 'morl' wavelet
scales = central_frequency / frequencies
wavelet = 'morl'

columns = ['Static_X', 'Static_Y', 'Static_Z', 'Dynamic_X', 'Dynamic_Y', 'Dynamic_Z', 'VEDBA']

def calculate_peak_amplitude_frequency(signal):
    # Handle NaN values in the signal
    signal_clean = signal[~np.isnan(signal)]
    
    # Check if signal is too short for analysis
    if len(signal_clean) < 4:  # Need a minimum number of points
        return np.nan, np.nan
        
    # Compute CWT with our frequency-focused scales
    cwtmatr, freqs = pywt.cwt(signal_clean, scales, wavelet)
    
    # Convert scales back to corresponding frequencies
    freqs = central_frequency / scales
    
    # Compute the power spectrum (average power across time)
    power = np.mean(np.abs(cwtmatr)**2, axis=1)
    
    # Find the peak frequency and its amplitude
    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    peak_amplitude = np.sqrt(power[peak_idx])  # Back to amplitude scale
    
    return peak_freq, peak_amplitude

# Initialize features list
features = []

# Calculate features by bin
for bin_num, group in df.groupby('bin'):
    feature_row = {'bin': bin_num}
    for axis in columns:
        if axis in group.columns:
            freq, amp = calculate_peak_amplitude_frequency(group[axis].values)
            feature_row[f'{axis}_Peak_Frequency'] = freq
            feature_row[f'{axis}_Peak_Amplitude'] = amp
        else:
            print(f"Warning: Column {axis} not found in dataframe")
            feature_row[f'{axis}_Peak_Frequency'] = np.nan
            feature_row[f'{axis}_Peak_Amplitude'] = np.nan
    features.append(feature_row)

# Convert to DataFrame and merge back
features_df = pd.DataFrame(features)
result = df.merge(features_df, on='bin', how='left')

# Print shape information
print(f"Features dataframe shape: {features_df.shape}")
print(f"Final result shape: {result.shape}")

if result.shape[0] != original_rows:
    print(f"⚠️ WARNING: Row count changed from {original_rows} to {result.shape[0]}!")

# Display all columns and the first 100 rows before saving
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't truncate wide rows
pd.set_option('display.max_rows', 100)       # show all rows

print(result.head(100))  # Or df.iloc[:100]

result.to_csv('final_features_cwt.csv', index=False) #change to what acceleroemter is used!