import numpy as np
import pywt
import pandas as pd
import os
import matplotlib.pyplot as plt

# -------- CONFIG --------
input_dir = "/home/dio3/williamslab/SarahKerr/Processed_Features/"  # Your feature .csv files
output_dir = "/home/dio3/williamslab/SarahKerr/CWT_Features/"        # Save files here
os.makedirs(output_dir, exist_ok=True)

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
    # Check if signal is too short for analysis
    if len(signal) < 4:  # Need a minimum number of points
        return np.nan, np.nan
        
    # Compute CWT with our frequency-focused scales
    cwtmatr, freqs = pywt.cwt(signal, scales, wavelet)
    
    # Convert scales back to corresponding frequencies
    freqs = central_frequency / scales
    
    # Compute the power spectrum (average power across time)
    power = np.mean(np.abs(cwtmatr)**2, axis=1)
    
    # Find the peak frequency and its amplitude
    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    peak_amplitude = np.sqrt(power[peak_idx])  # Back to amplitude scale
    
    return peak_freq, peak_amplitude

# def plot_cwt_spectrum(signal, filename, axis_name, bin_num):
#     """Optional function to visualize the CWT spectrum"""
#     if len(signal) < 4:  # Need minimum data points
#         return
        
#     plt.figure(figsize=(12, 8))
    
#     # Compute CWT
#     cwtmatr, _ = pywt.cwt(signal, scales, wavelet)
    
#     # Convert scales to frequencies
#     freqs = central_frequency / scales
    
#     power = np.mean(np.abs(cwtmatr)**2, axis=1)
    
#     # Plot the average power spectrum
#     plt.subplot(211)
#     plt.plot(freqs, power)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Power')
#     plt.title(f'CWT Power Spectrum - {axis_name} - Bin {bin_num}')
#     plt.xscale('log')
    
#     # Plot the scalogram
#     plt.subplot(212)
#     plt.imshow(np.abs(cwtmatr), aspect='auto', extent=[0, len(signal)/fs, freqs[-1], freqs[0]])
#     plt.xlabel('Time (s)')
#     plt.ylabel('Frequency (Hz)')
#     plt.yscale('log')
#     plt.colorbar(label='Magnitude')
#     plt.title(f'CWT Scalogram - {axis_name} - Bin {bin_num}')
    
#     plot_path = os.path.join(output_dir, f"{filename.replace('.csv', '')}_{axis_name}_bin{bin_num}_cwt.png")
#     plt.tight_layout()
#     plt.savefig(plot_path)
#     plt.close()

# -------- BATCH LOOP --------
for filename in os.listdir(input_dir):
    if filename.endswith("_features.csv"):
        print(f"📂 Processing: {filename}")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, low_memory=False)
        
        # Store original shape for comparison
        original_rows = df.shape[0]
        print(f"Original dataframe shape: {df.shape}")

        # Calculate CWT features without changing the original dataframe
        # Create a new dataframe to store our bin-level CWT features
        all_features = []
        
        # Group by bin but don't modify the original dataframe
        for bin_num, group in df.groupby('bin'):
            bin_features = {'bin': bin_num}
            
            # Calculate CWT features for each axis
            for axis in columns:
                if axis in group.columns:
                    # Get signal values without dropping NAs in the original data
                    signal = group[axis].values
                    
                    # Skip the signal if it's all NaN
                    if np.all(np.isnan(signal)):
                        bin_features[f'{axis}_Peak_Frequency'] = np.nan
                        bin_features[f'{axis}_Peak_Amplitude'] = np.nan
                        continue
                    
                    # For the CWT calculation only, drop NaNs
                    signal_clean = signal[~np.isnan(signal)]
                    
                    # Generate diagnostic plot for the first bin (optional)
                    #if bin_num == df['bin'].min() and axis == columns[0]:
                    #    plot_cwt_spectrum(signal_clean, filename, axis, bin_num)
                        
                    freq, amp = calculate_peak_amplitude_frequency(signal_clean)
                    bin_features[f'{axis}_Peak_Frequency'] = freq
                    bin_features[f'{axis}_Peak_Amplitude'] = amp
                else:
                    print(f"⚠️ Column {axis} not found in dataframe")
                    bin_features[f'{axis}_Peak_Frequency'] = np.nan
                    bin_features[f'{axis}_Peak_Amplitude'] = np.nan
            
            all_features.append(bin_features)
        
        # Convert bin features to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Now we merge back to the original dataframe
        # The merge will keep all rows from the original df
        result = df.merge(features_df, on='bin', how='left')
        
        # Print shape information
        print(f"Features dataframe shape: {features_df.shape}")
        print(f"Final result shape: {result.shape}")
        
        if result.shape[0] != original_rows:
            print(f"⚠️ WARNING: Row count changed from {original_rows} to {result.shape[0]}!")
        
        # Save the enhanced dataframe
        out_name = filename.replace("_features.csv", "_features_cwt.csv")
        out_path = os.path.join(output_dir, out_name)
        result.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")