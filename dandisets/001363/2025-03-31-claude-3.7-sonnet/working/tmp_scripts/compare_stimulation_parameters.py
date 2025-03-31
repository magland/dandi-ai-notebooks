"""
This script compares neural responses across different tFUS stimulation parameters.
It loads and analyzes data from two different NWB files to compare responses
under different experimental conditions.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Function to load an NWB file and get basic info
def load_nwb_file(asset_id):
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001363/assets/{asset_id}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    return nwb, f

# Function to calculate trial-averaged response for a specific channel
def calculate_averaged_response(nwb, channel_idx=0, num_trials=10, window_before=0.5, window_after=1.5):
    trials = nwb.intervals["trials"]
    electrical_series = nwb.acquisition["ElectricalSeries"]
    sampling_rate = electrical_series.rate
    
    # Calculate buffer sizes in samples
    buffer_before = int(window_before * sampling_rate)
    buffer_after = int(window_after * sampling_rate)
    total_samples = buffer_before + buffer_after
    
    # Initialize array for averaged data
    averaged_data = np.zeros(total_samples)
    trials_used = 0
    
    # Get trial start times
    trial_start_times = trials['start_time'][:]
    
    # Use a subset of trials (skip the first few)
    start_trial = 10
    for trial_idx in range(start_trial, start_trial + num_trials):
        if trial_idx >= len(trial_start_times):
            break
            
        trial_start = trial_start_times[trial_idx]
        start_sample = int(trial_start * sampling_rate) - buffer_before
        stop_sample = int(trial_start * sampling_rate) + buffer_after
        
        # Skip trials that would go out of bounds
        if start_sample < 0 or stop_sample >= electrical_series.data.shape[0]:
            continue
        
        # Load and add the data for the specific channel
        trial_data = electrical_series.data[start_sample:stop_sample, channel_idx]
        averaged_data += trial_data
        trials_used += 1
    
    # Calculate the average
    if trials_used > 0:
        averaged_data /= trials_used
    
    # Create the time axis relative to trial onset
    time_axis = np.linspace(-window_before, window_after, total_samples)
    
    return time_axis, averaged_data, trials_used

# Load two different NWB files to compare
# First one is from first subject (BH589), session 1
file1_id = "59d1acbb-5ad5-45f1-b211-c2e311801824"  # BH589 first session
nwb1, f1 = load_nwb_file(file1_id)

# Second one is from first subject (BH589), session 5
file2_id = "6b9aa3e6-2389-4f84-a2d0-a3201894ad3c"  # BH589 fifth session
nwb2, f2 = load_nwb_file(file2_id)

# Print basic info about the two files
print("File 1 info:")
print(f"Session description: {nwb1.session_description}")
print(f"Identifier: {nwb1.identifier}")
print(f"Subject ID: {nwb1.subject.subject_id}")
print()

print("File 2 info:")
print(f"Session description: {nwb2.session_description}")
print(f"Identifier: {nwb2.identifier}")
print(f"Subject ID: {nwb2.subject.subject_id}")
print()

# Compare trial structure
trials1 = nwb1.intervals["trials"]
trials2 = nwb2.intervals["trials"]

print(f"File 1 number of trials: {len(trials1['start_time'][:])}")
print(f"File 2 number of trials: {len(trials2['start_time'][:])}")

# Calculate average trial duration
trial_durations1 = trials1['stop_time'][:] - trials1['start_time'][:]
trial_durations2 = trials2['stop_time'][:] - trials2['start_time'][:]

print(f"File 1 average trial duration: {np.mean(trial_durations1):.4f} seconds")
print(f"File 2 average trial duration: {np.mean(trial_durations2):.4f} seconds")

# Calculate and plot average responses for both files
# For each file, select a few channels to analyze
channels_to_analyze = [0, 10, 20, 30]  # Sample channels from different parts of the array
num_trials_to_average = 20

plt.figure(figsize=(15, 10))

# For each channel, plot the averaged responses from both files
for i, channel_idx in enumerate(channels_to_analyze):
    plt.subplot(len(channels_to_analyze), 1, i+1)
    
    # Calculate averaged response for file 1
    time1, avg_resp1, trials_used1 = calculate_averaged_response(
        nwb1, channel_idx=channel_idx, num_trials=num_trials_to_average)
    
    # Calculate averaged response for file 2
    time2, avg_resp2, trials_used2 = calculate_averaged_response(
        nwb2, channel_idx=channel_idx, num_trials=num_trials_to_average)
    
    # Plot both responses
    plt.plot(time1, avg_resp1, label=f'File 1: {nwb1.identifier}', color='blue')
    plt.plot(time2, avg_resp2, label=f'File 2: {nwb2.identifier}', color='red')
    
    # Add vertical line for stimulus onset
    plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
    
    plt.title(f'Channel {channel_idx} Response')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_between_sessions.png')

# Calculate the power spectrum for both files
plt.figure(figsize=(12, 10))

for i, channel_idx in enumerate(channels_to_analyze):
    plt.subplot(len(channels_to_analyze), 1, i+1)
    
    # Time windows
    pre_stim_window = (-0.5, -0.05)  # Before stimulus
    stim_window = (0.05, 1.0)       # During stimulus
    
    # File 1 - Get data for pre and during stimulation
    time1, avg_resp1, _ = calculate_averaged_response(
        nwb1, channel_idx=channel_idx, num_trials=num_trials_to_average)
    
    # Calculate indices for time windows
    pre_idx1 = np.logical_and(time1 >= pre_stim_window[0], time1 <= pre_stim_window[1])
    stim_idx1 = np.logical_and(time1 >= stim_window[0], time1 <= stim_window[1])
    
    # Calculate power spectrum
    fs = nwb1.acquisition["ElectricalSeries"].rate
    
    # Pre-stimulus power (File 1)
    f1_pre, psd1_pre = signal.welch(avg_resp1[pre_idx1], fs, nperseg=int(0.2*fs))
    
    # During-stimulus power (File 1)
    f1_stim, psd1_stim = signal.welch(avg_resp1[stim_idx1], fs, nperseg=int(0.2*fs))
    
    # File 2 - Get data for pre and during stimulation
    time2, avg_resp2, _ = calculate_averaged_response(
        nwb2, channel_idx=channel_idx, num_trials=num_trials_to_average)
    
    # Calculate indices for time windows
    pre_idx2 = np.logical_and(time2 >= pre_stim_window[0], time2 <= pre_stim_window[1])
    stim_idx2 = np.logical_and(time2 >= stim_window[0], time2 <= stim_window[1])
    
    # Calculate power spectrum
    fs = nwb2.acquisition["ElectricalSeries"].rate
    
    # Pre-stimulus power (File 2)
    f2_pre, psd2_pre = signal.welch(avg_resp2[pre_idx2], fs, nperseg=int(0.2*fs))
    
    # During-stimulus power (File 2)
    f2_stim, psd2_stim = signal.welch(avg_resp2[stim_idx2], fs, nperseg=int(0.2*fs))
    
    # Plot power spectra
    plt.plot(f1_pre, 10*np.log10(psd1_pre), color='blue', linestyle='--', label=f'File 1: Pre-stim')
    plt.plot(f1_stim, 10*np.log10(psd1_stim), color='blue', label=f'File 1: During-stim')
    plt.plot(f2_pre, 10*np.log10(psd2_pre), color='red', linestyle='--', label=f'File 2: Pre-stim')
    plt.plot(f2_stim, 10*np.log10(psd2_stim), color='red', label=f'File 2: During-stim')
    
    plt.title(f'Channel {channel_idx} Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.xlim(0, 500)  # Limit to 0-500 Hz
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_spectra_comparison.png')