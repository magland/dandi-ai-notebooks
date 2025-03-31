"""
This script analyzes a single trial from the NWB file to avoid timeout issues.
It loads minimal data and creates simple visualizations.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

print("Starting single trial analysis...")
start_time = time.time()

# Load the NWB file
asset_id = "59d1acbb-5ad5-45f1-b211-c2e311801824"
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001363/assets/{asset_id}/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Subject ID: {nwb.subject.subject_id}")

# Get trial information
trials = nwb.intervals["trials"]
trial_start_times = trials['start_time'][:]
trial_stop_times = trials['stop_time'][:]

# Get electrical series information
electrical_series = nwb.acquisition["ElectricalSeries"]
sampling_rate = electrical_series.rate

print(f"Number of trials: {len(trial_start_times)}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Data shape: {electrical_series.data.shape}")

# Select a single trial to analyze
trial_index = 2
channel_index = 0

trial_start = trial_start_times[trial_index]
trial_stop = trial_stop_times[trial_index]
trial_duration = trial_stop - trial_start

print(f"Analyzing trial {trial_index+1}, duration: {trial_duration:.4f} seconds")

# Add a window before and after the trial
window_before = 0.2  # seconds before trial
window_after = 0.2   # seconds after trial

# Calculate indices
start_idx = max(0, int((trial_start - window_before) * sampling_rate))
stop_idx = min(electrical_series.data.shape[0], int((trial_stop + window_after) * sampling_rate))

# Load data for a single channel
print(f"Loading data for channel {channel_index}...")
data = electrical_series.data[start_idx:stop_idx, channel_index]

# Create time axis
time_axis = np.arange(len(data)) / sampling_rate
time_axis = time_axis - (trial_start - start_idx / sampling_rate)

# Plot the raw data
plt.figure(figsize=(10, 6))
plt.plot(time_axis, data, label=f'Channel {channel_index}')

# Add vertical lines for trial boundaries
plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
plt.axvline(x=trial_duration, color='g', linestyle='--', label='Trial Stop')

# Add formatting
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Amplitude (μV)')
plt.title(f'Raw Data for Trial {trial_index+1}, Channel {channel_index}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('single_trial_raw.png')
print("Saved raw data plot")

# Calculate a simple frequency analysis
# Divide the data into three segments: before, during, and after stimulation
before_stim_idx = time_axis < 0
during_stim_idx = (time_axis >= 0) & (time_axis <= trial_duration)
after_stim_idx = time_axis > trial_duration

# Only proceed if we have enough data in each segment
if (sum(before_stim_idx) > 0 and sum(during_stim_idx) > 0 and sum(after_stim_idx) > 0):
    # Calculate power spectra
    print("Calculating frequency content...")
    
    # Apply a notch filter to remove line noise (assuming 60 Hz)
    b, a = signal.iirnotch(60, 30, sampling_rate)
    filtered_data = signal.filtfilt(b, a, data)
    
    # Get segments
    before_data = filtered_data[before_stim_idx]
    during_data = filtered_data[during_stim_idx]
    after_data = filtered_data[after_stim_idx]
    
    # Calculate power spectra using Welch's method
    f_before, psd_before = signal.welch(before_data, sampling_rate, nperseg=min(1024, len(before_data)))
    f_during, psd_during = signal.welch(during_data, sampling_rate, nperseg=min(1024, len(during_data)))
    f_after, psd_after = signal.welch(after_data, sampling_rate, nperseg=min(1024, len(after_data)))
    
    # Plot frequency content
    plt.figure(figsize=(10, 6))
    
    # Convert to dB scale
    plt.plot(f_before, 10 * np.log10(psd_before), label='Before stimulus', color='blue')
    plt.plot(f_during, 10 * np.log10(psd_during), label='During stimulus', color='red')
    plt.plot(f_after, 10 * np.log10(psd_after), label='After stimulus', color='green')
    
    # Add formatting
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title(f'Frequency Content During Trial {trial_index+1}, Channel {channel_index}')
    plt.xlim(0, 500)  # Limit to 0-500 Hz
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('single_trial_frequency.png')
    print("Saved frequency content plot")

# Simple filtering to see if we can detect any events
print("Performing basic filtering...")

# 1. Apply bandpass filter to isolate potential LFPs (1-30 Hz)
b_lfp, a_lfp = signal.butter(4, [1, 30], 'bandpass', fs=sampling_rate)
lfp_filtered = signal.filtfilt(b_lfp, a_lfp, data)

# 2. Apply highpass filter to isolate potential spikes (>300 Hz)
b_spikes, a_spikes = signal.butter(4, 300, 'highpass', fs=sampling_rate)
spike_filtered = signal.filtfilt(b_spikes, a_spikes, data)

# Plot filtered data
plt.figure(figsize=(12, 8))

# Raw data
plt.subplot(3, 1, 1)
plt.plot(time_axis, data, label='Raw data')
plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=trial_duration, color='g', linestyle='--')
plt.title('Raw signal')
plt.legend()
plt.grid(True, alpha=0.3)

# LFP band
plt.subplot(3, 1, 2)
plt.plot(time_axis, lfp_filtered, label='LFP band (1-30 Hz)')
plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=trial_duration, color='g', linestyle='--')
plt.title('LFP band (1-30 Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

# Spike band
plt.subplot(3, 1, 3)
plt.plot(time_axis, spike_filtered, label='Spike band (>300 Hz)')
plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=trial_duration, color='g', linestyle='--')
plt.title('Spike band (>300 Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('single_trial_filtered.png')
print("Saved filtered data plot")

end_time = time.time()
print(f"Script completed in {end_time - start_time:.2f} seconds")