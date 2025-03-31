"""
This script performs spike detection analysis on a single NWB file.
It extracts short segments of data around trials and detects spikes.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# Start timing the script
start_time = time.time()

# Load the NWB file
print("Loading NWB file...")
asset_id = "59d1acbb-5ad5-45f1-b211-c2e311801824"  # First file
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001363/assets/{asset_id}/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic info
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Subject ID: {nwb.subject.subject_id}")

# Get trials and electrical series info
electrical_series = nwb.acquisition["ElectricalSeries"]
sampling_rate = electrical_series.rate
trials = nwb.intervals["trials"]

print(f"Data shape: {electrical_series.data.shape}")
print(f"Number of trials: {len(trials['start_time'][:])}")
print(f"Sampling rate: {sampling_rate} Hz")

# Function to detect spikes in a segment of data
def detect_spikes(data, threshold_factor=5.0):
    """
    Detect spikes in data using threshold crossing.
    
    Args:
        data: 1D array of neural data
        threshold_factor: Factor to multiply std for threshold
    
    Returns:
        Arrays of spike times in samples
    """
    # Filter data (300 Hz highpass) to isolate spikes
    b, a = signal.butter(3, 300/(sampling_rate/2), 'highpass')
    filtered = signal.filtfilt(b, a, data)
    
    # Compute threshold based on filtered data
    threshold = threshold_factor * np.std(filtered)
    
    # Detect threshold crossings
    spike_indices = np.where(filtered < -threshold)[0]
    
    # Ensure spikes are separated
    if len(spike_indices) > 1:
        spike_times = [spike_indices[0]]
        for i in range(1, len(spike_indices)):
            if spike_indices[i] > spike_indices[i-1] + int(0.002 * sampling_rate):  # 2ms refractory period
                spike_times.append(spike_indices[i])
        return np.array(spike_times)
    return spike_indices

# Analyze a subset of channels and trials
channels_to_analyze = [0, 5, 10, 15, 20, 25, 30]
trials_to_analyze = 10
spike_counts_pre = np.zeros((len(channels_to_analyze), trials_to_analyze))
spike_counts_post = np.zeros((len(channels_to_analyze), trials_to_analyze))

# Window sizes (in seconds)
pre_window = 0.5
post_window = 1.0

print(f"Analyzing {len(channels_to_analyze)} channels across {trials_to_analyze} trials...")

# Extract data and detect spikes for each channel and trial
for ch_idx, channel in enumerate(channels_to_analyze):
    for tr_idx in range(trials_to_analyze):
        trial_start = trials['start_time'][tr_idx]
        
        # Calculate sample indices
        pre_start_idx = int((trial_start - pre_window) * sampling_rate)
        post_start_idx = int(trial_start * sampling_rate)
        post_end_idx = int((trial_start + post_window) * sampling_rate)
        
        # Skip if out of bounds
        if pre_start_idx < 0 or post_end_idx >= electrical_series.data.shape[0]:
            continue
        
        # Extract data segments
        pre_data = electrical_series.data[pre_start_idx:post_start_idx, channel]
        post_data = electrical_series.data[post_start_idx:post_end_idx, channel]
        
        # Detect spikes
        pre_spikes = detect_spikes(pre_data)
        post_spikes = detect_spikes(post_data)
        
        # Store spike counts
        spike_counts_pre[ch_idx, tr_idx] = len(pre_spikes)
        spike_counts_post[ch_idx, tr_idx] = len(post_spikes)

# Calculate average spike rates (spikes per second)
avg_rate_pre = np.mean(spike_counts_pre, axis=1) / pre_window
avg_rate_post = np.mean(spike_counts_post, axis=1) / post_window

# Plot average spike rates before and after stimulus
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(channels_to_analyze))

plt.bar(index, avg_rate_pre, bar_width, label='Pre-stimulus', color='blue', alpha=0.7)
plt.bar(index + bar_width, avg_rate_post, bar_width, label='Post-stimulus', color='red', alpha=0.7)

plt.xlabel('Channel')
plt.ylabel('Average Spike Rate (Hz)')
plt.title('Pre vs Post-Stimulus Spike Rates')
plt.xticks(index + bar_width/2, [f'Ch {ch}' for ch in channels_to_analyze])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spike_rates_comparison.png')

# Examine raw traces and detected spikes for one example
example_channel = channels_to_analyze[0]
example_trial = 5

trial_start = trials['start_time'][example_trial]
window_before = 0.1  # Use a smaller window for clarity
window_after = 0.3

start_idx = int((trial_start - window_before) * sampling_rate)
end_idx = int((trial_start + window_after) * sampling_rate)

if start_idx >= 0 and end_idx < electrical_series.data.shape[0]:
    # Extract data
    example_data = electrical_series.data[start_idx:end_idx, example_channel]
    time_axis = np.arange(len(example_data)) / sampling_rate - window_before
    
    # Filter data for spike detection
    b, a = signal.butter(3, 300/(sampling_rate/2), 'highpass')
    filtered_data = signal.filtfilt(b, a, example_data)
    
    # Detect spikes
    threshold = 5.0 * np.std(filtered_data)
    spikes = detect_spikes(example_data)
    spike_times = (spikes / sampling_rate) - window_before
    
    # Plot raw and filtered data with detected spikes
    plt.figure(figsize=(12, 8))
    
    # Raw data
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, example_data, label='Raw Data')
    if len(spikes) > 0:
        plt.scatter(spike_times, [example_data[s] for s in spikes], 
                    color='red', s=50, label='Detected Spikes')
    plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
    plt.title(f'Channel {example_channel}, Trial {example_trial+1}')
    plt.xlabel('Time relative to stimulus (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Filtered data
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, filtered_data, label='Filtered Data (>300 Hz)')
    plt.axhline(y=-threshold, color='r', linestyle='--', label='Spike Threshold')
    plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
    plt.xlabel('Time relative to stimulus (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spike_detection_example.png')

# Calculate spike timing relative to stimulus
combined_pre_spikes = []
combined_post_spikes = []

# Analyze more trials for PSTH
num_trials_psth = 50
pre_window_psth = 0.5
post_window_psth = 1.0
bin_size = 0.01  # 10 ms bins

# Use a single channel for PSTH
psth_channel = channels_to_analyze[0]

for tr_idx in range(min(num_trials_psth, len(trials['start_time'][:]))):
    trial_start = trials['start_time'][tr_idx]
    
    # Calculate sample indices
    pre_start_idx = int((trial_start - pre_window_psth) * sampling_rate)
    post_end_idx = int((trial_start + post_window_psth) * sampling_rate)
    
    # Skip if out of bounds
    if pre_start_idx < 0 or post_end_idx >= electrical_series.data.shape[0]:
        continue
    
    # Extract full segment
    data_segment = electrical_series.data[pre_start_idx:post_end_idx, psth_channel]
    
    # Detect spikes
    spike_indices = detect_spikes(data_segment)
    
    # Convert to seconds relative to stimulus onset
    if len(spike_indices) > 0:
        spike_times = spike_indices / sampling_rate - pre_window_psth
        
        # Add to appropriate list
        pre_mask = spike_times < 0
        post_mask = spike_times >= 0
        
        combined_pre_spikes.extend(spike_times[pre_mask])
        combined_post_spikes.extend(spike_times[post_mask])

# Create PSTH
bins = np.arange(-pre_window_psth, post_window_psth + bin_size, bin_size)
plt.figure(figsize=(12, 5))
plt.hist(combined_pre_spikes + combined_post_spikes, bins=bins, color='blue', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
plt.xlabel('Time relative to stimulus onset (s)')
plt.ylabel('Spike Count')
plt.title(f'Peri-Stimulus Time Histogram (Channel {psth_channel})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spike_psth.png')

# Print execution time
end_time = time.time()
print(f"Script completed in {end_time - start_time:.2f} seconds")