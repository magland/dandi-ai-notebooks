"""
This script visualizes the electrical recordings from the NWB file.
It plots electrical activity for a subset of channels during a sample trial
to examine neural responses to transcranial focused ultrasound stimulation.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get electrical series data
electrical_series = nwb.acquisition["ElectricalSeries"]
sampling_rate = electrical_series.rate

# Get trial information
trials = nwb.intervals["trials"]
trial_start_times = trials['start_time'][:]
trial_stop_times = trials['stop_time'][:]

# Select a trial to examine (trial index 10)
trial_index = 10
trial_start = trial_start_times[trial_index]
trial_stop = trial_stop_times[trial_index]

# Convert trial times to data indices
start_idx = int(trial_start * sampling_rate)
stop_idx = int(trial_stop * sampling_rate)

# Add a buffer before and after the trial (1 second before, 1 second after)
buffer_samples = int(1 * sampling_rate)
plot_start_idx = max(0, start_idx - buffer_samples)
plot_stop_idx = min(electrical_series.data.shape[0], stop_idx + buffer_samples)

# Load subset of data (only a few channels to save memory)
channels_to_plot = [0, 1, 2, 3, 4]  # First 5 channels
data_subset = electrical_series.data[plot_start_idx:plot_stop_idx, channels_to_plot]

# Convert indices to time (in seconds)
time_axis = np.arange(plot_start_idx, plot_stop_idx) / sampling_rate
relative_time = time_axis - trial_start

# Create the figure
plt.figure(figsize=(12, 8))

# Plot the data for each channel with offset
channel_offsets = np.arange(len(channels_to_plot)) * 200  # Offset each channel for visibility
for i, channel_idx in enumerate(channels_to_plot):
    plt.plot(relative_time, data_subset[:, i] + channel_offsets[i], 
             label=f'Channel {channel_idx}')

# Add vertical lines for trial start and stop
plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
plt.axvline(x=trial_stop - trial_start, color='g', linestyle='--', label='Trial Stop')

# Formatting
plt.xlabel('Time relative to trial onset (s)')
plt.ylabel('Amplitude (μV + offset)')
plt.title(f'Electrical Activity During Trial {trial_index+1}')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Add text to indicate stimulus parameters
plt.text(0.02, 0.02, 'tFUS stimulation window', 
         transform=plt.gca().transAxes, color='red', 
         bbox=dict(facecolor='white', alpha=0.7))

# Save the figure
plt.tight_layout()
plt.savefig('trial_electrical_activity.png')

# Create a spectrogram for the first channel
plt.figure(figsize=(10, 6))
channel_to_analyze = 0  # First channel

# Compute spectrogram using scipy
from scipy import signal
f, t, Sxx = signal.spectrogram(data_subset[:, channel_to_analyze], 
                               fs=sampling_rate, 
                               nperseg=int(0.1 * sampling_rate),  # 100ms window
                               noverlap=int(0.09 * sampling_rate))  # 90% overlap

# Plot spectrogram
plt.pcolormesh(t + relative_time[0], f, 10 * np.log10(Sxx), shading='gouraud')
plt.colorbar(label='Power/Frequency (dB/Hz)')

# Add vertical lines for trial start and stop
plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
plt.axvline(x=trial_stop - trial_start, color='g', linestyle='--', label='Trial Stop')

plt.ylabel('Frequency (Hz)')
plt.xlabel('Time relative to trial onset (s)')
plt.title(f'Spectrogram of Channel {channel_to_analyze} During Trial {trial_index+1}')
plt.ylim(0, 500)  # Limit to 0-500 Hz
plt.tight_layout()
plt.savefig('trial_spectrogram.png')

# Plot average activity across multiple trials
# Select 10 trials to average
trial_indices = range(10, 20)
trial_window_before = 0.5  # seconds before trial start
trial_window_after = 1.5   # seconds after trial start

# Calculate average across trials
buffer_before = int(trial_window_before * sampling_rate)
buffer_after = int(trial_window_after * sampling_rate)
averaged_data = np.zeros((buffer_before + buffer_after, len(channels_to_plot)))
num_trials_used = 0

for trial_idx in trial_indices:
    trial_start = trial_start_times[trial_idx]
    start_sample = int(trial_start * sampling_rate) - buffer_before
    stop_sample = int(trial_start * sampling_rate) + buffer_after
    
    # Skip trials that would go out of bounds
    if start_sample < 0 or stop_sample >= electrical_series.data.shape[0]:
        continue
    
    # Load and add the data
    trial_data = electrical_series.data[start_sample:stop_sample, channels_to_plot]
    averaged_data += trial_data
    num_trials_used += 1

# Calculate the average
if num_trials_used > 0:
    averaged_data /= num_trials_used

# Create the time axis relative to trial onset
relative_time = np.linspace(-trial_window_before, trial_window_after, 
                           buffer_before + buffer_after)

# Plot the averaged data
plt.figure(figsize=(10, 6))
for i, channel_idx in enumerate(channels_to_plot):
    plt.plot(relative_time, averaged_data[:, i], 
             label=f'Channel {channel_idx}')

# Add vertical line for trial start
plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')

# Formatting
plt.xlabel('Time relative to stimulus onset (s)')
plt.ylabel('Amplitude (μV)')
plt.title(f'Average Neural Response Across {num_trials_used} Trials')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('average_response.png')