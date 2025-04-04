# This script explores a different NWB file in the dataset to compare neural responses
# across different ultrasound stimulation parameters

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load a different NWB file to compare - using one from a different recording session
# Let's try the file from subject BH589 from a different session with potentially different parameters
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/096d8401-3c74-4c72-b403-64bc56c66656/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("==== BASIC METADATA ====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")

# Get electrical series and trial information
electrical_series = nwb.acquisition["ElectricalSeries"]
trials = nwb.intervals["trials"]
sample_rate = electrical_series.rate

print(f"\nNeural data shape: {electrical_series.data.shape}, sampling rate: {sample_rate} Hz")
print(f"Number of trials: {len(trials.id[:])}")

# Get information about electrodes
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes.id[:])}")

# Check trial timing
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
durations = trial_stops - trial_starts

print(f"Mean trial duration: {np.mean(durations)*1000:.2f} ms (std: {np.std(durations)*1000:.2f} ms)")

# Calculate inter-trial intervals
inter_trial_intervals = np.zeros(len(trial_starts) - 1)
for i in range(len(trial_starts) - 1):
    inter_trial_intervals[i] = trial_starts[i+1] - trial_stops[i]

print(f"Mean inter-trial interval: {np.mean(inter_trial_intervals)*1000:.2f} ms (std: {np.std(inter_trial_intervals)*1000:.2f} ms)")

# Select subset of channels to analyze (including a few additional channels)
channels_to_analyze = [0, 1, 2, 3, 8, 16]
print(f"Analyzing channels: {channels_to_analyze}")

# Define parameters for trial-aligned analysis
pre_stim_time = 0.2  # seconds before trial onset
post_stim_time = 0.5  # seconds after trial onset
pre_samples = int(pre_stim_time * sample_rate)
post_samples = int(post_stim_time * sample_rate)
total_samples = pre_samples + post_samples

# Select subset of trials to analyze (first 50 to keep processing manageable)
max_trials = 50
trial_indices = range(min(max_trials, len(trials.id[:])))
print(f"Analyzing first {len(trial_indices)} trials")

# Function to extract trial-aligned data
def extract_trial_data(trial_idx, channel_idx):
    trial_start_time = trials['start_time'][trial_idx]
    trial_start_sample = int(trial_start_time * sample_rate)
    start_idx = max(0, trial_start_sample - pre_samples)
    end_idx = min(electrical_series.data.shape[0], trial_start_sample + post_samples)
    
    if end_idx - start_idx < total_samples:
        # Handle edge cases where we can't get the full window
        return None
    
    # Extract data
    data = electrical_series.data[start_idx:end_idx, channel_idx]
    
    # Ensure we have the right number of samples
    if len(data) != total_samples:
        if len(data) < total_samples:
            pad_length = total_samples - len(data)
            data = np.pad(data, (0, pad_length), 'constant', constant_values=0)
        else:
            data = data[:total_samples]
    
    return data

# Create time axis (in seconds) relative to stimulus onset
time_axis = np.linspace(-pre_stim_time, post_stim_time, total_samples)

# Initialize arrays to store trial-aligned data
trial_data = {ch: [] for ch in channels_to_analyze}

# Extract trial-aligned data for each channel
for channel in channels_to_analyze:
    print(f"Processing channel {channel}...")
    for trial_idx in trial_indices:
        data = extract_trial_data(trial_idx, channel)
        if data is not None:
            trial_data[channel].append(data)

# Convert lists to numpy arrays
for channel in channels_to_analyze:
    trial_data[channel] = np.array(trial_data[channel])
    print(f"Channel {channel}: collected data for {trial_data[channel].shape[0]} trials")

# Plot trial-averaged responses for each channel
plt.figure(figsize=(15, 12))
for i, channel in enumerate(channels_to_analyze):
    plt.subplot(len(channels_to_analyze), 1, i+1)
    
    # Calculate mean and standard error across trials
    channel_mean = np.mean(trial_data[channel], axis=0)
    channel_sem = np.std(trial_data[channel], axis=0) / np.sqrt(trial_data[channel].shape[0])
    
    # Plot mean and confidence interval
    plt.plot(time_axis, channel_mean)
    plt.fill_between(time_axis, 
                     channel_mean - channel_sem,
                     channel_mean + channel_sem,
                     alpha=0.3)
    
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)  # Mark stimulus onset
    plt.axvline(x=0.067, color='g', linestyle='--', alpha=0.7)  # Mark stimulus offset (67ms)
    
    plt.title(f'Channel {channel} - Trial-Averaged Response (File 2)')
    plt.xlabel('Time from Stimulus Onset (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tmp_scripts/trial_averaged_responses_file2.png')
plt.close()

# Select one channel for detailed time-frequency analysis
channel_for_analysis = 0

# Compute and plot time-frequency representations
print("\nComputing time-frequency analysis for Channel 0...")

# Select a subset of trials for time-frequency analysis
tf_trials = np.arange(0, min(20, len(trial_data[channel_for_analysis])))
    
# Parameters for wavelet transform
freqs = np.linspace(5, 150, 30)  # frequencies from 5-150 Hz
n_cycles = freqs / 2.  # Use fewer cycles for lower frequencies

# Time-frequency analysis
trial_avg_tf = np.zeros((len(freqs), len(time_axis)))

for trial_idx in tf_trials:
    # Get trial data
    trial_signal = trial_data[channel_for_analysis][trial_idx, :]
    
    # Compute wavelet transform
    for i, freq in enumerate(freqs):
        # Compute wavelet width
        width = n_cycles[i] / (2 * np.pi * freq)
        # Compute wavelet transform
        try:
            # Use pywt instead of deprecated morlet2
            import pywt
            scales = width * sample_rate
            wavelet = signal.morlet2(len(time_axis), width, freq / sample_rate * len(time_axis))
            # Convolve signal with wavelet
            output = np.abs(signal.fftconvolve(trial_signal, wavelet, mode='same'))
            # Add to average
            trial_avg_tf[i, :] += output
        except:
            # Fallback if there's an error
            pass

# Divide by number of trials to get average
trial_avg_tf /= len(tf_trials)

# Plot time-frequency representation
plt.figure(figsize=(12, 8))
plt.imshow(trial_avg_tf, aspect='auto', origin='lower', 
           extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]], 
           cmap='viridis')
plt.colorbar(label='Power')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)  # Mark stimulus onset
plt.axvline(x=0.067, color='g', linestyle='--', alpha=0.7)  # Mark stimulus offset (67ms)
plt.title(f'Time-Frequency Representation - Channel {channel_for_analysis} (File 2)')
plt.xlabel('Time from Stimulus Onset (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('tmp_scripts/time_frequency_analysis_file2.png')
plt.close()

# Compare signal amplitudes across channels
plt.figure(figsize=(12, 6))
channel_means = []
channel_stds = []

for channel in channels_to_analyze:
    # Calculate the mean and std of the trial-averaged signal
    channel_mean = np.mean(trial_data[channel], axis=0)
    # Calculate RMS amplitude
    rms_amp = np.sqrt(np.mean(channel_mean**2))
    # Calculate standard deviation
    std_amp = np.std(channel_mean)
    
    channel_means.append(rms_amp)
    channel_stds.append(std_amp)

# Plot as bar chart with error bars
plt.bar(range(len(channels_to_analyze)), channel_means, yerr=channel_stds, alpha=0.7)
plt.xticks(range(len(channels_to_analyze)), [f"Ch {ch}" for ch in channels_to_analyze])
plt.title('RMS Amplitude by Channel (File 2)')
plt.xlabel('Channel')
plt.ylabel('RMS Amplitude (µV)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tmp_scripts/channel_amplitude_comparison_file2.png')
plt.close()