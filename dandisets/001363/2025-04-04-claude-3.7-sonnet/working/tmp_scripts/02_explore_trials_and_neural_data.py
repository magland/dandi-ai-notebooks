# This script explores both trial timing and a sample of the neural data
# to better understand the dataset characteristics

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# PART 1: Explore trial timing
print("==== TRIAL TIMING ANALYSIS ====")
trials = nwb.intervals["trials"]
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
durations = trial_stops - trial_starts

# Calculate inter-trial intervals
inter_trial_intervals = np.zeros(len(trial_starts) - 1)
for i in range(len(trial_starts) - 1):
    inter_trial_intervals[i] = trial_starts[i+1] - trial_stops[i]

print(f"Number of trials: {len(trials.id[:])}")
print(f"Mean trial duration: {np.mean(durations)*1000:.2f} ms (std: {np.std(durations)*1000:.2f} ms)")
print(f"Mean inter-trial interval: {np.mean(inter_trial_intervals)*1000:.2f} ms (std: {np.std(inter_trial_intervals)*1000:.2f} ms)")

# Plot trial durations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(durations * 1000, bins=20)  # Convert to milliseconds
plt.title('Trial Durations')
plt.xlabel('Duration (ms)')
plt.ylabel('Count')

# Plot inter-trial intervals
plt.subplot(1, 2, 2)
plt.hist(inter_trial_intervals * 1000, bins=20)  # Convert to milliseconds
plt.title('Inter-Trial Intervals')
plt.xlabel('Interval (ms)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('tmp_scripts/trial_timing_analysis.png')
plt.close()

# PART 2: Explore neural data from a subset of channels and time
print("\n==== NEURAL DATA EXPLORATION ====")
electrical_series = nwb.acquisition["ElectricalSeries"]
print(f"Neural data shape: {electrical_series.data.shape}, sampling rate: {electrical_series.rate} Hz")

# Calculate data duration
data_duration_seconds = electrical_series.data.shape[0] / electrical_series.rate
print(f"Total recording duration: {data_duration_seconds:.2f} seconds ({data_duration_seconds/60:.2f} minutes)")

# Get electrode information
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes.id[:])}")

# Plot data from first few trials, subsampling for visualization
first_trial_start_idx = int(trial_starts[0] * electrical_series.rate)
# Let's look at the first two trials
second_trial_end_idx = int(trial_stops[1] * electrical_series.rate)
# Select first 4 channels for visualization
channels_to_plot = [0, 1, 2, 3]

# Create time axis in seconds
time_axis = np.arange(first_trial_start_idx, second_trial_end_idx) / electrical_series.rate
# Sample every 100th point to reduce data size for plotting
sample_rate = 100
sampled_time = time_axis[::sample_rate]

# Sample data for plotting
print("Sampling neural data for plotting...")
sampled_data = electrical_series.data[first_trial_start_idx:second_trial_end_idx:sample_rate, channels_to_plot]

# Create a plot of the first two trials for the selected channels
plt.figure(figsize=(15, 10))
for i, channel in enumerate(channels_to_plot):
    plt.subplot(len(channels_to_plot), 1, i+1)
    plt.plot(sampled_time, sampled_data[:, i])
    plt.title(f'Channel {channel}', fontsize=10)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (µV)')
    
    # Add vertical lines to mark trial boundaries
    for t_start in trial_starts[:2]:
        plt.axvline(x=t_start, color='r', linestyle='--', alpha=0.5)
    for t_stop in trial_stops[:2]:
        plt.axvline(x=t_stop, color='g', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('tmp_scripts/neural_data_first_two_trials.png')
plt.close()

# PART 3: Power spectrum analysis of a sample of neural data
# For spectral analysis, let's use a 10-second continuous segment
print("\n==== SPECTRAL ANALYSIS ====")
segment_duration = 10  # seconds
segment_start_idx = first_trial_start_idx
segment_end_idx = segment_start_idx + int(segment_duration * electrical_series.rate)

try:
    # Sample segment data for spectral analysis
    print("Extracting data segment for spectral analysis...")
    segment_data = electrical_series.data[segment_start_idx:segment_end_idx, channels_to_plot]
    
    # Compute power spectra using Welch's method
    print("Computing power spectra...")
    from scipy import signal
    
    plt.figure(figsize=(12, 8))
    for i, channel in enumerate(channels_to_plot):
        # Calculate power spectrum
        f, Pxx = signal.welch(segment_data[:, i], fs=electrical_series.rate, nperseg=int(electrical_series.rate))
        
        # Plot power spectrum (limit to 0-200 Hz for better visualization)
        max_freq = 200
        freq_mask = f <= max_freq
        plt.semilogy(f[freq_mask], Pxx[freq_mask])
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (µV²/Hz)')
    plt.title('Power Spectra of Selected Channels (0-200 Hz)')
    plt.legend([f'Channel {ch}' for ch in channels_to_plot])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('tmp_scripts/power_spectra.png')
    plt.close()
    
except Exception as e:
    print(f"Error in spectral analysis: {e}")