"""
Initial exploration of the NWB file to understand:
1. Basic dataset information
2. Electrode locations and configuration
3. Sample of electrical recordings around a trial
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information
print("\nBasic Dataset Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Protocol: {nwb.protocol}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Subject Age: {nwb.subject.age}")
print(f"Subject Sex: {nwb.subject.sex}")

# Get electrode information
electrodes = nwb.electrodes
x_pos = electrodes['x'].data[:]
y_pos = electrodes['y'].data[:]
z_pos = electrodes['z'].data[:]

# Plot electrode positions
plt.figure(figsize=(10, 10))
plt.scatter(x_pos, y_pos, c=z_pos, cmap='viridis', s=100)
plt.colorbar(label='Z Position')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Electrode Array Configuration')
plt.savefig('tmp_scripts/electrode_positions.png')
plt.close()

# Get trial information
trials = nwb.intervals['trials']
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]

print("\nTrial Information:")
print(f"Number of trials: {len(trial_starts)}")
print(f"Average trial duration: {np.mean(trial_stops - trial_starts):.3f} seconds")

# Get information about the electrical series
electrical_series = nwb.acquisition['ElectricalSeries']
sampling_rate = electrical_series.rate
print(f"\nElectrical Series Information:")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Data shape: {electrical_series.data.shape}")
print(f"First trial start time: {trial_starts[0]} seconds")

# Get a smaller sample of data (100ms before and 200ms after trial start)
time_before = 0.1  # seconds before trial
time_after = 0.2   # seconds after trial

# Calculate sample indices
start_idx = max(0, int((trial_starts[0] - time_before) * sampling_rate))
end_idx = min(electrical_series.data.shape[0], 
              int((trial_starts[0] + time_after) * sampling_rate))

print(f"\nData Selection:")
print(f"Start index: {start_idx}")
print(f"End index: {end_idx}")

# Load the data slice
data_sample = electrical_series.data[start_idx:end_idx, :]
print(f"Sample data shape: {data_sample.shape}")

# Create time array relative to trial start
time = np.arange(data_sample.shape[0]) / sampling_rate - time_before

# Plot sample data for first 4 channels
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.plot(time, data_sample[:, i] + i*200, label=f'Channel {i+1}')
plt.axvline(x=0, color='r', linestyle='--', label='Trial Start')
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Voltage (μV)')
plt.title('Sample Neural Recordings Around First Trial')
plt.legend()
plt.grid(True)
plt.savefig('tmp_scripts/sample_data.png')
plt.close()