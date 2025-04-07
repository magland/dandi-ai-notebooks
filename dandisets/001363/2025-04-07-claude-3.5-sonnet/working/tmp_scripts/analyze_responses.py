"""
Analysis of neural responses to ultrasound stimulation:
1. Visualize raw signals around stimulation
2. Look at trial-averaged responses
3. Examine response variability across trials
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

# Get trial information
trials = nwb.intervals['trials']
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]

# Get electrical series info
electrical_series = nwb.acquisition['ElectricalSeries']
sampling_rate = electrical_series.rate

# Parameters for analysis
time_before = 0.2  # seconds before trial
time_after = 0.3   # seconds after trial
num_trials_to_plot = 3  # Number of example trials to plot
num_channels = 4  # Number of channels to analyze

# Function to get data around a trial
def get_trial_data(trial_start):
    start_idx = max(0, int((trial_start - time_before) * sampling_rate))
    end_idx = min(electrical_series.data.shape[0], 
                  int((trial_start + time_after) * sampling_rate))
    data = electrical_series.data[start_idx:end_idx, :num_channels]
    time = np.arange(data.shape[0]) / sampling_rate - time_before
    return time, data

# Plot individual trial examples
plt.figure(figsize=(15, 12))
for trial_idx in range(num_trials_to_plot):
    trial_start = trial_starts[trial_idx]
    time, data = get_trial_data(trial_start)
    
    plt.subplot(num_trials_to_plot, 1, trial_idx + 1)
    for ch in range(num_channels):
        plt.plot(time, data[:, ch], label=f'Channel {ch+1}')
    plt.axvline(x=0, color='r', linestyle='--', label='Stimulation')
    plt.axvline(x=0.067, color='r', linestyle=':', label='Stim End')  # 67ms stim duration
    plt.grid(True)
    if trial_idx == 0:
        plt.title('Example Single Trial Responses')
    plt.ylabel('Voltage (μV)')
    if trial_idx == num_trials_to_plot - 1:
        plt.xlabel('Time relative to stimulation (s)')
    plt.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/trial_examples.png')
plt.close()

# Calculate trial-averaged response
print("Calculating trial-averaged response...")
num_trials_avg = 50  # Number of trials to average
samples_per_trial = int((time_before + time_after) * sampling_rate)
trial_avg = np.zeros((samples_per_trial, num_channels))
count = 0

for trial_idx in range(min(num_trials_avg, len(trial_starts))):
    time, data = get_trial_data(trial_starts[trial_idx])
    if data.shape[0] == samples_per_trial:  # Only use complete trials
        trial_avg += data
        count += 1

if count > 0:
    trial_avg /= count
    
    # Plot trial-averaged response
    plt.figure(figsize=(15, 8))
    time = np.linspace(-time_before, time_after, samples_per_trial)
    for ch in range(num_channels):
        plt.plot(time, trial_avg[:, ch], label=f'Channel {ch+1}')
    plt.axvline(x=0, color='r', linestyle='--', label='Stimulation')
    plt.axvline(x=0.067, color='r', linestyle=':', label='Stim End')
    plt.title(f'Trial-Averaged Response (n={count} trials)')
    plt.xlabel('Time relative to stimulation (s)')
    plt.ylabel('Voltage (μV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tmp_scripts/trial_average.png')
    plt.close()