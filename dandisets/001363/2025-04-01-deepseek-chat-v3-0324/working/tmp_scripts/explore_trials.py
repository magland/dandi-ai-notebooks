# Script to explore trial durations and visualize example waveforms from NWB file

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001363/assets/5fe5283e-d987-4fad-bf65-ca1045b5bb51/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trial information
trials = nwb.intervals["trials"]
start_times = trials["start_time"][:]
stop_times = trials["stop_time"][:]
durations = stop_times - start_times

print(f"Number of trials: {len(trials)}")
print(f"Average trial duration: {np.mean(durations):.3f}s")
print(f"First trial start time: {start_times[0]:.3f}s")
print(f"Last trial stop time: {stop_times[-1]:.3f}s")

# Plot trial durations
plt.figure(figsize=(10, 4))
plt.plot(trials["id"][:], durations, 'o')
plt.xlabel('Trial ID')
plt.ylabel('Duration (s)')
plt.title('Trial Durations')
plt.savefig('trial_durations.png')

# Get electrical series data
electrical_series = nwb.acquisition["ElectricalSeries"]
sample_rate = electrical_series.rate
electrode_ids = electrical_series.electrodes["id"][:]

# Plot waveform examples from first trial for first 3 electrodes
start_sample = int(start_times[0] * sample_rate)
end_sample = int(start_sample + 0.1 * sample_rate)  # 100ms window

plt.figure(figsize=(12, 6))
for i in range(3):
    data = electrical_series.data[start_sample:end_sample, i]
    time = np.arange(len(data)) / sample_rate
    plt.plot(time, data + i*500, label=f'Electrode {electrode_ids[i]}')

plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.title('Example Waveforms (First Trial)')
plt.legend()
plt.savefig('example_waveforms.png')