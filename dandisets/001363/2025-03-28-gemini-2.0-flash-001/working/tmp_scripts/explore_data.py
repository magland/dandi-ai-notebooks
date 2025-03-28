# Explore ElectricalSeries data and electrode locations

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001363/assets/b8de194c-d6ad-42e4-9e8f-bddffd2dc86b/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the ElectricalSeries data
electrical_series = nwb.acquisition["ElectricalSeries"]
data = electrical_series.data
rate = electrical_series.rate
electrodes = nwb.electrodes

# Plot the first few seconds of data from a few channels
num_channels = 4
duration = 5  # seconds
num_samples = int(duration * rate)
channels = range(num_channels)

plt.figure(figsize=(10, 6))
for i, channel in enumerate(channels):
    plt.plot(np.linspace(0, duration, num_samples), data[:num_samples, channel] + i * 100, label=f"Channel {channel}")  # Add offset for clarity
plt.xlabel("Time (s)")
plt.ylabel("Voltage (uV) + Offset")
plt.title("Sample Data from ElectricalSeries")
plt.legend()
plt.savefig("tmp_scripts/electrical_series.png")
plt.close()

# Plot electrode locations
electrode_x = electrodes["x"].data[:]
electrode_y = electrodes["y"].data[:]

plt.figure(figsize=(6, 6))
plt.scatter(electrode_x, electrode_y)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Electrode Locations")
plt.savefig("tmp_scripts/electrode_locations.png")
plt.close()

# Access trials data
trials = nwb.intervals["trials"]
start_time = trials["start_time"].data[:]
stop_time = trials["stop_time"].data[:]

# Plot trials start and stop times
plt.figure(figsize=(10, 6))
plt.plot(start_time, label="Start Time")
plt.plot(stop_time, label="Stop Time")
plt.xlabel("Trial Index")
plt.ylabel("Time (s)")
plt.title("Trials Start and Stop Times")
plt.legend()
plt.savefig("tmp_scripts/trials_start_stop.png")
plt.close()