import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the LFP data
LFPs = nwb.acquisition["LFPs"]
lfp_data = LFPs.data
electrodes = LFPs.electrodes

# Select a few channels to plot
channel_ids = [0, 1, 2]

# Select a time window to plot (first 10 seconds)
start_time = 0
end_time = 10
start_index = int(start_time * LFPs.rate)
end_index = int(end_time * LFPs.rate)

# Generate timestamps if they don't exist
if LFPs.timestamps is None:
    timestamps = np.arange(LFPs.starting_time, LFPs.starting_time + lfp_data.shape[0] / LFPs.rate, 1 / LFPs.rate)
else:
    timestamps = LFPs.timestamps[:]

# Plot the LFP data for each channel
plt.figure(figsize=(10, 6))
for channel_id in channel_ids:
    channel_index = np.where(electrodes["id"].data[:] == channel_id)[0][0]
    plt.plot(timestamps[start_index:end_index], lfp_data[start_index:end_index, channel_index], label=f"Channel {channel_id}")

plt.xlabel("Time (s)")
plt.ylabel("LFP (mV)")
plt.title("LFP Data for Selected Channels")
plt.legend()
plt.savefig("tmp_scripts/lfp_plot.png")
print("LFP plot saved to tmp_scripts/lfp_plot.png")

# Display channel locations
print("Channel Locations:")
for channel_id in channel_ids:
    channel_index = np.where(electrodes["id"].data[:] == channel_id)[0][0]
    x = electrodes["x"].data[channel_index]
    y = electrodes["y"].data[channel_index]
    z = electrodes["z"].data[channel_index]
    location = electrodes["location"].data[channel_index]
    print(f"Channel {channel_id}: x={x}, y={y}, z={z}, location={location}")

print("Done!")