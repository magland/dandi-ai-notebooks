# tmp_scripts/explore_ecephys.py
# This script explores the extracellular electrophysiology data in the NWB file,
# loads a subset of the data, and plots it.

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/0bbd0039-0f40-4eb2-b67a-93802fb3b28d/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
except Exception as e:
    print(f"Error loading NWB file: {e}")
    exit()

# Get the ElectricalSeries object
try:
    electrical_series = nwb.acquisition["ElectricalSeriesVP0"]
    data = electrical_series.data
    rate = electrical_series.rate
    electrodes = electrical_series.electrodes
    electrode_ids = electrodes["id"].data[:]

except Exception as e:
    print(f"Error accessing ElectricalSeries data: {e}")
    exit()

# Select a subset of channels and time points
num_channels = 4
num_timepoints = 1000
start_channel = 0
start_time = 0

# Extract the data subset
try:
    subset_channels = electrode_ids[start_channel:start_channel + num_channels]
    channel_indices = np.where(np.isin(electrodes["id"].data[:], subset_channels))[0]
    subset_data = data[start_time:start_time + num_timepoints, channel_indices]
    time = np.arange(start_time, start_time + num_timepoints) / rate
except Exception as e:
    print(f"Error extracting data subset: {e}")
    exit()

# Plot the data
try:
    plt.figure(figsize=(10, 6))
    for i in range(num_channels):
        plt.plot(time, subset_data[:, i], label=f"Channel {subset_channels[i]}")

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title("Extracellular Electrophysiology Data")
    plt.legend()
    plt.savefig("tmp_scripts/ecephys_plot.png")
    plt.close()

except Exception as e:
    print(f"Error plotting data: {e}")
    exit()

print("Plot saved to tmp_scripts/ecephys_plot.png")