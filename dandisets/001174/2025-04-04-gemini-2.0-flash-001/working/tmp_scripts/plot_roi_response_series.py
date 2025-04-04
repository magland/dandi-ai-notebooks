# Plots the RoiResponseSeries data from the NWB file.

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the RoiResponseSeries object
RoiResponseSeries = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]

# Get the data
data = RoiResponseSeries.data

# Get the number of neurons
num_neurons = data.shape[1]

# Select a subset of neurons to plot (e.g., the first 5)
neurons_to_plot = min(5, num_neurons)

# Select a subset of time points to plot (e.g., the first 1000)
time_points_to_plot = min(1000, data.shape[0])

# Plot the data for the selected neurons
plt.figure(figsize=(10, 6))
for i in range(neurons_to_plot):
    plt.plot(data[:time_points_to_plot, i], label=f"Neuron {i}")

plt.title("RoiResponseSeries Data")
plt.xlabel("Time (frames)")
plt.ylabel("Fluorescence")
plt.legend()
plt.savefig("tmp_scripts/roi_response_series.png")
plt.close()