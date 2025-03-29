"""
This script explores the calcium imaging fluorescence data to:
1. Plot fluorescence traces for selected cells
2. Create a heatmap of all cell activities
3. Calculate and plot basic statistics of neural activity
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the fluorescence data
fluorescence = nwb.processing['ophys']['Fluorescence']['RoiResponseSeries']
data = fluorescence.data[:]  # shape [time_points, num_cells]
sampling_rate = fluorescence.rate  # Hz

# Create time vector (in seconds)
time = np.arange(data.shape[0]) / sampling_rate

# Plot 1: Fluorescence traces for first 5 cells
plt.figure(figsize=(15, 10))
for i in range(5):
    # Normalize the trace for better visualization
    trace = data[:, i]
    normalized_trace = (trace - np.mean(trace)) / np.std(trace)
    plt.plot(time, normalized_trace + i*4, label=f'Cell {i+1}')

plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Fluorescence (offset for clarity)')
plt.title('Fluorescence Traces of Individual Cells')
plt.legend()
plt.grid(True)
plt.xlim([0, 300])  # Show first 300 seconds
plt.savefig('tmp_scripts/fluorescence_traces.png')
plt.close()

# Plot 2: Activity heatmap for first 300 seconds
plt.figure(figsize=(15, 8))
time_slice = int(300 * sampling_rate)  # 300 seconds worth of data points
# Normalize each cell's activity
normalized_data = np.zeros_like(data[:time_slice, :])
for i in range(data.shape[1]):
    normalized_data[:, i] = (data[:time_slice, i] - np.mean(data[:time_slice, i])) / np.std(data[:time_slice, i])

plt.imshow(normalized_data.T, aspect='auto', cmap='viridis',
           extent=[0, 300, 0, data.shape[1]])
plt.colorbar(label='Z-scored Fluorescence')
plt.xlabel('Time (seconds)')
plt.ylabel('Cell Number')
plt.title('Neural Activity Heatmap (First 300 seconds)')
plt.savefig('tmp_scripts/activity_heatmap.png')
plt.close()

# Plot 3: Basic statistics
mean_activity = np.mean(data, axis=0)
std_activity = np.std(data, axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(mean_activity, std_activity, alpha=0.6)
plt.xlabel('Mean Fluorescence')
plt.ylabel('Standard Deviation of Fluorescence')
plt.title('Cell Activity Statistics')
plt.grid(True)
plt.savefig('tmp_scripts/activity_statistics.png')
plt.close()

# Print some basic information
print(f"Number of cells: {data.shape[1]}")
print(f"Recording duration: {data.shape[0]/sampling_rate:.2f} seconds")
print(f"Sampling rate: {sampling_rate:.2f} Hz")