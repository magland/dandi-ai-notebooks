"""
This script explores the calcium imaging event data to:
1. Plot event amplitude traces for selected cells
2. Create a heatmap of neural events
3. Calculate and plot basic event statistics
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the event amplitude data
events = nwb.processing['ophys']['EventAmplitude']
data = events.data[:]  # shape [time_points, num_cells]
sampling_rate = events.rate  # Hz

# Create time vector (in seconds)
time = np.arange(data.shape[0]) / sampling_rate

# Plot 1: Event amplitude traces for first 5 cells
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.plot(time[:3000], data[:3000, i] + i*2, label=f'Cell {i+1}')  # First 300 seconds

plt.xlabel('Time (seconds)')
plt.ylabel('Event Amplitude (offset for clarity)')
plt.title('Neural Event Amplitudes of Individual Cells')
plt.legend()
plt.grid(True)
plt.savefig('tmp_scripts/event_traces.png')
plt.close()

# Plot 2: Event amplitude heatmap
plt.figure(figsize=(15, 8))
# Create heatmap for first 300 seconds
time_slice = 3000  # 300 seconds * 10 Hz
plt.imshow(data[:time_slice, :].T, aspect='auto', cmap='viridis',
           extent=[0, 300, 0, data.shape[1]], vmin=0, vmax=np.percentile(data, 99))
plt.colorbar(label='Event Amplitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Cell Number')
plt.title('Neural Event Amplitude Heatmap (First 300 seconds)')
plt.savefig('tmp_scripts/event_heatmap.png')
plt.close()

# Plot 3: Basic event statistics
mean_events = np.mean(data, axis=0)
max_events = np.max(data, axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(mean_events, max_events, alpha=0.6)
plt.xlabel('Mean Event Amplitude')
plt.ylabel('Max Event Amplitude')
plt.title('Cell Event Statistics')
plt.grid(True)
plt.savefig('tmp_scripts/event_statistics.png')
plt.close()

# Print some basic information
print(f"Number of cells: {data.shape[1]}")
print(f"Recording duration: {data.shape[0]/sampling_rate:.2f} seconds")
print(f"Sampling rate: {sampling_rate:.2f} Hz")
print(f"Mean event amplitude: {np.mean(mean_events):.4f}")
print(f"Max event amplitude: {np.max(max_events):.4f}")