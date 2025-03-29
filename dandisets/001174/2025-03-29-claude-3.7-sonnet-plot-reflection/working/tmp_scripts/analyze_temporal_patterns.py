"""
This script analyzes temporal patterns in the calcium imaging data, including:
1. Activity patterns over the full recording session
2. Power spectral density to identify frequency components
3. Active neuron count over time
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

# Set the output directory for plots
output_dir = "tmp_scripts"

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get fluorescence data
fluorescence = nwb.processing['ophys']['Fluorescence']['RoiResponseSeries']
fluor_data = fluorescence.data[:, :]  # Get all data
sampling_rate = fluorescence.rate  # Hz

# Get basic information about the recording
num_frames = fluor_data.shape[0]
num_neurons = fluor_data.shape[1]
recording_duration = num_frames / sampling_rate  # seconds

print(f"Recording contains {num_frames} frames from {num_neurons} neurons")
print(f"Sampling rate: {sampling_rate:.2f} Hz")
print(f"Recording duration: {recording_duration:.2f} seconds ({recording_duration/60:.2f} minutes)")

# Calculate z-scored data for better comparison between neurons
z_scored_data = np.zeros_like(fluor_data, dtype=np.float32)
for i in range(num_neurons):
    neuron_data = fluor_data[:, i]
    z_scored_data[:, i] = (neuron_data - np.mean(neuron_data)) / np.std(neuron_data)

# Get a subset of the data to avoid memory issues
# Use every 20th frame to reduce data size (effectively downsampling)
subset_inds = np.arange(0, num_frames, 20)
z_subset = z_scored_data[subset_inds, :]
time_subset = subset_inds / sampling_rate  # in seconds

# 1. Plot average activity across all neurons over time
plt.figure(figsize=(14, 6))
mean_activity = np.mean(z_subset, axis=1)
plt.plot(time_subset, mean_activity)
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Z-scored Fluorescence')
plt.title('Average Neural Activity Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'average_activity_time.png'), dpi=150)
plt.close()

# 2. Compute and plot power spectral density to identify oscillatory patterns
# Choose a few active neurons
most_active_neurons = np.argsort(np.std(z_scored_data, axis=0))[-3:]  # Top 3 most variable neurons

plt.figure(figsize=(12, 8))
for i, neuron_id in enumerate(most_active_neurons):
    # Compute Power Spectral Density
    freqs, psd = signal.welch(z_scored_data[:, neuron_id], fs=sampling_rate, 
                             nperseg=min(1024, len(z_scored_data)), 
                             scaling='spectrum')
    
    # Plot only frequencies up to 1 Hz (typical for calcium imaging)
    mask = freqs <= 1.0
    plt.semilogy(freqs[mask], psd[mask], label=f'Neuron {neuron_id}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectrum of Neural Activity')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'power_spectrum.png'), dpi=150)
plt.close()

# 3. Count active neurons over time
# Define active as z-score > 2 (2 std above mean)
active_threshold = 2.0
is_active = z_scored_data > active_threshold
active_count = np.sum(is_active, axis=1)

# Plot number of active neurons over time
plt.figure(figsize=(14, 6))
# Again, use downsampled time points to make plotting faster
plt.plot(time_subset, active_count[subset_inds])
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Active Neurons (Z > 2)')
plt.title('Number of Simultaneously Active Neurons Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'active_neuron_count.png'), dpi=150)
plt.close()

# 4. Create a raster plot showing binary activity patterns
plt.figure(figsize=(14, 8))
plt.imshow(is_active[subset_inds, :].T, aspect='auto', cmap='binary', 
           extent=[0, time_subset[-1], 0, num_neurons])
plt.xlabel('Time (seconds)')
plt.ylabel('Neuron ID')
plt.title('Neural Activity Raster Plot (Z-score > 2)')
plt.colorbar(label='Active (1) / Inactive (0)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'raster_plot.png'), dpi=150)
plt.close()

print("Temporal analysis completed successfully.")