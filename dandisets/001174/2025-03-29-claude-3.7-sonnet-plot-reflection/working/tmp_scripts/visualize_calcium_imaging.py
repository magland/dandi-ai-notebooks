"""
This script visualizes calcium imaging data from the NWB file, including:
1. A sample frame from the raw imaging data
2. The spatial footprints (masks) of detected neurons
3. Fluorescence traces for selected neurons
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the output directory for plots
output_dir = "tmp_scripts"

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the OnePhotonSeries data
one_photon_series = nwb.acquisition["OnePhotonSeries"]

# Get the segmentation data
plane_seg = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
image_masks = plane_seg['image_mask'].data

# Get fluorescence data
fluorescence = nwb.processing['ophys']['Fluorescence']['RoiResponseSeries']

# Define a subset of frames to visualize
frame_index = 1000  # Use frame 1000 as an example
num_frames_to_plot = 500  # For time series plots

# 1. Plot a sample frame from the calcium imaging data
plt.figure(figsize=(10, 8))
sample_frame = one_photon_series.data[frame_index]
plt.imshow(sample_frame, cmap='gray')
plt.title(f'Raw Calcium Imaging Frame (Frame #{frame_index})')
plt.colorbar(label='Fluorescence Intensity')
plt.savefig(os.path.join(output_dir, 'sample_frame.png'), dpi=150)
plt.close()

# 2. Plot the spatial footprints (masks) of all neurons overlaid on the sample frame
plt.figure(figsize=(10, 8))

# First plot the sample frame in grayscale
plt.imshow(sample_frame, cmap='gray')

# Create a combined mask to overlay
combined_mask = np.zeros(image_masks.shape[1:])  # shape should be (298, 160)
for i in range(image_masks.shape[0]):  # Loop through all neurons
    combined_mask = np.maximum(combined_mask, image_masks[i])

# Overlay the combined mask with some transparency and a different colormap
plt.imshow(combined_mask, alpha=0.7, cmap='hot')

plt.title(f'Neuron Masks Overlaid on Frame #{frame_index}')
plt.colorbar(label='Mask Value')
plt.savefig(os.path.join(output_dir, 'neuron_masks_overlay.png'), dpi=150)
plt.close()

# 3. Plot individual spatial footprints for the first 6 neurons
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(min(6, image_masks.shape[0])):
    ax = axes[i]
    mask = image_masks[i]
    ax.imshow(mask, cmap='hot')
    ax.set_title(f'Neuron {i} Spatial Footprint')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'individual_neuron_masks.png'), dpi=150)
plt.close()

# 4. Plot fluorescence traces for a few neurons over time
plt.figure(figsize=(14, 8))

# Get a subset of the fluorescence data
start_frame = 1000
neurons_to_plot = min(5, fluorescence.data.shape[1])  # Plot up to 5 neurons

for i in range(neurons_to_plot):
    # Extract the fluorescence time series for this neuron
    trace = fluorescence.data[start_frame:start_frame+num_frames_to_plot, i]
    
    # Create time points (in seconds)
    time_points = np.arange(num_frames_to_plot) / fluorescence.rate
    
    # Plot the trace
    plt.plot(time_points, trace, label=f'Neuron {i}')

plt.xlabel('Time (seconds)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Fluorescence Traces for Selected Neurons')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fluorescence_traces.png'), dpi=150)
plt.close()

# 5. Plot event amplitude traces for the same neurons
event_amplitude = nwb.processing['ophys']['EventAmplitude']

plt.figure(figsize=(14, 8))

# Get a subset of the event amplitude data
for i in range(neurons_to_plot):
    # Extract the event amplitude time series for this neuron
    trace = event_amplitude.data[start_frame:start_frame+num_frames_to_plot, i]
    
    # Create time points (in seconds)
    time_points = np.arange(num_frames_to_plot) / event_amplitude.rate
    
    # Plot the trace
    plt.plot(time_points, trace, label=f'Neuron {i}')

plt.xlabel('Time (seconds)')
plt.ylabel('Event Amplitude')
plt.title('Event Amplitude Traces for Selected Neurons')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'event_amplitude_traces.png'), dpi=150)
plt.close()

# 6. Plot a heatmap of all neuron fluorescence activity
plt.figure(figsize=(14, 8))

# Get data for all neurons for a subset of time
all_traces = fluorescence.data[start_frame:start_frame+num_frames_to_plot, :]

# Create a heatmap
plt.imshow(all_traces.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Fluorescence (a.u.)')
plt.xlabel('Frame Number')
plt.ylabel('Neuron ID')
plt.title('Heatmap of All Neuron Fluorescence Activity')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fluorescence_heatmap.png'), dpi=150)
plt.close()

print("Visualization complete! All plots saved to the tmp_scripts directory.")