"""
This script explores the processed ROI data from the NWB file, including
PlaneSegmentation, Fluorescence, and EventAmplitude.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information about the ophys module
print("\nOphys Module Information:")
ophys = nwb.processing["ophys"]
print(f"Available data: {list(ophys.data_interfaces.keys())}")

# Get PlaneSegmentation information
plane_seg = ophys["ImageSegmentation"]["PlaneSegmentation"]
print("\nPlaneSegmentation Information:")
print(f"Number of ROIs: {plane_seg['image_mask'].data.shape[0]}")
print(f"Image mask shape: {plane_seg['image_mask'].data.shape}")

# Get Fluorescence information
fluor = ophys["Fluorescence"]["RoiResponseSeries"]
print("\nFluorescence Information:")
print(f"Data shape: {fluor.data.shape}")
print(f"Rate: {fluor.rate} Hz")
print(f"Duration: {fluor.data.shape[0] / fluor.rate:.2f} seconds")

# Get EventAmplitude information
event_amp = ophys["EventAmplitude"]
print("\nEventAmplitude Information:")
print(f"Data shape: {event_amp.data.shape}")

# Plot a few example ROI masks
print("\nPlotting example ROI masks...")
num_rois_to_plot = min(6, plane_seg['image_mask'].data.shape[0])
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(num_rois_to_plot):
    axes[i].imshow(plane_seg['image_mask'].data[i], cmap='viridis')
    axes[i].set_title(f'ROI {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/example_roi_masks.png', dpi=300)
plt.close()

# Plot the combined ROI mask
print("Creating combined ROI mask...")
combined_mask = np.zeros_like(plane_seg['image_mask'].data[0])
for i in range(plane_seg['image_mask'].data.shape[0]):
    combined_mask = np.maximum(combined_mask, plane_seg['image_mask'].data[i])

plt.figure(figsize=(10, 10))
plt.imshow(combined_mask, cmap='hot')
plt.colorbar(label='ROI Presence')
plt.title('Combined ROI Masks')
plt.axis('off')
plt.savefig('tmp_scripts/combined_roi_masks.png', dpi=300)
plt.close()

# Plot fluorescence traces for a few ROIs
print("Plotting fluorescence traces...")
# Sample every 5th frame to keep data size reasonable
sample_rate = 5
num_rois_to_plot = min(5, fluor.data.shape[1])
sample_indices = np.arange(0, min(1000, fluor.data.shape[0]), sample_rate)
times = sample_indices / fluor.rate

plt.figure(figsize=(12, 8))
for i in range(num_rois_to_plot):
    plt.plot(times, fluor.data[sample_indices, i], label=f'ROI {i}')

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence')
plt.title('Fluorescence Traces for Selected ROIs')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/fluorescence_traces.png', dpi=300)
plt.close()

# Plot event amplitudes for a few ROIs
print("Plotting event amplitudes...")
# Sample every 5th frame to keep data size reasonable
num_rois_to_plot = min(5, event_amp.data.shape[1])
sample_indices = np.arange(0, min(1000, event_amp.data.shape[0]), sample_rate)
times = sample_indices / event_amp.rate

plt.figure(figsize=(12, 8))
for i in range(num_rois_to_plot):
    plt.plot(times, event_amp.data[sample_indices, i], label=f'ROI {i}')

plt.xlabel('Time (s)')
plt.ylabel('Event Amplitude')
plt.title('Event Amplitudes for Selected ROIs')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/event_amplitudes.png', dpi=300)
plt.close()

# Create a heatmap of fluorescence activity across ROIs
print("Creating fluorescence activity heatmap...")
# Sample a subset of time points
time_subset = fluor.data[sample_indices, :num_rois_to_plot].T

plt.figure(figsize=(12, 6))
im = plt.imshow(time_subset, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar(im, label='Fluorescence')
plt.xlabel('Time Point')
plt.ylabel('ROI')
plt.title('Fluorescence Activity Across ROIs')
plt.tight_layout()
plt.savefig('tmp_scripts/fluorescence_heatmap.png', dpi=300)
plt.close()

print("Script completed successfully!")