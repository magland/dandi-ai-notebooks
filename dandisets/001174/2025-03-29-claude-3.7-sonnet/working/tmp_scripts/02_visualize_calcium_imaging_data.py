"""
Script to visualize calcium imaging data from the NWB file.
We'll visualize:
1. A sample frame from the raw imaging data
2. The spatial masks of the ROIs
3. Fluorescence traces for a few ROIs
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
print(f"Loading NWB file from {url}")
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the one photon series data
one_photon_series = nwb.acquisition["OnePhotonSeries"]

# Get the ROI masks
plane_segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
roi_masks = plane_segmentation["image_mask"]

# Get the fluorescence data
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]

# Plot 1: Display a sample frame (middle frame)
print("Getting a sample frame from the middle of the recording...")
middle_frame_idx = one_photon_series.data.shape[0] // 2
sample_frame = one_photon_series.data[middle_frame_idx, :, :]

plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.title(f"Sample Frame (Frame #{middle_frame_idx})")
plt.colorbar(label='Intensity')
plt.savefig('sample_frame.png', dpi=150)
plt.close()

# Plot 2: Display ROI masks
print("Plotting ROI masks...")
# Create a combined image of all ROI masks
roi_masks_combined = np.zeros((roi_masks.data.shape[1], roi_masks.data.shape[2]))
for i in range(roi_masks.data.shape[0]):
    roi_masks_combined = np.maximum(roi_masks_combined, roi_masks.data[i])

plt.figure(figsize=(10, 8))
plt.imshow(roi_masks_combined, cmap='viridis')
plt.title(f"Combined ROI Masks (n={roi_masks.data.shape[0]})")
plt.colorbar(label='Mask Value')
plt.savefig('combined_roi_masks.png', dpi=150)
plt.close()

# Plot 3: Display individual ROI masks for first 9 ROIs
print("Plotting individual ROI masks...")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(min(9, roi_masks.data.shape[0])):
    ax = axes[i // 3, i % 3]
    ax.imshow(roi_masks.data[i], cmap='hot')
    ax.set_title(f"ROI #{i}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('individual_roi_masks.png', dpi=150)
plt.close()

# Plot 4: Plot fluorescence traces for 5 ROIs
print("Plotting fluorescence traces...")
plt.figure(figsize=(12, 8))
time_vector = np.arange(fluorescence.data.shape[0]) / fluorescence.rate
for i in range(5):  # Plot first 5 ROIs
    plt.plot(time_vector, fluorescence.data[:, i], label=f'ROI #{i}')

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Fluorescence Traces for 5 ROIs')
plt.legend()
plt.savefig('fluorescence_traces.png', dpi=150)
plt.close()

# Plot 5: Overlay ROI masks on sample frame
print("Creating overlay of ROIs on sample frame...")
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')

# Plot contours of each ROI on top of the image
colors = plt.cm.rainbow(np.linspace(0, 1, roi_masks.data.shape[0]))
for i in range(roi_masks.data.shape[0]):
    mask = roi_masks.data[i]
    # Find contours of the mask
    plt.contour(mask, levels=[0.5], colors=[colors[i]], linewidths=1)

plt.title('ROIs Overlaid on Sample Frame')
plt.savefig('roi_overlay.png', dpi=150)
plt.close()

print("All visualizations complete and saved.")