"""
Script to visualize ROI masks from the NWB file.
This focuses only on the ROI masks to avoid timeout issues.
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

# Get the ROI masks
print("Accessing ROI masks...")
plane_segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
print(f"Number of ROIs: {plane_segmentation['image_mask'].data.shape[0]}")

# Plot 1: Display ROI masks combined
print("Creating combined ROI masks visualization...")
# Create a combined image of all ROI masks
roi_masks_combined = np.zeros((plane_segmentation['image_mask'].data.shape[1], 
                               plane_segmentation['image_mask'].data.shape[2]))

for i in range(plane_segmentation['image_mask'].data.shape[0]):
    # Load one mask at a time
    mask = plane_segmentation['image_mask'].data[i]
    roi_masks_combined = np.maximum(roi_masks_combined, mask)
    print(f"Processed mask {i+1}/{plane_segmentation['image_mask'].data.shape[0]}")

plt.figure(figsize=(10, 8))
plt.imshow(roi_masks_combined, cmap='viridis')
plt.title(f"Combined ROI Masks (n={plane_segmentation['image_mask'].data.shape[0]})")
plt.colorbar(label='Mask Value')
plt.savefig('combined_roi_masks.png', dpi=150)
print("Saved combined ROI masks visualization.")
plt.close()

# Plot 2: Display individual ROI masks for first 9 ROIs
print("Creating individual ROI masks visualization...")
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(min(9, plane_segmentation['image_mask'].data.shape[0])):
    ax = axes[i // 3, i % 3]
    ax.imshow(plane_segmentation['image_mask'].data[i], cmap='hot')
    ax.set_title(f"ROI #{i}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('individual_roi_masks.png', dpi=150)
print("Saved individual ROI masks visualization.")
plt.close()

print("ROI mask visualizations complete.")