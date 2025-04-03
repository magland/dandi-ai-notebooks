"""
Visualize the ROIs (Regions of Interest) from the PlaneSegmentation data.
This script will load the ROI masks and create a visualization showing where
the individual cells are located in the field of view.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Load the NWB file using lindi
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the PlaneSegmentation object that contains ROI masks
plane_segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
roi_masks = plane_segmentation["image_mask"].data[:]  # Get all ROI masks
num_rois = roi_masks.shape[0]
print(f"Number of ROIs (cells): {num_rois}")
print(f"Mask dimensions: {roi_masks.shape[1]} x {roi_masks.shape[2]} pixels")

# Create a combined mask that shows all ROIs with different colors
# By taking the maximum across ROIs, we ensure overlapping regions show the most recent ROI
combined_mask = np.zeros((roi_masks.shape[1], roi_masks.shape[2], 3))
all_rois_mask = np.max(roi_masks, axis=0)  # For visualization of all ROIs in one image

# Create a colormap for the ROIs
colors = plt.cm.jet(np.linspace(0, 1, num_rois))

# Create a separate visualization with all ROIs color-coded
fig, ax = plt.subplots(figsize=(10, 8))

# For the color-coded visualization
colored_mask = np.zeros((roi_masks.shape[1], roi_masks.shape[2], 4))
for i in range(num_rois):
    # Create an RGBA array for this ROI
    roi_rgba = np.zeros((roi_masks.shape[1], roi_masks.shape[2], 4))
    # Set the RGB values based on the color from the colormap
    roi_rgba[..., 0] = colors[i, 0] * roi_masks[i]
    roi_rgba[..., 1] = colors[i, 1] * roi_masks[i]
    roi_rgba[..., 2] = colors[i, 2] * roi_masks[i]
    roi_rgba[..., 3] = roi_masks[i] * 0.7  # Alpha (transparency)
    
    # Add this ROI to the combined mask
    # Only update pixels where this ROI has data
    mask = roi_masks[i] > 0
    colored_mask[mask] = roi_rgba[mask]

# Display the combined color-coded mask
ax.imshow(colored_mask)
ax.set_title('ROIs (Color-coded by ROI ID)')
ax.axis('off')

# Create a legend with a subset of ROIs to avoid overcrowding
legend_elements = []
step = max(1, num_rois // 10)  # Show about 10 ROIs in the legend
for i in range(0, num_rois, step):
    legend_elements.append(Patch(facecolor=colors[i], label=f'ROI {i}'))
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('tmp_scripts/rois_colored.png', dpi=300, bbox_inches='tight')

# Create a heatmap showing all ROIs combined (for density visualization)
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = np.sum(roi_masks, axis=0)  # Sum across all ROIs
im = ax.imshow(heatmap, cmap='viridis')
ax.set_title('ROI Density (Sum of all ROIs)')
plt.colorbar(im, ax=ax, label='Number of overlapping ROIs')
ax.axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/rois_heatmap.png', dpi=300, bbox_inches='tight')

# Get the first ROI to understand typical structure
if num_rois > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(roi_masks[0], cmap='gray')
    ax.set_title(f'Example ROI (ID: 0)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('tmp_scripts/roi_example.png', dpi=300)

print("ROI visualizations created and saved.")