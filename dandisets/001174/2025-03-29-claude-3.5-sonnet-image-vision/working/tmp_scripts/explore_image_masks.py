"""
This script explores the image masks from the calcium imaging data to:
1. Visualize the spatial distribution of all cells
2. Create a heatmap showing the overlap of cell masks
3. Display individual cell masks
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the image masks
plane_seg = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
image_masks = plane_seg['image_mask'].data[:]  # shape: [num_masks, num_rows, num_columns]

# Create superimposed image of all masks
all_masks = np.max(image_masks, axis=0)  # Combine all masks using maximum value

# Plot 1: Heatmap of all cells
plt.figure(figsize=(10, 8))
plt.imshow(all_masks, cmap='hot')
plt.colorbar(label='Max Mask Value')
plt.title('Spatial Distribution of All Cells')
plt.savefig('tmp_scripts/all_cells_heatmap.png')
plt.close()

# Plot 2: Individual cell masks (first 6 cells)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(6):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(image_masks[i], cmap='viridis')
    axes[row, col].set_title(f'Cell {i+1}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('tmp_scripts/individual_cells.png')
plt.close()