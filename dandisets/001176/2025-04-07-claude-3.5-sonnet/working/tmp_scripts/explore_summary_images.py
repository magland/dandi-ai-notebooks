"""
This script loads and visualizes the summary images from the ophys data:
1. Average image of the stack
2. Correlation image of the stack
3. Image masks overlaid on the average image

This will help us understand the structure of the imaging field and the ROIs.
"""

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/b22180d0-41dc-4091-a334-2e5bd4b5c548/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the summary images and masks
avg_img = nwb.processing['ophys']['SummaryImages_chan1']['average'].data[:]
corr_img = nwb.processing['ophys']['SummaryImages_chan1']['correlation'].data[:]
masks = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation1']['image_mask'].data[:]

# Create figure
plt.figure(figsize=(15, 5))

# Plot average image
plt.subplot(131)
plt.imshow(avg_img, cmap='gray')
plt.title('Average Image')
plt.axis('off')

# Plot correlation image
plt.subplot(132)
plt.imshow(corr_img, cmap='viridis')
plt.title('Correlation Image')
plt.axis('off')

# Plot masks overlaid on average image
plt.subplot(133)
plt.imshow(avg_img, cmap='gray')
# Sum masks across first dimension and overlay
mask_overlay = np.max(masks, axis=0)  # Use max for overlay since masks are 0-1
plt.imshow(mask_overlay, cmap='hot', alpha=0.5)
plt.title('ROI Masks')
plt.axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/summary_images.png', dpi=300, bbox_inches='tight')
plt.close()