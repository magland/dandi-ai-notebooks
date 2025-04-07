# This script loads segmentation masks (PlaneSegmentation1) from NWB file,
# and visualizes the 2D mask as a grayscale heatmap.
# Saved as tmp_scripts/segmentation_masks.png

import matplotlib.pyplot as plt
import lindi
import pynwb
import numpy as np

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/be84b6ff-7016-4ed8-af63-aa0e07c02530/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwbfile = pynwb.NWBHDF5IO(file=f, mode='r').read()

masks = nwbfile.processing['ophys']['ImageSegmentation']['PlaneSegmentation1']['image_mask'].data[:]
mask = masks[0]  # First and only mask in this case

plt.figure(figsize=(5,5))
plt.imshow(mask, cmap='gray')
plt.title('Segmentation Mask (ROI)')
plt.axis('off')
plt.savefig("tmp_scripts/segmentation_masks.png")