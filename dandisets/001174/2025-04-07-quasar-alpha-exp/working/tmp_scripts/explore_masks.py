# This script loads ROI segmentation masks from the NWB file,
# computes a maximum intensity projection, and saves a heatmap
# visualizing all segmented regions at once.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lindi
import pynwb

sns.set_theme()

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

plane_seg = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
masks = plane_seg["image_mask"].data  # shape: [num_cells, rows, cols]

max_mask = np.max(masks[:], axis=0)

plt.figure(figsize=(6, 6))
plt.imshow(max_mask, cmap="hot")
plt.title("Max Projection of Segmentation Masks (all ROIs)")
plt.colorbar(label="Max Mask Value")
plt.axis("off")
plt.savefig("tmp_scripts/masks_projection.png", bbox_inches='tight')
plt.close()