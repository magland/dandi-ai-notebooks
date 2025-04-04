# Plots the image masks.
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

image_mask = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation1"]["image_mask"].data[:]
# Take the maximum across the first dimension (all masks)
max_image_mask = np.max(image_mask, axis=0)

plt.figure(figsize=(8, 8))
sns.heatmap(max_image_mask, cmap="viridis")
plt.title("Maximum of Image Masks")
plt.savefig("tmp_scripts/image_masks.png")
plt.close()