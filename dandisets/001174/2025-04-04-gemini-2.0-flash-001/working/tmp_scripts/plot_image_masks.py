# Plots the image masks from the PlaneSegmentation object in the NWB file.

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the PlaneSegmentation object
PlaneSegmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]

# Get the image masks data
image_masks = PlaneSegmentation["image_mask"].data

# Plot the image masks superimposed on each other using a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(np.max(image_masks[:], axis=0), cmap="viridis")
plt.title("Image Masks (Superimposed)")
plt.xlabel("Column")
plt.ylabel("Row")
plt.colorbar(label="Max Value")
plt.savefig("tmp_scripts/image_masks.png")
plt.close()