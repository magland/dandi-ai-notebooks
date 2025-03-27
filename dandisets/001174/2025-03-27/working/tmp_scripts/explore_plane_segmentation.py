import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

PlaneSegmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
image_mask = PlaneSegmentation["image_mask"].data

# Plot the maximum of all image masks
max_image_mask = np.max(image_mask[:], axis=0)  # corrected slicing
plt.imshow(max_image_mask, cmap='viridis')
plt.title('Maximum of Image Masks')
plt.colorbar()
plt.savefig("tmp_scripts/plane_segmentation_image_masks.png")
plt.close()