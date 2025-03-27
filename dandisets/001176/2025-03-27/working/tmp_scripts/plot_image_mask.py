# Explore PlaneSegmentation1 and create a plot of the image masks
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get image_mask data from PlaneSegmentation1
ophys = nwb.processing["ophys"]
ImageSegmentation = ophys["ImageSegmentation"]
PlaneSegmentation1 = ImageSegmentation["PlaneSegmentation1"]
image_mask_data = PlaneSegmentation1["image_mask"].data[:]

# Superimpose all image masks using np.max
superimposed_mask = np.max(image_mask_data, axis=0)

# Plot the superimposed image mask
plt.figure(figsize=(8, 8))
sns.heatmap(superimposed_mask, cmap="viridis") # removed deprecated plt.style.use('seaborn') and used sns.set_theme()
plt.title("Superimposed Image Masks")
plt.xlabel("Column")
plt.ylabel("Row")
plt.savefig("tmp_scripts/image_mask.png")
plt.close()