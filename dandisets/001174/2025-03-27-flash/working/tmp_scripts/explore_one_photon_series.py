import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

OnePhotonSeries = nwb.acquisition["OnePhotonSeries"]
data = OnePhotonSeries.data

# Plot the first 3 frames
num_frames = 3
fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
for i in range(num_frames):
    axes[i].imshow(data[i, :, :], cmap='gray')
    axes[i].set_title(f'Frame {i}')
    axes[i].axis('off')

plt.savefig("tmp_scripts/one_photon_series_frames.png")
plt.close()