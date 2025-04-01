"""
Script to explore the OnePhotonSeries data from an NWB file containing calcium imaging data.
This will load the NWB file, examine the structure, and plot sample frames.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001174/assets/ac161d0e-7642-48bd-9bef-8eff59319d48/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the OnePhotonSeries data
one_photon = nwb.acquisition["OnePhotonSeries"]
print(f"Data shape: {one_photon.data.shape}")  # [frames, height, width]
print(f"Frame rate: {one_photon.rate} Hz")

# Plot sample frames from different times
sample_indices = [0, 1500, 3000]  # Start, middle, and end of recording
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    frame = one_photon.data[idx, :, :]
    axes[i].imshow(frame, cmap='gray')
    axes[i].set_title(f"Frame {idx}")
    axes[i].axis('off')
plt.suptitle("Sample Frames from OnePhotonSeries", fontsize=12)
plt.tight_layout()
plt.savefig('tmp_scripts/one_photon_samples.png')
plt.close()  # Important - don't display window to avoid hanging

# Plot fluorescence time series for first 200 frames to see data quality
num_frames = 200
time = np.arange(num_frames) / one_photon.rate
sample_cells = range(3)  # Plot first 3 cells for visualization

fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]
plt.figure(figsize=(10, 4))
for cell in sample_cells:
    plt.plot(time, fluorescence.data[:num_frames, cell], label=f"Cell {cell}")
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.title("Fluorescence Time Series (first 200 frames)")
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/fluorescence_samples.png')
plt.close()  # Important - don't display window to avoid hanging