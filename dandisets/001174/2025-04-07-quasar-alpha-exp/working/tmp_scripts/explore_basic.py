# This script explores the NWB file by:
# - Printing session and subject metadata
# - Plotting average calcium imaging frame
# - Plotting average ROI fluorescence trace
# Output PNG files are saved to avoid windowed display hanging.

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lindi
import pynwb

sns.set_theme()

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("Session description:", nwb.session_description)
print("Session start time:", nwb.session_start_time)
print("Subject ID:", nwb.subject.subject_id)
print("Subject species:", nwb.subject.species)
print("Acquisition keys:", list(nwb.acquisition.keys()))
print("Processing modules:", list(nwb.processing.keys()))

# Plot mean frame of imaging data
ophys_data = nwb.acquisition["OnePhotonSeries"].data
num_frames = ophys_data.shape[0]

# pick a subsample of frames to avoid large memory use
sample_indices = np.linspace(0, num_frames - 1, min(500, num_frames)).astype(int)
mean_frame = np.zeros(ophys_data.shape[1:], dtype=np.float32)
for idx in sample_indices:
    mean_frame += ophys_data[idx, :, :]
mean_frame /= len(sample_indices)

plt.figure(figsize=(6, 6))
plt.imshow(mean_frame, cmap="gray")
plt.title("Mean Calcium Imaging Frame ({} samples)".format(len(sample_indices)))
plt.axis("off")
plt.savefig("tmp_scripts/mean_frame.png", bbox_inches='tight')
plt.close()

# Plot average fluorescence trace over all ROIs
rrs_data = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"].data
num_timepoints = rrs_data.shape[0]

# sample all timepoints (since ~6k is decently small)
mean_trace = np.mean(rrs_data[:,:], axis=1)

plt.figure(figsize=(10, 4))
plt.plot(np.arange(num_timepoints) / 10.01, mean_trace)
plt.xlabel("Time (s)")
plt.ylabel("Mean Fluorescence (a.u.)")
plt.title("Average Fluorescence Trace Over ROIs")
plt.tight_layout()
plt.savefig("tmp_scripts/mean_fluorescence_trace.png")
plt.close()