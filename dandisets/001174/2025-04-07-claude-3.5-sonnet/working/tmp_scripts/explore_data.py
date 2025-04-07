"""
Initial exploration of the calcium imaging data from sub-Q NWB file.
This script will:
1. Load and examine the basic data structure
2. Plot a sample frame from the one-photon series
3. Visualize the ROI masks
4. Plot example fluorescence traces
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Extract basic information
print("Session description:", nwb.session_description)
print("Subject ID:", nwb.subject.subject_id)
print("Species:", nwb.subject.species)
print("Recording rate:", nwb.acquisition["OnePhotonSeries"].rate, "Hz")
print("Number of frames:", nwb.acquisition["OnePhotonSeries"].data.shape[0])
print("Image dimensions:", nwb.acquisition["OnePhotonSeries"].data.shape[1:])
print("Number of ROIs:", nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]["image_mask"].data.shape[0])

# Plot a sample frame (frame 1000)
print("\nLoading and plotting sample frame...")
sample_frame = nwb.acquisition["OnePhotonSeries"].data[1000]  # This loads just one frame
plt.figure(figsize=(8, 6))
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Intensity')
plt.title('Sample Frame from One-Photon Series')
plt.savefig('tmp_scripts/sample_frame.png')
plt.close()

# Visualize ROI masks
print("Processing ROI masks...")
# Load masks for first 5 ROIs to keep memory usage manageable
image_masks = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]["image_mask"].data[:5]
combined_mask = np.zeros_like(image_masks[0])
for mask in image_masks:
    combined_mask = np.maximum(combined_mask, mask)

plt.figure(figsize=(8, 6))
plt.imshow(combined_mask, cmap='hot')
plt.colorbar(label='Mask Value')
plt.title('Combined ROI Masks (First 5 ROIs)')
plt.savefig('tmp_scripts/roi_masks.png')
plt.close()

# Plot fluorescence traces for first 5 ROIs
print("Processing fluorescence traces...")
fluorescence_data = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"].data[:500, :5]  # First 500 timepoints, first 5 ROIs
time_points = np.arange(500) / nwb.acquisition["OnePhotonSeries"].rate

plt.figure(figsize=(12, 8))
for i in range(5):
    plt.plot(time_points, fluorescence_data[:, i], label=f'ROI {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence')
plt.title('Fluorescence Traces for First 5 ROIs (First 500 Timepoints)')
plt.legend()
plt.savefig('tmp_scripts/fluorescence_traces.png')
plt.close()

print("Done! Generated three plots: sample_frame.png, roi_masks.png, and fluorescence_traces.png")