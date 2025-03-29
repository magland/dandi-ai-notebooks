"""
This script explores the OnePhotonSeries data from the NWB file to understand
the structure of the calcium imaging data and visualize sample frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
import os

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information about the OnePhotonSeries
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print("\nOnePhotonSeries Information:")
print(f"Shape: {one_photon_series.data.shape}")
print(f"Data type: {one_photon_series.data.dtype}")
print(f"Rate: {one_photon_series.rate} Hz")
print(f"Starting time: {one_photon_series.starting_time} sec")

# Get sample frames - taking 3 frames at different times
sample_indices = [0, one_photon_series.data.shape[0] // 2, one_photon_series.data.shape[0] - 1]
sample_frames = []

print("\nExtracting sample frames...")
for idx in sample_indices:
    sample_frames.append(one_photon_series.data[idx, :, :])
    
# Plot sample frames
print("Plotting sample frames...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['First Frame', 'Middle Frame', 'Last Frame']

for i, (ax, frame, title) in enumerate(zip(axes, sample_frames, titles)):
    im = ax.imshow(frame, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    
plt.tight_layout()
plt.savefig('tmp_scripts/onephoton_sample_frames.png', dpi=300)
plt.close()

# Plot the average frame
print("Calculating and plotting average frame...")
# Take a subset of frames to calculate the average (every 100th frame to save memory)
frame_indices = np.arange(0, one_photon_series.data.shape[0], 100)
frames_subset = np.array([one_photon_series.data[i, :, :] for i in frame_indices])
avg_frame = np.mean(frames_subset, axis=0)

plt.figure(figsize=(8, 8))
plt.imshow(avg_frame, cmap='gray')
plt.colorbar(label='Average Intensity')
plt.title('Average Frame (sampled every 100th frame)')
plt.axis('off')
plt.savefig('tmp_scripts/onephoton_average_frame.png', dpi=300)
plt.close()

# Plot intensity over time for a few pixels
print("Plotting intensity over time for sample pixels...")
# Select a few pixels
center_row = one_photon_series.data.shape[1] // 2
center_col = one_photon_series.data.shape[2] // 2
pixels = [
    (center_row, center_col),  # center
    (center_row - 50, center_col - 50),  # top-left
    (center_row + 50, center_col + 50),  # bottom-right
]

# Sample every 10th frame to reduce data size
time_indices = np.arange(0, one_photon_series.data.shape[0], 10)
times = time_indices / one_photon_series.rate

plt.figure(figsize=(12, 6))
for i, (row, col) in enumerate(pixels):
    if row >= 0 and row < one_photon_series.data.shape[1] and col >= 0 and col < one_photon_series.data.shape[2]:
        intensities = [one_photon_series.data[t, row, col] for t in time_indices]
        plt.plot(times, intensities, label=f'Pixel ({row}, {col})')

plt.xlabel('Time (s)')
plt.ylabel('Intensity')
plt.title('Pixel Intensity Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/onephoton_pixel_intensity.png', dpi=300)
plt.close()

print("Script completed successfully!")