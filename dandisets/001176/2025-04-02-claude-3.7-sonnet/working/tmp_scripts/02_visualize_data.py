# Script to visualize key data from the NWB file
# This script loads and visualizes pupil tracking data, treadmill velocity, 
# fluorescence data, and summary images

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/babeee4c-bb8f-4d0b-b898-3edf99244f25/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Create a figure with 4 subplots to display different aspects of the data
fig, axs = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)

# 1. Plot pupil tracking data (pupil radius)
print("Plotting pupil tracking data...")
pupil_tracking = nwb.acquisition["PupilTracking"]
pupil_radius = pupil_tracking["pupil_raw_radius"]
# Get a subset to avoid loading too much data
sample_size = 5000
idx = np.linspace(0, len(pupil_radius.timestamps)-1, sample_size, dtype=int)
timestamps = pupil_radius.timestamps[idx]
radius_data = pupil_radius.data[idx]

axs[0].plot(timestamps, radius_data)
axs[0].set_title("Pupil Radius Over Time")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Pupil Radius (pixels)")

# 2. Plot treadmill velocity data
print("Plotting treadmill velocity data...")
treadmill = nwb.acquisition["treadmill_velocity"]
# Get a subset to avoid loading too much data
sample_size = 5000
idx = np.linspace(0, len(treadmill.timestamps)-1, sample_size, dtype=int)
treadmill_timestamps = treadmill.timestamps[idx]
treadmill_data = treadmill.data[idx]

# Filter out NaN values for plotting
valid_idx = ~np.isnan(treadmill_timestamps) & ~np.isnan(treadmill_data)
axs[1].plot(treadmill_timestamps[valid_idx], treadmill_data[valid_idx])
axs[1].set_title("Treadmill Velocity Over Time")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Velocity (units/s)")

# 3. Plot fluorescence trace (acetylcholine sensor activity)
print("Plotting fluorescence trace...")
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"]
# Get a subset to avoid loading too much data
sample_size = 5000
idx = np.linspace(0, len(fluorescence.timestamps)-1, sample_size, dtype=int)
fluor_timestamps = fluorescence.timestamps[idx]
fluor_data = fluorescence.data[idx, 0]  # Just the first ROI

axs[2].plot(fluor_timestamps, fluor_data)
axs[2].set_title("Acetylcholine Sensor Fluorescence Over Time")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Fluorescence (a.u.)")

# 4. Display summary images from two-photon imaging
print("Displaying summary images...")
summary_images = nwb.processing["ophys"]["SummaryImages_chan1"]
# Get average and correlation images
avg_img = summary_images["average"].data[:]
corr_img = summary_images["correlation"].data[:]

# Normalize for better visualization
avg_img_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
corr_img_norm = (corr_img - corr_img.min()) / (corr_img.max() - corr_img.min())

# Create a two-panel plot within the fourth subplot
ax4 = axs[3]
ax4.axis('off')  # Turn off the main axes

# Create two new axes within ax4's space
ax4_1 = fig.add_subplot(4, 2, 7)
ax4_2 = fig.add_subplot(4, 2, 8)

# Plot the average and correlation images
im1 = ax4_1.imshow(avg_img_norm, cmap='gray')
ax4_1.set_title('Average Image')
ax4_1.axis('off')

im2 = ax4_2.imshow(corr_img_norm, cmap='gray')
ax4_2.set_title('Correlation Image')
ax4_2.axis('off')

plt.savefig("tmp_scripts/data_visualization.png", dpi=300, bbox_inches='tight')
print("Figure saved as tmp_scripts/data_visualization.png")

# Save the correlation and average images separately for better inspection
plt.figure(figsize=(8, 8))
plt.imshow(avg_img_norm, cmap='gray')
plt.title('Average Image')
plt.axis('off')
plt.savefig("tmp_scripts/average_image.png", dpi=300, bbox_inches='tight')
print("Average image saved as tmp_scripts/average_image.png")

plt.figure(figsize=(8, 8))
plt.imshow(corr_img_norm, cmap='gray')
plt.title('Correlation Image')
plt.axis('off')
plt.savefig("tmp_scripts/correlation_image.png", dpi=300, bbox_inches='tight')
print("Correlation image saved as tmp_scripts/correlation_image.png")

# Also plot the image mask to understand the ROIs
plt.figure(figsize=(8, 8))
image_masks = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation1"]["image_mask"].data[:]
# Sum across the first dimension if there are multiple masks
if image_masks.shape[0] > 1:
    combined_mask = np.max(image_masks, axis=0)
else:
    combined_mask = image_masks[0]
plt.imshow(combined_mask, cmap='hot')
plt.title('ROI Masks')
plt.colorbar(label='Mask Value')
plt.axis('off')
plt.savefig("tmp_scripts/roi_masks.png", dpi=300, bbox_inches='tight')
print("ROI masks saved as tmp_scripts/roi_masks.png")