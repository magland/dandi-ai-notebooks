"""
Visualize raw calcium imaging data from the OnePhotonSeries.
This script will load sample frames from the raw calcium imaging data
and create visualizations to understand the data before ROI segmentation.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file using lindi
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the OnePhotonSeries
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"OnePhotonSeries shape: {one_photon_series.data.shape}")
print(f"Data type: {one_photon_series.data.dtype}")

# Sample a few frames from different time points
num_frames = one_photon_series.data.shape[0]
sample_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]

print(f"Total number of frames: {num_frames}")
print(f"Sampling frames at indices: {sample_indices}")

# Create a figure to display the sample frames
fig, axes = plt.subplots(1, len(sample_indices), figsize=(15, 5))
vmin, vmax = None, None  # Will be computed from the first frame

# Load frames from various timepoints
for i, idx in enumerate(sample_indices):
    # Load the frame data
    frame = one_photon_series.data[idx, :, :]
    
    # For the first frame, compute reasonable min/max values
    if i == 0:
        # Use percentiles to avoid extreme outliers
        vmin = np.percentile(frame, 1)
        vmax = np.percentile(frame, 99)
    
    # Display the frame
    im = axes[i].imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
    axes[i].set_title(f"Frame {idx}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/raw_frames_samples.png', dpi=300)

# Now create a more detailed view of a single frame with ROI overlays
# Choose a middle frame
middle_frame_idx = num_frames // 2
middle_frame = one_photon_series.data[middle_frame_idx, :, :]

# Access the ROI masks
plane_segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
roi_masks = plane_segmentation["image_mask"].data[:]

# Create a figure comparing raw frame with ROI overlays
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Display the raw frame
axes[0].imshow(middle_frame, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title(f"Raw Frame {middle_frame_idx}")
axes[0].axis('off')

# Display the raw frame with ROI contours overlaid
axes[1].imshow(middle_frame, cmap='gray', vmin=vmin, vmax=vmax)

# Overlay ROI contours
# Note: We need to resize the masks to match the frame size if they differ
if roi_masks.shape[1:] != middle_frame.shape:
    print(f"Warning: ROI mask size ({roi_masks.shape[1:]}) doesn't match frame size ({middle_frame.shape})")
    print("ROI overlay may not be perfectly aligned")

for i in range(roi_masks.shape[0]):
    # Get the binary mask
    mask = roi_masks[i]
    
    # For visualization only, we'll just use the center coordinates to mark ROIs
    if mask.max() > 0:  # ensure the mask isn't empty
        # Get mask centroid (approximate)
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:  # Check there are indeed coordinates
            y_center = int(np.mean(coords[0]))
            x_center = int(np.mean(coords[1]))
            
            # Scale to match frame dimensions if needed
            scale_y = middle_frame.shape[0] / mask.shape[0]
            scale_x = middle_frame.shape[1] / mask.shape[1]
            
            y_center_scaled = int(y_center * scale_y)
            x_center_scaled = int(x_center * scale_x)
            
            # Draw a circle at the ROI center
            circle = plt.Circle((x_center_scaled, y_center_scaled), 5, color='red', fill=False, linewidth=1.5)
            axes[1].add_patch(circle)
            
            # Add ROI number
            axes[1].text(x_center_scaled, y_center_scaled, str(i), color='white', 
                         fontsize=8, ha='center', va='center')

axes[1].set_title(f"Frame {middle_frame_idx} with ROI Centers")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('tmp_scripts/raw_frame_with_rois.png', dpi=300)

print("Raw calcium imaging visualizations created and saved.")