# Script to compare summary images and image masks across subjects
# Since there are issues with the behavioral data in some files,
# let's focus on comparing the imaging results

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# File paths dictionary for different subjects
file_urls = {
    "subject_23892": "https://lindi.neurosift.org/dandi/dandisets/001176/assets/babeee4c-bb8f-4d0b-b898-3edf99244f25/nwb.lindi.json",
    "subject_23893": "https://lindi.neurosift.org/dandi/dandisets/001176/assets/38cc792e-0f9b-4255-a57f-78fd6c1315a8/nwb.lindi.json",
    "subject_16": "https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json"
}

# Create a figure to compare summary images and ROI masks
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig)

# Process each subject
for i, (subject_name, file_url) in enumerate(file_urls.items()):
    print(f"\nProcessing {subject_name}...")
    
    # Load the NWB file
    try:
        f = lindi.LindiH5pyFile.from_lindi_file(file_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Print basic metadata
        print(f"Subject ID: {nwb.subject.subject_id}")
        print(f"Session ID: {nwb.identifier}")
        print(f"Experiment description: {nwb.experiment_description}")
        
        # Get average image
        if "ophys" in nwb.processing and "SummaryImages_chan1" in nwb.processing["ophys"].data_interfaces:
            summary_images = nwb.processing["ophys"]["SummaryImages_chan1"]
            avg_img = summary_images["average"].data[:]
            
            # Normalize for better visualization
            avg_img_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
            
            # Plot average image in the first row
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(avg_img_norm, cmap='gray')
            ax1.set_title(f'Average Image - {subject_name}')
            ax1.axis('off')
            
            # Get ROI masks if available
            if "ImageSegmentation" in nwb.processing["ophys"].data_interfaces:
                img_seg = nwb.processing["ophys"]["ImageSegmentation"]
                # Get the first plane segmentation
                plane_name = list(img_seg.plane_segmentations.keys())[0]
                plane_seg = img_seg[plane_name]
                
                if "image_mask" in plane_seg:
                    masks = plane_seg["image_mask"].data[:]
                    
                    # Combine masks (take max across masks if multiple)
                    if masks.shape[0] > 1:
                        combined_mask = np.max(masks, axis=0)
                    else:
                        combined_mask = masks[0]
                    
                    # Plot ROI masks in the second row
                    ax2 = fig.add_subplot(gs[1, i])
                    ax2.imshow(combined_mask, cmap='hot')
                    ax2.set_title(f'ROI Masks - {subject_name}')
                    ax2.axis('off')
                else:
                    print(f"No image masks found for {subject_name}")
            else:
                print(f"No ImageSegmentation found for {subject_name}")
        else:
            print(f"No summary images found for {subject_name}")
            
    except Exception as e:
        print(f"Error processing {subject_name}: {e}")

plt.tight_layout()
plt.savefig("tmp_scripts/multi_subject_imaging_comparison.png", dpi=300)
print("\nMulti-subject comparison saved as tmp_scripts/multi_subject_imaging_comparison.png")