#!/usr/bin/env python
# This script finds pairs of DAPI and FITC images for the same subject/experiment
# and extracts samples for comparison and analysis

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

print("Finding pairs of DAPI and FITC images for the same subjects...")

# Create client and get all assets
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Function to determine image type from description
def determine_image_type(description):
    if "FITC" in description and "post" in description:
        return "FITC"
    elif "DAPI" in description and "pre" in description:
        return "DAPI"
    else:
        return "Unknown"

# Process all files and organize by subject
subjects = {}
for i, asset in enumerate(assets):
    if i % 20 == 0:
        print(f"Processing file {i+1}/{len(assets)}...")
    
    path = asset.path
    asset_id = asset.identifier
    
    # Extract subject ID from path
    parts = path.split('/')
    subject_id = parts[0]
    
    if subject_id not in subjects:
        subjects[subject_id] = {"DAPI": [], "FITC": [], "Unknown": []}
    
    try:
        # Get file metadata
        lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Determine image type
        description = nwb.session_description
        image_type = determine_image_type(description)
        
        # Add to the appropriate category
        subjects[subject_id][image_type].append({
            "path": path,
            "asset_id": asset_id,
            "url": lindi_url
        })
        
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")

# Find subjects with both DAPI and FITC images
print("\nFound subjects with both DAPI and FITC images:")
paired_subjects = []

for subject, data in subjects.items():
    if len(data["DAPI"]) > 0 and len(data["FITC"]) > 0:
        paired_subjects.append(subject)
        print(f"- {subject}: {len(data['DAPI'])} DAPI, {len(data['FITC'])} FITC")

if not paired_subjects:
    print("No subjects with both DAPI and FITC images found.")
    exit()

# Select the first subject with pairs for analysis
selected_subject = paired_subjects[0]
subject_data = subjects[selected_subject]

print(f"\nAnalyzing paired images for subject: {selected_subject}")
print(f"DAPI files: {len(subject_data['DAPI'])}")
print(f"FITC files: {len(subject_data['FITC'])}")

# Extract samples from the first DAPI and first FITC file
dapi_file = subject_data["DAPI"][0]
fitc_file = subject_data["FITC"][0]

print(f"\nDAPI file: {dapi_file['path']}")
print(f"FITC file: {fitc_file['path']}")

# Function to extract a central image sample
def extract_sample(file_info, crop_size=1000):
    try:
        # Open the file
        f = lindi.LindiH5pyFile.from_lindi_file(file_info['url'])
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Get the image data
        img_data = nwb.acquisition["SingleTimePointImaging"]
        shape = img_data.data.shape
        print(f"Image shape: {shape}")
        
        # Extract a small central section to avoid memory issues
        if len(shape) == 3:  # (frames, height, width)
            n_frames, height, width = shape
            # Make sure the crop size isn't larger than the image
            crop_size = min(crop_size, height, width)
            start_h = (height - crop_size) // 2
            start_w = (width - crop_size) // 2
            
            # Extract the central crop from the first frame
            sample = img_data.data[0, 
                                   start_h:start_h+crop_size, 
                                   start_w:start_w+crop_size]
            
        elif len(shape) == 2:  # (height, width)
            height, width = shape
            # Make sure the crop size isn't larger than the image
            crop_size = min(crop_size, height, width)
            start_h = (height - crop_size) // 2
            start_w = (width - crop_size) // 2
            
            # Extract the central crop
            sample = img_data.data[start_h:start_h+crop_size, 
                                  start_w:start_w+crop_size]
                                  
        return sample, (start_h, start_w, crop_size)
        
    except Exception as e:
        print(f"Error extracting sample: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

# Extract samples from both files (using same crop area if possible)
print("\nExtracting samples...")
dapi_sample, dapi_crop_info = extract_sample(dapi_file)
fitc_sample, fitc_crop_info = extract_sample(fitc_file)

if dapi_sample is not None and fitc_sample is not None:
    # Create a side-by-side comparison
    plt.figure(figsize=(16, 8))
    
    # DAPI image (pre-exposure)
    plt.subplot(1, 2, 1)
    dapi_min = np.min(dapi_sample)
    dapi_max = np.max(dapi_sample)
    if dapi_min != dapi_max:
        dapi_norm = (dapi_sample - dapi_min) / (dapi_max - dapi_min)
    else:
        dapi_norm = dapi_sample
    
    plt.imshow(dapi_norm, cmap='viridis')
    plt.title("DAPI (pre-exposure)", fontsize=14)
    plt.axis('off')
    plt.colorbar(label="Normalized Intensity")
    
    # FITC image (post-exposure)
    plt.subplot(1, 2, 2)
    fitc_min = np.min(fitc_sample)
    fitc_max = np.max(fitc_sample)
    if fitc_min != fitc_max:
        fitc_norm = (fitc_sample - fitc_min) / (fitc_max - fitc_min)
    else:
        fitc_norm = fitc_sample
    
    plt.imshow(fitc_norm, cmap='hot')
    plt.title("FITC (post-exposure)", fontsize=14)
    plt.axis('off')
    plt.colorbar(label="Normalized Intensity")
    
    plt.suptitle(f"Comparison of DAPI and FITC images for {selected_subject}", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("tmp_scripts/dapi_fitc_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved comparison image to tmp_scripts/dapi_fitc_comparison.png")
    plt.close()
    
    # Also save individual images for reference
    plt.figure(figsize=(10, 8))
    plt.imshow(dapi_norm, cmap='viridis')
    plt.title(f"DAPI image from {dapi_file['path']}", fontsize=14)
    plt.colorbar(label="Normalized Intensity")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("tmp_scripts/dapi_sample.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(fitc_norm, cmap='hot')
    plt.title(f"FITC image from {fitc_file['path']}", fontsize=14)
    plt.colorbar(label="Normalized Intensity")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("tmp_scripts/fitc_sample_normalized.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional Analysis - Compute difference to highlight permeabilization
    # This assumes that the DAPI and FITC images are registered
    try:
        if dapi_norm.shape == fitc_norm.shape:
            # Create a difference image to highlight areas with high FITC but low DAPI
            diff_img = fitc_norm - dapi_norm
            
            plt.figure(figsize=(10, 8))
            plt.imshow(diff_img, cmap='RdBu_r')
            plt.title("Difference (FITC - DAPI)", fontsize=14)
            plt.colorbar(label="Intensity Difference")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("tmp_scripts/difference_image.png", dpi=150, bbox_inches='tight')
            print("Saved difference image to tmp_scripts/difference_image.png")
            plt.close()
    except Exception as e:
        print(f"Error creating difference image: {str(e)}")
else:
    print("Could not extract samples from both files.")
    
print("\nPaired image analysis complete.")