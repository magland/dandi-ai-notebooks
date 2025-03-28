#!/usr/bin/env python
# This script focuses on exploring the image data in the NWB files of Dandiset 001276
# It will extract images and analyze their properties

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
import os

print("Starting image data exploration...")

# Create client and get assets
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Categorize assets
# Structure will be: {"subject_id": {"DAPI": [list of DAPI files], "FITC": [list of FITC files]}}
organized_assets = {}

for asset in assets[:20]:  # Limit to first 20 assets for now
    path = asset.path
    asset_id = asset.identifier
    
    # Extract subject info from path
    parts = path.split('/')
    subject_id = parts[0]  # e.g., 'sub-P1-20240627-A2'
    
    if subject_id not in organized_assets:
        organized_assets[subject_id] = {"DAPI": [], "FITC": [], "Other": []}
    
    # Try to identify if this is DAPI (pre) or FITC (post) from metadata
    try:
        print(f"Checking file: {path}")
        lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
        
        # Just open the file to check session description
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        desc = nwb.session_description
        if "DAPI" in desc and "pre" in desc:
            category = "DAPI"
        elif "FITC" in desc and "post" in desc:
            category = "FITC"
        else:
            category = "Other"
            
        organized_assets[subject_id][category].append({
            "path": path,
            "asset_id": asset_id,
            "url": lindi_url
        })
        print(f"  Categorized as: {category}")
        
    except Exception as e:
        print(f"  Error processing {path}: {str(e)}")
        continue

# Print the organization summary
print("\nOrganized Assets Summary:")
for subject, categories in organized_assets.items():
    dapi_count = len(categories["DAPI"])
    fitc_count = len(categories["FITC"])
    other_count = len(categories["Other"])
    
    if dapi_count > 0 or fitc_count > 0:
        print(f"Subject {subject}: DAPI (pre): {dapi_count}, FITC (post): {fitc_count}, Other: {other_count}")

# Find a matching DAPI/FITC pair for the same subject
matched_pairs = []
for subject, categories in organized_assets.items():
    if len(categories["DAPI"]) > 0 and len(categories["FITC"]) > 0:
        matched_pairs.append({
            "subject": subject,
            "dapi": categories["DAPI"][0],  # Take first DAPI file
            "fitc": categories["FITC"][0]   # Take first FITC file
        })

if len(matched_pairs) == 0:
    print("\nNo matching DAPI/FITC pairs found.")
else:
    # Select the first matched pair
    selected_pair = matched_pairs[0]
    print(f"\nSelected pair for subject: {selected_pair['subject']}")
    print(f"  DAPI file: {selected_pair['dapi']['path']}")
    print(f"  FITC file: {selected_pair['fitc']['path']}")
    
    # Extract sample images from both files
    for img_type in ["dapi", "fitc"]:
        try:
            file_info = selected_pair[img_type]
            print(f"\nExtracting image from {img_type.upper()} file: {file_info['path']}")
            
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
                # Extract a central 1000x1000 region from the first frame
                # Make sure the crop size isn't larger than the image
                crop_size = min(1000, height, width)
                start_h = (height - crop_size) // 2
                start_w = (width - crop_size) // 2
                
                # Extract the central crop
                sample = img_data.data[0, 
                                      start_h:start_h+crop_size, 
                                      start_w:start_w+crop_size]
                
            elif len(shape) == 2:  # (height, width)
                height, width = shape
                # Extract a central 1000x1000 region
                # Make sure the crop size isn't larger than the image
                crop_size = min(1000, height, width)
                start_h = (height - crop_size) // 2
                start_w = (width - crop_size) // 2
                
                # Extract the central crop
                sample = img_data.data[start_h:start_h+crop_size, 
                                     start_w:start_w+crop_size]
            
            # Normalize for visualization
            sample_min = np.min(sample)
            sample_max = np.max(sample)
            if sample_min != sample_max:  # Avoid division by zero
                normalized = (sample - sample_min) / (sample_max - sample_min)
            else:
                normalized = sample
                
            # Create and save image
            plt.figure(figsize=(10, 8))
            plt.imshow(normalized, cmap='viridis')
            plt.title(f"{img_type.upper()} image from {file_info['path']}")
            plt.colorbar(label="Normalized Intensity")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            output_file = f"tmp_scripts/{img_type}_sample.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved sample to {output_file}")
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_type} file: {str(e)}")
            import traceback
            print(traceback.format_exc())

print("\nImage data exploration complete.")