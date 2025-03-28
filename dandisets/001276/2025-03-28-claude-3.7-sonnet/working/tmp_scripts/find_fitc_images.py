#!/usr/bin/env python
# This script specifically looks for FITC images in the dataset
# and extracts samples for visualization

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

print("Looking for FITC (post-exposure) images in the dataset...")

# Create client and get all assets
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Function to check if a file contains FITC data
def check_file(asset, debug=False):
    path = asset.path
    asset_id = asset.identifier
    
    try:
        if debug:
            print(f"Examining file: {path}")
        
        lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
        
        # Open the file
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Check session description for key terms
        desc = nwb.session_description
        
        # Look for specific indicators of file type
        is_fitc = False
        if "FITC" in desc:
            is_fitc = True
            
        if debug:
            print(f"Description excerpt: {desc[:200]}...")
            print(f"Is FITC: {is_fitc}")
            
        return {
            "path": path,
            "asset_id": asset_id,
            "url": lindi_url,
            "is_fitc": is_fitc,
            "description": desc
        }
        
    except Exception as e:
        if debug:
            print(f"Error with file {path}: {str(e)}")
        return None
        
# Check first 10 files with debug output
print("\nChecking first 10 files for debugging:")
for i, asset in enumerate(assets[:10]):
    result = check_file(asset, debug=True)
    print("-" * 50)
    
# Now scan all files to find FITC images
print("\nScanning all files to find FITC images...")
fitc_files = []
dapi_files = []

for i, asset in enumerate(assets):
    if i % 10 == 0:
        print(f"Checking file {i}/{len(assets)}...")
    
    result = check_file(asset)
    if result:
        if result["is_fitc"]:
            fitc_files.append(result)
        else:
            dapi_files.append(result)
            
print(f"\nFound {len(fitc_files)} FITC files and {len(dapi_files)} DAPI files")

# If we found any FITC files
if fitc_files:
    print("\nExample FITC files:")
    for i, file in enumerate(fitc_files[:5]):  # Show first 5
        print(f"{i+1}. {file['path']}")
        
    # Extract and display a sample from the first FITC file
    if fitc_files:
        selected_file = fitc_files[0]
        print(f"\nExtracting sample from: {selected_file['path']}")
        
        try:
            # Open the file
            f = lindi.LindiH5pyFile.from_lindi_file(selected_file['url'])
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
            if sample_min != sample_max:
                normalized = (sample - sample_min) / (sample_max - sample_min)
            else:
                normalized = sample
                
            # Create and save image
            plt.figure(figsize=(10, 8))
            plt.imshow(normalized, cmap='hot')  # Use 'hot' colormap for FITC images
            plt.title(f"FITC image sample from {selected_file['path']}")
            plt.colorbar(label="Normalized Intensity")
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            output_file = f"tmp_scripts/fitc_sample.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved FITC sample to {output_file}")
            plt.close()
            
        except Exception as e:
            print(f"Error processing FITC file: {str(e)}")
            import traceback
            print(traceback.format_exc())
else:
    print("No FITC files found. Trying different search criteria...")
    
    # For debugging: look for any mention of "post" in descriptions
    post_files = []
    for file in dapi_files:
        if "post" in file["description"].lower():
            post_files.append(file)
    
    print(f"Found {len(post_files)} files containing 'post' in the description.")
    if post_files:
        for i, file in enumerate(post_files[:5]):
            print(f"{i+1}. {file['path']}")
            print(f"   Description excerpt: {file['description'][:200]}...")
    
print("\nFITC image search complete.")