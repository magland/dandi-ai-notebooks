#!/usr/bin/env python
# This script loads and visualizes paired DAPI and FITC images
# using the correct approach to access the files

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
import os

print("Loading and visualizing paired DAPI and FITC images...")

# Create client and get all assets
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Organize assets by subject
subject_files = {}

for asset in assets:
    path = asset.path
    asset_id = asset.identifier
    
    # Extract subject from path
    subject_id = path.split('/')[0]
    
    if subject_id not in subject_files:
        subject_files[subject_id] = []
    
    subject_files[subject_id].append({
        'path': path,
        'asset_id': asset_id
    })

# Find a subject with multiple files for analysis
print("\nSubjects with multiple files:")
for subject, files in subject_files.items():
    if len(files) > 1:
        print(f"{subject}: {len(files)} files")

# Select a subject for detailed analysis (sub-P1-20240627-A2)
selected_subject = "sub-P1-20240627-A2"
print(f"\nAnalyzing files for subject: {selected_subject}")

for i, file_info in enumerate(subject_files[selected_subject]):
    print(f"{i+1}. {file_info['path']} (asset_id: {file_info['asset_id']})")

# Function to examine an asset and determine its type
def examine_asset(asset_id):
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
    try:
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Get description
        description = nwb.session_description
        
        # Determine type
        is_dapi = "DAPI" in description and "pre" in description
        is_fitc = "FITC" in description and "post" in description
        
        return {
            'asset_id': asset_id,
            'nwb': nwb,
            'description': description[:200] + "...",
            'is_dapi': is_dapi,
            'is_fitc': is_fitc
        }
    except Exception as e:
        print(f"Error examining asset {asset_id}: {str(e)}")
        return None

# Examine each file for the selected subject
print("\nExamining files to determine types...")
file_types = {}

for file_info in subject_files[selected_subject]:
    result = examine_asset(file_info['asset_id'])
    if result:
        print(f"Asset: {file_info['path']}")
        print(f"  Description: {result['description']}")
        print(f"  Is DAPI (pre): {result['is_dapi']}")
        print(f"  Is FITC (post): {result['is_fitc']}")
        
        if result['is_dapi']:
            file_types['dapi'] = result
        elif result['is_fitc']:
            file_types['fitc'] = result
            
# Check if we have both DAPI and FITC files
if 'dapi' in file_types and 'fitc' in file_types:
    print("\nFound both DAPI and FITC files for subject!")
    
    # Function to extract a sample from an image
    def extract_sample(nwb, crop_size=1000):
        img_data = nwb.acquisition["SingleTimePointImaging"]
        shape = img_data.data.shape
        print(f"  Image shape: {shape}")
        
        if len(shape) == 3:  # (frames, height, width)
            n_frames, height, width = shape
            crop_size = min(crop_size, height, width)
            start_h = (height - crop_size) // 2
            start_w = (width - crop_size) // 2
            sample = img_data.data[0, start_h:start_h+crop_size, start_w:start_w+crop_size]
        elif len(shape) == 2:  # (height, width)
            height, width = shape
            crop_size = min(crop_size, height, width)
            start_h = (height - crop_size) // 2
            start_w = (width - crop_size) // 2
            sample = img_data.data[start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        return sample
    
    # Extract samples
    print("Extracting DAPI sample...")
    dapi_sample = extract_sample(file_types['dapi']['nwb'])
    
    print("Extracting FITC sample...")
    fitc_sample = extract_sample(file_types['fitc']['nwb'])
    
    # Normalize for visualization
    def normalize_sample(sample):
        sample_min = np.min(sample)
        sample_max = np.max(sample)
        if sample_min != sample_max:
            return (sample - sample_min) / (sample_max - sample_min)
        else:
            return sample

    dapi_norm = normalize_sample(dapi_sample)
    fitc_norm = normalize_sample(fitc_sample)
    
    # Create comparison figure
    plt.figure(figsize=(16, 8))
    
    # DAPI image (blue)
    plt.subplot(1, 2, 1)
    plt.imshow(dapi_norm, cmap='Blues')
    plt.title("DAPI (pre-exposure, nuclei staining)", fontsize=14)
    plt.axis('off')
    plt.colorbar(label="Normalized Intensity")
    
    # FITC image (green/hot)
    plt.subplot(1, 2, 2)
    plt.imshow(fitc_norm, cmap='hot')
    plt.title("FITC (post-exposure, YoPro-1 uptake)", fontsize=14)
    plt.axis('off')
    plt.colorbar(label="Normalized Intensity")
    
    plt.suptitle(f"Comparison for subject {selected_subject}", fontsize=16)
    plt.tight_layout()
    
    # Save the comparison figure
    plt.savefig("tmp_scripts/dapi_fitc_comparison_final.png", dpi=150, bbox_inches='tight')
    print("Saved comparison image to tmp_scripts/dapi_fitc_comparison_final.png")
    
    # Analyze the images separately to get detailed metrics
    plt.figure(figsize=(10, 8))
    plt.imshow(dapi_norm, cmap='Blues')
    plt.title(f"DAPI image (pre-exposure, nuclei staining)\nSubject: {selected_subject}", fontsize=14)
    plt.colorbar(label="Normalized Intensity")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("tmp_scripts/dapi_final.png", dpi=150, bbox_inches='tight')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(fitc_norm, cmap='hot')
    plt.title(f"FITC image (post-exposure, YoPro-1 uptake)\nSubject: {selected_subject}", fontsize=14)
    plt.colorbar(label="Normalized Intensity")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("tmp_scripts/fitc_final.png", dpi=150, bbox_inches='tight')
    
    # Calculate histogram of intensities
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(dapi_sample.flatten(), bins=100, alpha=0.7, color='blue')
    plt.title("DAPI Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(fitc_sample.flatten(), bins=100, alpha=0.7, color='red')
    plt.title("FITC Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("tmp_scripts/intensity_histograms.png", dpi=150, bbox_inches='tight')
    print("Saved intensity histograms to tmp_scripts/intensity_histograms.png")
    
    # Try to create a composite/overlay image
    # This assumes the images are aligned and same size
    if dapi_norm.shape == fitc_norm.shape:
        print("Creating composite image...")
        
        # Create RGB image with DAPI in blue channel and FITC in green channel
        composite = np.zeros((dapi_norm.shape[0], dapi_norm.shape[1], 3))
        composite[:,:,0] = 0                 # Red channel empty
        composite[:,:,1] = fitc_norm * 0.8   # Green channel for FITC
        composite[:,:,2] = dapi_norm * 0.8   # Blue channel for DAPI
        
        plt.figure(figsize=(10, 8))
        plt.imshow(composite)
        plt.title(f"Composite DAPI (blue) and FITC (green)\nSubject: {selected_subject}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("tmp_scripts/composite_image.png", dpi=150, bbox_inches='tight')
        print("Saved composite image to tmp_scripts/composite_image.png")
    
    # Use dandi-notebook-gen-tools to analyze the plots
    print("\nAnalyzing plots...")
    os.system("dandi-notebook-gen-tools analyze-plot tmp_scripts/dapi_fitc_comparison_final.png")
    
else:
    print("Could not find both DAPI and FITC files for the selected subject.")

print("\nVisualization complete.")