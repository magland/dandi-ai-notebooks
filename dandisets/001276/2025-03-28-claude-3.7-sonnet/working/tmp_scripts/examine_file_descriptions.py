#!/usr/bin/env python
# This script examines file descriptions in detail to understand how to categorize them
# and find pairs of DAPI and FITC images

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
from collections import defaultdict
import re
import os

print("Examining file descriptions to understand image types...")

# Create client and get all assets
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# Function to extract key parts from file path
def parse_file_path(path):
    parts = path.split('/')
    
    # Extract subject ID and obj ID if present
    subject_id = parts[0]
    
    obj_id = None
    if 'obj-' in parts[1]:
        match = re.search(r'obj-([a-z0-9]+)', parts[1])
        if match:
            obj_id = match.group(1)
    
    return {
        'subject_id': subject_id,
        'obj_id': obj_id,
        'full_path': path
    }

# Extract and save descriptions for a sample of files
descriptions = []
subject_files = defaultdict(list)  # Organize files by subject

for i, asset in enumerate(assets[:50]):  # Limit to first 50 for speed
    if i % 10 == 0:
        print(f"Processing file {i+1}/50...")
    
    path = asset.path
    asset_id = asset.identifier
    path_info = parse_file_path(path)
    
    try:
        # Get file metadata
        lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Get the description
        description = nwb.session_description
        
        # Save the description and related info
        desc_info = {
            'path': path,
            'subject_id': path_info['subject_id'],
            'obj_id': path_info['obj_id'],
            'description': description
        }
        
        descriptions.append(desc_info)
        subject_files[path_info['subject_id']].append(desc_info)
        
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")

# Save full descriptions to a file for reference
with open('tmp_scripts/file_descriptions.txt', 'w') as f:
    for i, desc in enumerate(descriptions):
        f.write(f"File {i+1}: {desc['path']}\n")
        f.write(f"Subject: {desc['subject_id']}, Object: {desc['obj_id']}\n")
        f.write(f"Description: {desc['description'][:500]}...\n")  # First 500 chars
        f.write("-" * 80 + "\n\n")

print(f"\nSaved {len(descriptions)} file descriptions to tmp_scripts/file_descriptions.txt")

# Extract key patterns from descriptions
channel_patterns = {}
phase_patterns = {}

for desc in descriptions:
    description = desc['description']
    
    # Extract channel information
    if "Fluorescent Channel: DAPI" in description:
        channel = "DAPI"
    elif "Fluorescent Channel: FITC" in description:
        channel = "FITC"
    else:
        channel = "Unknown"
    
    # Extract phase information
    if "Phase: pre" in description:
        phase = "pre"
    elif "Phase: post" in description:
        phase = "post"
    else:
        phase = "Unknown"
    
    # Save pattern
    pattern = f"{channel}_{phase}"
    if pattern not in channel_patterns:
        channel_patterns[pattern] = []
    channel_patterns[pattern].append(desc['path'])
    
    # Also save by subject for later analysis
    subject = desc['subject_id']
    if subject not in phase_patterns:
        phase_patterns[subject] = []
    phase_patterns[subject].append(f"{channel}_{phase}")

# Print summary of patterns
print("\nChannel and phase patterns found:")
for pattern, files in channel_patterns.items():
    print(f"{pattern}: {len(files)} files")
    print(f"  Example: {files[0]}")

# Look for subjects with multiple image types
print("\nSubjects with multiple image types:")
for subject, patterns in phase_patterns.items():
    if len(set(patterns)) > 1:
        print(f"{subject}: {set(patterns)}")

# Look for pre/post pairs
print("\nSearching for pre/post pairs within subjects...")
pairs_found = []

for subject, files in subject_files.items():
    # Group files by combination of fluorescent channel and phase
    subject_patterns = {}
    for file in files:
        desc = file['description']
        
        # Extract channel
        if "Fluorescent Channel: DAPI" in desc:
            channel = "DAPI"
        elif "Fluorescent Channel: FITC" in desc:
            channel = "FITC"
        else:
            continue  # Skip if we can't identify the channel
            
        # Extract phase
        if "Phase: pre" in desc:
            phase = "pre"
        elif "Phase: post" in desc:
            phase = "post"
        else:
            continue  # Skip if we can't identify the phase
            
        key = f"{channel}_{phase}"
        if key not in subject_patterns:
            subject_patterns[key] = []
        subject_patterns[key].append(file)
    
    # Check for pre/post pairs
    if "DAPI_pre" in subject_patterns and "FITC_post" in subject_patterns:
        pairs_found.append({
            'subject': subject,
            'dapi_pre': subject_patterns["DAPI_pre"][0],   # Take the first one
            'fitc_post': subject_patterns["FITC_post"][0]  # Take the first one
        })

print(f"Found {len(pairs_found)} subjects with DAPI_pre/FITC_post pairs")

if pairs_found:
    # Print details of the first few pairs
    for i, pair in enumerate(pairs_found[:3]):
        print(f"\nPair {i+1}: Subject {pair['subject']}")
        print(f"  DAPI (pre): {pair['dapi_pre']['path']}")
        print(f"  FITC (post): {pair['fitc_post']['path']}")
    
    # Extract and display sample images from the first pair
    selected_pair = pairs_found[0]
    dapi_file = selected_pair['dapi_pre']
    fitc_file = selected_pair['fitc_post']
    
    print(f"\nExtracting sample images for comparison from subject {selected_pair['subject']}...")
    
    # Extract samples for comparison
    try:
        # DAPI (pre) image
        lindi_url_dapi = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{dapi_file['path'].split('/')[1].split('_obj-')[0]}_{dapi_file['obj_id']}/nwb.lindi.json"
        f_dapi = lindi.LindiH5pyFile.from_lindi_file(lindi_url_dapi)
        nwb_dapi = pynwb.NWBHDF5IO(file=f_dapi, mode='r').read()
        
        # FITC (post) image
        lindi_url_fitc = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{fitc_file['path'].split('/')[1].split('_obj-')[0]}_{fitc_file['obj_id']}/nwb.lindi.json"
        f_fitc = lindi.LindiH5pyFile.from_lindi_file(lindi_url_fitc)
        nwb_fitc = pynwb.NWBHDF5IO(file=f_fitc, mode='r').read()
        
        # Extract small central regions from both images
        def extract_sample(nwb, crop_size=1000):
            img_data = nwb.acquisition["SingleTimePointImaging"]
            shape = img_data.data.shape
            
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
        
        print("Extracting DAPI sample...")
        dapi_sample = extract_sample(nwb_dapi)
        print(f"DAPI sample shape: {dapi_sample.shape}")
        
        print("Extracting FITC sample...")
        fitc_sample = extract_sample(nwb_fitc)
        print(f"FITC sample shape: {fitc_sample.shape}")
        
        # Normalize samples for visualization
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
        
        plt.subplot(1, 2, 1)
        plt.imshow(dapi_norm, cmap='viridis')
        plt.title("DAPI (pre-exposure)", fontsize=14)
        plt.axis('off')
        plt.colorbar(label="Normalized Intensity")
        
        plt.subplot(1, 2, 2)
        plt.imshow(fitc_norm, cmap='hot')
        plt.title("FITC (post-exposure)", fontsize=14)
        plt.axis('off')
        plt.colorbar(label="Normalized Intensity")
        
        plt.suptitle(f"Comparison for subject {selected_pair['subject']}", fontsize=16)
        plt.tight_layout()
        
        # Save the comparison figure
        plt.savefig("tmp_scripts/dapi_fitc_comparison_corrected.png", dpi=150, bbox_inches='tight')
        print("Saved comparison image to tmp_scripts/dapi_fitc_comparison_corrected.png")
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison images: {str(e)}")
        import traceback
        print(traceback.format_exc())
else:
    print("No suitable pairs found for visualization.")

print("\nFile description analysis complete.")