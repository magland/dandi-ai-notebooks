#!/usr/bin/env python
# This script explores the basic structure of Dandiset 001276 and examines 
# the contents of an NWB file to understand its structure and metadata.

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
from dandi.dandiapi import DandiAPIClient

# Set up the text file to save output
output_file = open('tmp_scripts/dandiset_exploration.txt', 'w')

# Function to write both to console and file
def write_out(text):
    print(text)
    output_file.write(text + '\n')

# Get the Dandiset information
write_out("Exploring Dandiset 001276")
write_out("-" * 50)

client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())

write_out(f"Dandiset name: {dandiset.version.name}")
write_out(f"Dandiset metadata: {dandiset.version}")
# Get the description since accessing it directly isn't working
try:
    write_out(f"Dandiset description: {dandiset.version.metadata['description']}")
except (AttributeError, KeyError):
    write_out("Dandiset description: [Not available through this interface]")
write_out(f"Number of assets: {len(assets)}")
write_out("-" * 50)

# Analyze asset paths to understand the structure
subjects = set()
obj_patterns = set()
file_types = set()

write_out("Asset path patterns:")
for asset in assets:
    path = asset.path
    parts = path.split('/')
    subject = parts[0]  # This should be the subject identifier
    subjects.add(subject)
    
    # Check for patterns in the filenames
    filename = parts[-1]  # This is the filename
    if 'obj-' in filename:
        obj_id = filename.split('obj-')[1].split('_')[0]
        obj_patterns.add(obj_id)
    
    # Check for file types
    if '.' in filename:
        ext = filename.split('.')[-1]
        file_types.add(ext)

write_out(f"Number of unique subjects: {len(subjects)}")
write_out(f"Example subjects: {list(subjects)[:5]}")
write_out(f"File types: {file_types}")
write_out(f"Number of unique object patterns: {len(obj_patterns)}")
write_out(f"Example object patterns: {list(obj_patterns)[:5]}")
write_out("-" * 50)

# Load a specific NWB file for exploration
# Choose the first asset for detailed examination
sample_asset = assets[0]
asset_id = sample_asset.identifier
asset_path = sample_asset.path
write_out(f"Examining asset: {asset_path}")
write_out(f"Asset ID: {asset_id}")

try:
    # Construct the URL for lindi
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
    write_out(f"Loading file from: {lindi_url}")

    # Open the file
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    
    # Extract and report basic metadata
    write_out("NWB File Metadata:")
    write_out(f"Session description: {nwb.session_description[:200]}...")  # First 200 chars
    write_out(f"Identifier: {nwb.identifier}")
    write_out(f"Session start time: {nwb.session_start_time}")
    write_out(f"File creation date: {nwb.file_create_date}")
    write_out(f"Experimenter(s): {nwb.experimenter}")
    write_out(f"Institution: {nwb.institution}")
    write_out(f"Lab: {nwb.lab}")
    
    # Subject information
    write_out("\nSubject Information:")
    write_out(f"Subject ID: {nwb.subject.subject_id}")
    write_out(f"Species: {nwb.subject.species}")
    write_out(f"Sex: {nwb.subject.sex}")
    
    # Check the acquisition structure
    write_out("\nAcquisition Data:")
    for name, data in nwb.acquisition.items():
        write_out(f"  - {name}: {type(data).__name__}")
        
        # If it's an ImageSeries, get more info
        if isinstance(data, pynwb.image.ImageSeries):
            write_out(f"    Dimensions: {data.data.shape}")
            write_out(f"    Data type: {data.data.dtype}")
            
            # Get a small sample of the data to understand its range (without loading everything)
            sample_size = min(100, data.data.shape[0])
            sample = data.data[:sample_size]
            write_out(f"    Data sample min: {np.min(sample)}")
            write_out(f"    Data sample max: {np.max(sample)}")

    # Check for additional NWB structures
    write_out("\nAdditional NWB Structures:")
    if len(nwb.processing) > 0:
        write_out("Processing Modules:")
        for module_name, module in nwb.processing.items():
            write_out(f"  - {module_name}")
            for data_name, data in module.data_interfaces.items():
                write_out(f"    - {data_name}: {type(data).__name__}")
    else:
        write_out("No processing modules found.")
        
    if len(nwb.intervals) > 0:
        write_out("\nIntervals:")
        for interval_name, interval in nwb.intervals.items():
            write_out(f"  - {interval_name}: {type(interval).__name__}")
    else:
        write_out("No intervals found.")
        
    if len(nwb.units) > 0:
        write_out("\nUnits:")
        write_out(f"  - {len(nwb.units)} units found")
    else:
        write_out("No units found.")

    # Get a small sample of the image data and create a visualization
    write_out("\nCreating a visualization of the image data...")
    if "SingleTimePointImaging" in nwb.acquisition:
        img_data = nwb.acquisition["SingleTimePointImaging"]
        # Get image dimensions
        shape = img_data.data.shape
        write_out(f"Image shape: {shape}")
        
        # Extract a small portion to visualize (to avoid memory issues)
        # For example, take a small central region if it's a large image
        scale_factor = 10  # Reduce size by this factor to avoid loading too much data
        
        if len(shape) == 3:  # Multi-frame image series
            n_frames = shape[0]
            height, width = shape[1], shape[2]
            sample_frame = 0  # Take the first frame
            sample_height = height // scale_factor
            sample_width = width // scale_factor
            start_h = (height - sample_height) // 2
            start_w = (width - sample_width) // 2
            
            # Extract small central patch from the first frame
            sample_img = img_data.data[sample_frame, 
                                       start_h:start_h+sample_height, 
                                       start_w:start_w+sample_width]
            
            write_out(f"Extracted sample of size: {sample_img.shape} from frame {sample_frame}")
            
        elif len(shape) == 2:  # Single image
            height, width = shape
            sample_height = height // scale_factor
            sample_width = width // scale_factor
            start_h = (height - sample_height) // 2
            start_w = (width - sample_width) // 2
            
            # Extract small central patch
            sample_img = img_data.data[start_h:start_h+sample_height, 
                                      start_w:start_w+sample_width]
            
            write_out(f"Extracted sample of size: {sample_img.shape}")
            
        # Save the visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(sample_img, cmap='gray')
        plt.title(f"Sample from {asset_path}")
        plt.colorbar(label='Intensity')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('tmp_scripts/sample_image.png', dpi=150, bbox_inches='tight')
        write_out("Saved sample visualization to tmp_scripts/sample_image.png")

except Exception as e:
    write_out(f"Error loading or processing the NWB file: {str(e)}")
    import traceback
    write_out(traceback.format_exc())

output_file.close()
write_out("Exploration complete. Results saved to tmp_scripts/dandiset_exploration.txt")