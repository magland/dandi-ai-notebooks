#!/usr/bin/env python
# This script explicitly extracts and visualizes specific DAPI and FITC images
# using their directly identified asset IDs

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient
import os

print("Extracting specific DAPI and FITC images...")

# Define the specific assets we want to use
# Based on the file paths in the previous script's output
dapi_asset_id = "95141d7a-82aa-4552-940a-1438a430a0d7"  # This is the DAPI (pre) file
dapi_path = "sub-P1-20240627-A2/sub-P1-20240627-A2_obj-1aoyzxh_image.nwb"

fitc_asset_id = "d22476ad-fa18-4aa0-84bf-13fd0113a52c"  # This is the FITC (post) file
fitc_path = "sub-P1-20240627-A2/sub-P1-20240627-A2_obj-fniblx_image.nwb"

# Function to load an NWB file from the asset ID
def load_nwb(asset_id, path):
    print(f"Loading NWB file: {path} (asset_id: {asset_id})")
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001276/assets/{asset_id}/nwb.lindi.json"
    
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    
    # Print basic info
    desc = nwb.session_description
    print(f"File description: {desc[:100]}...")
    
    return nwb

# Load both files
dapi_nwb = load_nwb(dapi_asset_id, dapi_path)
fitc_nwb = load_nwb(fitc_asset_id, fitc_path)

# Function to extract a sample from an image
def extract_image_sample(nwb, crop_size=1000):
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

# Extract image samples
print("\nExtracting image samples...")
dapi_sample = extract_image_sample(dapi_nwb)
fitc_sample = extract_image_sample(fitc_nwb)

# Print basic stats about the samples
print(f"DAPI sample shape: {dapi_sample.shape}")
print(f"DAPI min: {np.min(dapi_sample)}, max: {np.max(dapi_sample)}, mean: {np.mean(dapi_sample):.2f}")

print(f"FITC sample shape: {fitc_sample.shape}")
print(f"FITC min: {np.min(fitc_sample)}, max: {np.max(fitc_sample)}, mean: {np.mean(fitc_sample):.2f}")

# Normalize for visualization
def normalize_sample(sample):
    sample_min = np.min(sample)
    sample_max = np.max(sample)
    if sample_min != sample_max:
        return (sample - sample_min) / (sample_max - sample_min)
    else:
        return np.zeros_like(sample)

dapi_norm = normalize_sample(dapi_sample)
fitc_norm = normalize_sample(fitc_sample)

# Create comparison of the two images
plt.figure(figsize=(16, 8))

# DAPI image (nuclei staining, pre-exposure)
plt.subplot(1, 2, 1)
plt.imshow(dapi_norm, cmap='Blues')
plt.title("DAPI (pre-exposure)\nNuclei staining with Hoechst", fontsize=14)
plt.axis('off')
plt.colorbar(label="Normalized Intensity")

# FITC image (YoPro-1, post-exposure)
plt.subplot(1, 2, 2)
plt.imshow(fitc_norm, cmap='hot')
plt.title("FITC (post-exposure)\nYoPro-1 uptake showing cell permeabilization", fontsize=14)
plt.axis('off')
plt.colorbar(label="Normalized Intensity")

plt.suptitle("Comparison of DAPI and FITC images for subject P1-20240627-A2", fontsize=16)
plt.tight_layout()

# Save the comparison figure
plt.savefig("tmp_scripts/dapi_fitc_comparison_explicit.png", dpi=150, bbox_inches='tight')
print("\nSaved comparison image to tmp_scripts/dapi_fitc_comparison_explicit.png")

# Create a composite image (DAPI in blue channel, FITC in green)
if dapi_norm.shape == fitc_norm.shape:
    # Create RGB image
    composite = np.zeros((dapi_norm.shape[0], dapi_norm.shape[1], 3))
    composite[:,:,0] = 0                  # Red channel empty
    composite[:,:,1] = fitc_norm * 0.8    # Green channel for FITC
    composite[:,:,2] = dapi_norm * 0.8    # Blue channel for DAPI
    
    plt.figure(figsize=(10, 8))
    plt.imshow(composite)
    plt.title("Composite image: DAPI (blue) and FITC (green)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("tmp_scripts/composite_explicit.png", dpi=150, bbox_inches='tight')
    print("Saved composite image to tmp_scripts/composite_explicit.png")

# Generate histograms of pixel intensities
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(dapi_sample.flatten(), bins=100, alpha=0.7, color='blue')
plt.title("DAPI Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.yscale('log')  # Use log scale for better visualization of distribution

plt.subplot(1, 2, 2)
plt.hist(fitc_sample.flatten(), bins=100, alpha=0.7, color='red')
plt.title("FITC Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.yscale('log')  # Use log scale for better visualization of distribution

plt.tight_layout()
plt.savefig("tmp_scripts/intensity_histograms_explicit.png", dpi=150, bbox_inches='tight')
print("Saved intensity histograms to tmp_scripts/intensity_histograms_explicit.png")

# Analyze the plots using dandi-notebook-gen-tools
print("\nAnalyzing the comparison plot...")
os.system("dandi-notebook-gen-tools analyze-plot tmp_scripts/dapi_fitc_comparison_explicit.png")
print("\nAnalyzing the composite plot...")
os.system("dandi-notebook-gen-tools analyze-plot tmp_scripts/composite_explicit.png")

print("\nImage extraction and visualization complete.")