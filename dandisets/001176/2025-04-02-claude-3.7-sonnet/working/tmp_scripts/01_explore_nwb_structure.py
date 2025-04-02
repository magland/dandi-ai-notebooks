# Script to explore the basic structure of the NWB file
# This script loads the NWB file and examines its structure to understand the available data

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/babeee4c-bb8f-4d0b-b898-3edf99244f25/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic metadata
print("\n=== Basic Metadata ===")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Keywords: {nwb.keywords}")
print(f"Lab: {nwb.lab}")

# Print subject information
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Species: {nwb.subject.species}")
print(f"Genotype: {nwb.subject.genotype}")
print(f"Description: {nwb.subject.description}")

# Examine the acquisition data
print("\n=== Acquisition Data ===")
for name, obj in nwb.acquisition.items():
    print(f"{name}: {type(obj).__name__}")
    if hasattr(obj, 'data'):
        print(f"  - Data shape: {obj.data.shape}, dtype: {obj.data.dtype}")
    if hasattr(obj, 'timestamps'):
        print(f"  - Timestamps shape: {obj.timestamps.shape}, dtype: {obj.timestamps.dtype}")
        print(f"  - Time range: {obj.timestamps[0]} to {obj.timestamps[-1]}, duration: {obj.timestamps[-1] - obj.timestamps[0]:.2f} seconds")

# Examine the processing modules
print("\n=== Processing Modules ===")
for module_name, module in nwb.processing.items():
    print(f"Module: {module_name}")
    for data_name, data_obj in module.data_interfaces.items():
        print(f"  - {data_name}: {type(data_obj).__name__}")

# Look at the ophys processing module in more detail
print("\n=== Ophys Processing Module Details ===")
ophys = nwb.processing["ophys"]

# Fluorescence data
print("\nFluorescence data:")
fluor = ophys["Fluorescence"]
for series_name, series in fluor.roi_response_series.items():
    print(f"  - {series_name}:")
    print(f"    - Data shape: {series.data.shape}, dtype: {series.data.dtype}")
    print(f"    - Timestamps shape: {series.timestamps.shape}, dtype: {series.timestamps.dtype}")
    print(f"    - Time range: {series.timestamps[0]} to {series.timestamps[-1]}, duration: {series.timestamps[-1] - series.timestamps[0]:.2f} seconds")

# Image Segmentation
print("\nImage Segmentation:")
seg = ophys["ImageSegmentation"]
for plane_name, plane in seg.plane_segmentations.items():
    print(f"  - {plane_name}:")
    for col_name in plane.colnames:
        data = plane[col_name].data
        if hasattr(data, 'shape'):
            print(f"    - {col_name}: shape {data.shape}, dtype {data.dtype}")
        else:
            print(f"    - {col_name}: {data}")

# Summary Images
print("\nSummary Images:")
if "SummaryImages_chan1" in ophys.data_interfaces:
    summary_imgs = ophys["SummaryImages_chan1"]
    for img_name, img in summary_imgs.images.items():
        print(f"  - {img_name}: shape {img.data.shape}, dtype {img.data.dtype}")