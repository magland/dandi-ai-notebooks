"""
This script explores the structure of an NWB file from the Dandiset 001174.
It provides basic information about the file contents, including metadata,
data series, and the shape/size of key datasets.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic info
print("----- Basic Dataset Information -----")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File created: {nwb.file_create_date}")

# Subject info
print("\n----- Subject Information -----")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# Acquisition data
print("\n----- Acquisition Data -----")
for name, data in nwb.acquisition.items():
    print(f"Dataset: {name}")
    if hasattr(data, 'data'):
        print(f"  Shape: {data.data.shape}")
        print(f"  Data type: {data.data.dtype}")
    if hasattr(data, 'rate'):
        print(f"  Rate: {data.rate} Hz")
    print()

# Processing modules
print("\n----- Processing Modules -----")
for module_name, module in nwb.processing.items():
    print(f"Module: {module_name}")
    for data_name, data_interface in module.data_interfaces.items():
        print(f"  Interface: {data_name}")
        if hasattr(data_interface, 'data'):
            print(f"    Shape: {data_interface.data.shape}")
        print(f"    Type: {type(data_interface).__name__}")
        
        # Handle specific interface types with nested data
        if isinstance(data_interface, pynwb.ophys.Fluorescence):
            for roi_series_name, roi_series in data_interface.roi_response_series.items():
                print(f"      ROI Series: {roi_series_name}")
                print(f"        Shape: {roi_series.data.shape}")
                print(f"        Rate: {roi_series.rate} Hz")
                
        elif isinstance(data_interface, pynwb.ophys.ImageSegmentation):
            for plane_seg_name, plane_seg in data_interface.plane_segmentations.items():
                print(f"      Plane Segmentation: {plane_seg_name}")
                for column_name in plane_seg.colnames:
                    if column_name == 'image_mask' and hasattr(plane_seg[column_name], 'data'):
                        print(f"        Image masks shape: {plane_seg[column_name].data.shape}")
    print()

# Get a list of all neuron IDs
print("\n----- Neuron Information -----")
try:
    plane_seg = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
    num_neurons = len(plane_seg.id.data[:])
    print(f"Number of neurons: {num_neurons}")
    
    if num_neurons > 0:
        print(f"First 5 neuron IDs: {plane_seg.id.data[:5]}")
except Exception as e:
    print(f"Error accessing neuron information: {e}")