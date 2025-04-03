"""
Explore the structure of the NWB file and print information about its components.
This script will load the NWB file and print detailed information about the available
data structures, datasets, and attributes.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file using lindi
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information about the file
print("\n==== BASIC INFORMATION ====")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"File Create Date: {nwb.file_create_date}")

# Subject information
print("\n==== SUBJECT INFORMATION ====")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# OnePhotonSeries information
print("\n==== ONE PHOTON SERIES ====")
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"Data shape: {one_photon_series.data.shape}")
print(f"Data type: {one_photon_series.data.dtype}")
print(f"Starting time: {one_photon_series.starting_time} sec")
print(f"Rate: {one_photon_series.rate} Hz")

# Describe processing module
print("\n==== PROCESSING MODULE ====")
ophys = nwb.processing["ophys"]
print(f"Description: {ophys.description}")
print(f"Available interfaces: {list(ophys.data_interfaces.keys())}")

# Fluorescence data
print("\n==== FLUORESCENCE DATA ====")
fluorescence = ophys["Fluorescence"]["RoiResponseSeries"]
print(f"Data shape: {fluorescence.data.shape}")
print(f"Data type: {fluorescence.data.dtype}")
print(f"Starting time: {fluorescence.starting_time} sec")
print(f"Rate: {fluorescence.rate} Hz")

# Event amplitude data
print("\n==== EVENT AMPLITUDE DATA ====")
event_amplitude = ophys["EventAmplitude"]
print(f"Data shape: {event_amplitude.data.shape}")
print(f"Data type: {event_amplitude.data.dtype}")
print(f"Starting time: {event_amplitude.starting_time} sec")
print(f"Rate: {event_amplitude.rate} Hz")

# Image segmentation (ROI masks)
print("\n==== IMAGE SEGMENTATION ====")
plane_segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]
print(f"Number of ROIs: {len(plane_segmentation)}")
print(f"Image mask shape: {plane_segmentation['image_mask'].data.shape}")

# Check for other available attributes
print("\n==== AVAILABLE COLUMNS IN PLANESEGMENTATION ====")
for column_name in plane_segmentation.colnames:
    print(f"- {column_name}")