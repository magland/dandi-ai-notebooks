"""
Script to explore the basic structure of the NWB file and print out key information.
This will help us understand what data is available in the file and how it's organized.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
print(f"Loading NWB file from {url}")
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information about the NWB file
print("\nBasic NWB Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"File Create Date: {nwb.file_create_date}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Subject Sex: {nwb.subject.sex}")
print(f"Subject Age: {nwb.subject.age}")

# Print information about the one photon series
print("\nOne Photon Series Information:")
one_photon_series = nwb.acquisition["OnePhotonSeries"]
print(f"Name: {one_photon_series.name}")
print(f"Data Shape: {one_photon_series.data.shape}")
print(f"Data Type: {one_photon_series.data.dtype}")
print(f"Sample Rate: {one_photon_series.rate} Hz")
print(f"Starting Time: {one_photon_series.starting_time} sec")

# Print information about the processing module
print("\nProcessing Module Information:")
ophys = nwb.processing["ophys"]
print(f"Name: {ophys.name}")
print(f"Description: {ophys.description}")

# Get information about the plane segmentation
print("\nPlane Segmentation Information:")
plane_segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]
print(f"Number of ROIs: {plane_segmentation['image_mask'].data.shape[0]}")
print(f"ROI Mask Shape: {plane_segmentation['image_mask'].data.shape}")

# Get information about the fluorescence data
print("\nFluorescence Data Information:")
fluorescence = ophys["Fluorescence"]["RoiResponseSeries"]
print(f"Data Shape: {fluorescence.data.shape}")
print(f"Data Type: {fluorescence.data.dtype}")
print(f"Sample Rate: {fluorescence.rate} Hz")

# Get information about event amplitude data
print("\nEvent Amplitude Information:")
event_amplitude = ophys["EventAmplitude"]
print(f"Data Shape: {event_amplitude.data.shape}")
print(f"Data Type: {event_amplitude.data.dtype}")
print(f"Sample Rate: {event_amplitude.rate} Hz")