"""
This script explores the basic metadata of an NWB file from the Parkinson's Electrophysiological Signal Dataset.
We'll examine the session information, subject details, and electrode configuration.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load a beta band NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic file metadata
print("=== NWB File Metadata ===")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"Keywords: {nwb.keywords}")

# Print subject information
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Species: {nwb.subject.species}")
print(f"Description: {nwb.subject.description}")

# Examine the electrode table
print("\n=== Electrode Table Information ===")
electrodes = nwb.ec_electrodes
print(f"Electrode columns: {electrodes.colnames}")
electrode_ids = electrodes["id"].data[:]
print(f"Number of electrodes: {len(electrode_ids)}")

# Print details for each electrode
print("\nElectrode Details:")
locations = electrodes["location"].data[:]
labels = electrodes["label"].data[:]
group_names = electrodes["group_name"].data[:]

for i in range(len(electrode_ids)):
    print(f"ID: {electrode_ids[i]}, Label: {labels[i]}, Location: {locations[i]}, Group: {group_names[i]}")