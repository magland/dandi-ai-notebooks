'''
This script explores the electrode information in an NWB file from the PESD dataset.
It displays information about electrode locations, groups, and other metadata.
'''

import pynwb
import lindi
import numpy as np
import pandas as pd

# Load an NWB file using the lindi library
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/00df5264-001b-4bb0-a987-0ddfb6058961/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Extract electrode information
print("NWB File Session Information:")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Session start time: {nwb.session_start_time}")
print("\n----------------------\n")

# Get electrode information
electrodes = nwb.electrodes
electrode_ids = electrodes["id"].data[:]

# Convert electrode data to a pandas DataFrame for easier viewing
electrode_data = {
    "id": electrode_ids,
    "location": [loc for loc in electrodes["location"].data[:]],
    "group_name": [name for name in electrodes["group_name"].data[:]],
    "label": [label for label in electrodes["label"].data[:]]
}

df_electrodes = pd.DataFrame(electrode_data)
print("Electrode Information:")
print(df_electrodes)

# Get electrode groups information
print("\nElectrode Groups:")
for group_name, group in nwb.electrode_groups.items():
    print(f"Group: {group_name}")
    print(f"  Description: {group.description}")
    print(f"  Location: {group.location}")
    print(f"  Device: {group.device}")

# Print information about LFP data
print("\nLFP Data Information:")
lfp_series = nwb.processing["ecephys"]["LFP"]["LFP"]
print(f"LFP data shape: {lfp_series.data.shape}")
print(f"Sampling rate: {lfp_series.rate} Hz")
print(f"Starting time: {lfp_series.starting_time} seconds")