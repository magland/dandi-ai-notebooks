# This script explores the basic structure of the NWB file and extracts key metadata
# to understand the overall dataset organization.

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# Load NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic metadata
print("==== BASIC METADATA ====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject sex: {nwb.subject.sex}")

# Explore NWB file structure
print("\n==== NWB FILE STRUCTURE ====")
print("Available groups in NWB file:")
for group_name in f.keys():
    print(f"- {group_name}")

# Get acquisition groups
print("\nAcquisition groups:")
for name in nwb.acquisition.keys():
    acq = nwb.acquisition[name]
    print(f"- {name} ({type(acq).__name__}): {acq.data.shape}, {acq.data.dtype}")

# Get information about electrodes
print("\n==== ELECTRODES INFORMATION ====")
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes.id[:])}")
print(f"Electrode columns: {electrodes.colnames}")

# Sample a few electrode locations to understand their placement
print("\nSample electrode locations:")
for i in range(min(5, len(electrodes.id[:]))):
    print(f"Electrode {electrodes.id[i]}: x={electrodes['x'][i]}, y={electrodes['y'][i]}, z={electrodes['z'][i]}, location={electrodes['location'][i]}")

# Get information about trials
print("\n==== TRIALS INFORMATION ====")
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials.id[:])}")
print(f"Trial columns: {trials.colnames}")
print("\nFirst 5 trials:")
for i in range(min(5, len(trials.id[:]))):
    print(f"Trial {trials.id[i]}: start={trials['start_time'][i]:.2f}s, stop={trials['stop_time'][i]:.2f}s, duration={(trials['stop_time'][i] - trials['start_time'][i])*1000:.1f}ms")

# Plot histogram of trial durations to check consistency
durations = trials['stop_time'][:] - trials['start_time'][:]
plt.figure(figsize=(10, 6))
plt.hist(durations * 1000, bins=50)  # Convert to milliseconds
plt.title('Distribution of Trial Durations')
plt.xlabel('Duration (ms)')
plt.ylabel('Count')
plt.savefig('tmp_scripts/trial_duration_histogram.png')
plt.close()