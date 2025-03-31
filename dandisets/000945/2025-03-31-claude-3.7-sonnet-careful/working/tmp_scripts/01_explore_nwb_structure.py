"""
This script explores the basic structure of an NWB file from Dandiset 000945.
It loads the file and prints out information about the file structure, trials,
electrodes, and units (spike times), while being careful not to load too much data.
"""
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os

# Make sure the output directory exists
if not os.path.exists('tmp_scripts'):
    os.makedirs('tmp_scripts')

# Load NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
print("File loaded successfully")

print("="*50)
print("NWB FILE BASIC INFO")
print("="*50)
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Age: {nwb.subject.age}")
print(f"Subject Sex: {nwb.subject.sex}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Subject Description: {nwb.subject.description}")

print("\n" + "="*50)
print("TRIALS INFORMATION")
print("="*50)
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['id'])}")
# Only load first 5 trials
start_times = trials['start_time'][:5]
stop_times = trials['stop_time'][:5]
print(f"Trial Start Times (first 5): {start_times}")
print(f"Trial Stop Times (first 5): {stop_times}")
print(f"Trial Duration (first 5): {stop_times - start_times}")

# Calculate inter-trial intervals for just the first 20 trials
iti = trials['start_time'][1:20] - trials['start_time'][:19]
print(f"Inter-trial intervals (first 5): {iti[:5]}")
print(f"Mean ITI (first 19 trials): {np.mean(iti)}")
print(f"Min ITI (first 19 trials): {np.min(iti)}")
print(f"Max ITI (first 19 trials): {np.max(iti)}")

print("\n" + "="*50)
print("ELECTRODES INFORMATION")
print("="*50)
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes['id'])}")
print(f"Electrode columns: {electrodes.colnames}")
# Only get unique values from the first few electrodes to avoid loading all data
locations = np.unique(electrodes['location'].data[:5])
groups = np.unique(electrodes['group_name'].data[:5])
print(f"Sample electrode locations: {locations}")
print(f"Sample electrode groups: {groups}")

print("\n" + "="*50)
print("UNITS INFORMATION")
print("="*50)
units = nwb.units
print(f"Number of units: {len(units['id'].data)}")
print(f"Unit columns: {units.colnames}")

# Count units by cell type
cell_types = units['celltype_label'].data[:]
unique_types, type_counts = np.unique(cell_types, return_counts=True)
print(f"Cell types: {unique_types}")
print(f"Count per cell type: {type_counts}")

# Plot inter-trial intervals for the first 20 trials
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(iti)+1), iti, 'o-', alpha=0.5)
plt.xlabel('Trial Number')
plt.ylabel('Inter-trial Interval (s)')
plt.title('Inter-trial Intervals (First 20 Trials)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tmp_scripts/inter_trial_intervals.png')

# Plot cell type distribution
plt.figure(figsize=(8, 6))
plt.bar(unique_types, type_counts, color=['skyblue', 'salmon'])
plt.xlabel('Cell Type Label')
plt.ylabel('Count')
plt.title('Distribution of Cell Types')
plt.xticks(unique_types)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('tmp_scripts/cell_type_distribution.png')

print("Exploration complete - plots saved to tmp_scripts directory")