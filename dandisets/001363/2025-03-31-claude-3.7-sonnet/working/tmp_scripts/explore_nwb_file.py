"""
This script explores the basic properties of an NWB file from Dandiset 001363.
It loads the NWB file and prints information about the dataset structure,
including details about electrical series data and trials.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic session information
print("==== Session Information ====")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject age: {nwb.subject.age}")
print()

# Print information about electrical series
print("==== Electrical Series Information ====")
electrical_series = nwb.acquisition["ElectricalSeries"]
print(f"Data shape: {electrical_series.data.shape}")
print(f"Data dtype: {electrical_series.data.dtype}")
print(f"Sampling rate: {electrical_series.rate} Hz")
print(f"Starting time: {electrical_series.starting_time}")
print()

# Print information about electrodes
print("==== Electrode Information ====")
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes.id[:])}")
print(f"Electrode columns: {electrodes.colnames}")
print()

# Print information about trials
print("==== Trials Information ====")
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['start_time'][:])}")
print(f"Trial duration (first 5 trials): ")
for i in range(min(5, len(trials['start_time'][:]))):
    duration = trials['stop_time'][i] - trials['start_time'][i]
    print(f"  Trial {i+1}: {duration:.4f} seconds")

# Calculate time between trials
trial_starts = trials['start_time'][:]
inter_trial_intervals = np.diff(trial_starts)
print(f"\nAverage time between trials: {np.mean(inter_trial_intervals):.4f} seconds")
print(f"Standard deviation of inter-trial intervals: {np.std(inter_trial_intervals):.4f} seconds")