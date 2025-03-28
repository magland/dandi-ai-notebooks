"""
Script to explore the basic structure of the NWB file from Dandiset 001335.
This script will:
1. Load the NWB file
2. Extract basic metadata
3. Examine the odor presentation intervals
4. Look at the LFP data structure and extract a sample
5. Examine a few units (neurons)
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
print("NWB file loaded successfully!")

# Extract basic metadata
print("\n=== Basic Metadata ===")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"Keywords: {nwb.keywords}")
print(f"Experimenter: {nwb.experimenter}")

# Subject information
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.subject.description}")

# Examine block intervals
print("\n=== Experimental Blocks ===")
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    start = block["start_time"].data[:]
    stop = block["stop_time"].data[:]
    duration = stop - start
    print(f"{block_name}: Start={start[0]:.2f}s, Stop={stop[0]:.2f}s, Duration={duration[0]:.2f}s")

# Examine odor presentation intervals
print("\n=== Odor Presentations ===")
odor_counts = {}
for odor_name in ["Odor A ON", "Odor B ON", "Odor C ON", "Odor D ON", "Odor E ON", "Odor F ON"]:
    odor_interval = nwb.intervals[odor_name]
    starts = odor_interval["start_time"].data[:]
    stops = odor_interval["stop_time"].data[:]
    durations = stops - starts
    avg_duration = np.mean(durations)
    odor_counts[odor_name] = len(starts)
    print(f"{odor_name}: {len(starts)} presentations, Avg Duration={avg_duration:.2f}s")

# Create a bar plot for odor presentation counts
plt.figure(figsize=(10, 6))
plt.bar(odor_counts.keys(), odor_counts.values())
plt.xlabel('Odor Type')
plt.ylabel('Number of Presentations')
plt.title('Number of Presentations per Odor Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('tmp_scripts/odor_presentation_counts.png')

# Examine LFP data
print("\n=== LFP Data ===")
lfp = nwb.processing["ecephys"]["LFP"]
print(f"LFP Shape: {lfp.data.shape}")
print(f"Sampling Rate: {lfp.rate} Hz")
print(f"Duration: {lfp.data.shape[0]/lfp.rate:.2f} seconds")
print(f"Number of Electrodes: {lfp.data.shape[1]}")

# Get electrode information
electrodes = nwb.electrodes
print("\n=== Electrode Information ===")
print(f"Number of Electrodes: {len(electrodes['id'].data[:])}")
print(f"Columns: {electrodes.colnames}")

# Create a dataframe with electrode information for a few electrodes
electrode_ids = electrodes["id"].data[:]
locations = electrodes["location"].data[:]
depths = electrodes["depth"].data[:]
hemispheres = electrodes["hemisphere"].data[:]
labels = electrodes["label"].data[:]

# Create a sample dataframe for the first 10 electrodes
electrode_df = pd.DataFrame({
    'id': electrode_ids[:10],
    'location': locations[:10],
    'depth': depths[:10],
    'hemisphere': hemispheres[:10],
    'label': labels[:10]
})
print("\nSample Electrode Data (first 10):")
print(electrode_df)

# Extract a small sample of LFP data for visualization
print("\nExtracting LFP sample for visualization...")
# Get 1 second of data from the beginning for 5 electrodes
sample_duration = 1.0  # seconds
sample_rate = lfp.rate
n_samples = int(sample_duration * sample_rate)
electrode_indices = [0, 15, 30, 45, 60]  # Select a few electrodes across the array

if len(electrode_indices) > lfp.data.shape[1]:
    electrode_indices = list(range(min(5, lfp.data.shape[1])))

lfp_sample = np.zeros((n_samples, len(electrode_indices)))
for i, e_idx in enumerate(electrode_indices):
    if e_idx < lfp.data.shape[1]:
        lfp_sample[:, i] = lfp.data[:n_samples, e_idx]

# Plot LFP sample
plt.figure(figsize=(12, 8))
time_vector = np.arange(n_samples) / sample_rate
for i, e_idx in enumerate(electrode_indices):
    if e_idx < lfp.data.shape[1]:
        plt.plot(time_vector, lfp_sample[:, i] + i*200, label=f'Electrode {e_idx}')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV) + Offset')
plt.title('Sample LFP Data from Different Electrodes')
plt.legend()
plt.grid(True)
plt.savefig('tmp_scripts/lfp_sample.png')

# Examine units (neurons)
print("\n=== Units (Neurons) ===")
units = nwb.units
print(f"Number of Units: {len(units['id'].data[:])}")
print(f"Columns: {units.colnames}")

# Get information for a few units
unit_ids = units["id"].data[:]
depths = units["depth"].data[:]
hemispheres = units["hemisphere"].data[:]
global_ids = units["global_id"].data[:]

# Count spikes for the first 10 units
print("\nSpike counts for first 10 units:")
for i in range(10):
    if i < len(unit_ids):
        spike_times = units["spike_times"][i]
        print(f"Unit {global_ids[i]}: {len(spike_times)} spikes")

# Plot spike counts for first 50 units
unit_spike_counts = []
for i in range(50):  # First 50 units
    if i < len(unit_ids):
        spike_times = units["spike_times"][i]
        unit_spike_counts.append(len(spike_times))

plt.figure(figsize=(12, 6))
plt.bar(range(len(unit_spike_counts)), unit_spike_counts)
plt.xlabel('Unit Index')
plt.ylabel('Number of Spikes')
plt.title('Spike Counts for First 50 Units')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('tmp_scripts/unit_spike_counts.png')

print("\nExploration complete. Check the generated plots.")