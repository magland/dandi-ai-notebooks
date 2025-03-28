"""
This script explores the basic structure of the Dandiset 001335, including:
- Dataset metadata
- Block intervals
- Odor presentation periods
- Basic information about electrodes and units
"""

import numpy as np
import pynwb
import lindi
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic metadata
print("=== Basic Metadata ===")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment description: {nwb.experiment_description}")
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

# Explore block intervals
print("\n=== Block Intervals ===")
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    start_time = block["start_time"].data[:]
    stop_time = block["stop_time"].data[:]
    duration = stop_time - start_time
    print(f"{block_name}: Start={start_time[0]:.2f}s, Stop={stop_time[0]:.2f}s, Duration={duration[0]:.2f}s")

# Explore odor presentation intervals
print("\n=== Odor Presentation ===")
for odor_name in ["Odor A ON", "Odor B ON", "Odor C ON", "Odor D ON", "Odor E ON", "Odor F ON"]:
    odor = nwb.intervals[odor_name]
    start_times = odor["start_time"].data[:]
    stop_times = odor["stop_time"].data[:]
    durations = stop_times - start_times
    avg_duration = np.mean(durations)
    print(f"{odor_name}: {len(start_times)} presentations, Avg Duration={avg_duration:.2f}s")
    
# Basic information about electrodes
print("\n=== Electrode Information ===")
electrodes = nwb.electrodes
n_electrodes = len(electrodes["id"].data[:])
locations = electrodes["location"].data[:]
unique_locations = np.unique(locations)
print(f"Number of electrodes: {n_electrodes}")
print(f"Unique electrode locations: {unique_locations}")

# Basic information about units
print("\n=== Unit Information ===")
units = nwb.units
n_units = len(units["id"].data[:])
hemispheres = units["hemisphere"].data[:]
unique_hemispheres = np.unique(hemispheres)
print(f"Number of units: {n_units}")
print(f"Unique hemispheres: {unique_hemispheres}")

# Create a plot showing the odor presentation timeline for the first 5 minutes (300 seconds)
fig, ax = plt.subplots(figsize=(10, 6))
time_limit = 300  # Show first 5 minutes

odors = ["Odor A ON", "Odor B ON", "Odor C ON", "Odor D ON", "Odor E ON", "Odor F ON"]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

for i, odor_name in enumerate(odors):
    odor = nwb.intervals[odor_name]
    start_times = odor["start_time"].data[:]
    stop_times = odor["stop_time"].data[:]
    
    # Filter to only show events in the first 5 minutes
    mask = start_times < time_limit
    filtered_starts = start_times[mask]
    filtered_stops = stop_times[mask]
    
    for start, stop in zip(filtered_starts, filtered_stops):
        ax.axvspan(start, stop, alpha=0.3, color=colors[i], label=f"{odor_name}" if start == filtered_starts[0] else "")

# Add block information
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    start_time = block["start_time"].data[0]
    stop_time = block["stop_time"].data[0]
    
    # Only show if block starts within the time limit
    if start_time < time_limit:
        # Use a stippled line to mark the start of each block
        ax.axvline(start_time, color='black', linestyle='--', label=f"{block_name} Start")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_xlim(0, time_limit)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Odor Presentations')
ax.set_title('Odor Presentation Timeline (First 5 Minutes)')
plt.tight_layout()
plt.savefig('tmp_scripts/odor_timeline.png')

# Create a histogram of unit depths
fig, ax = plt.subplots(figsize=(8, 6))
depths = units["depth"].data[:]
ax.hist(depths, bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Depth (μm)')
ax.set_ylabel('Number of Units')
ax.set_title('Distribution of Unit Depths')
plt.tight_layout()
plt.savefig('tmp_scripts/unit_depths.png')