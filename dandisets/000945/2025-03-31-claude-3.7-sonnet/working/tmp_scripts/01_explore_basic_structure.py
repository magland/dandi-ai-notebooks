"""
Exploratory script to understand the basic structure of an NWB file from the dataset.
This script will:
1. Load an NWB file
2. Print basic metadata
3. Explore the structure of trials data
4. Check unit data (neural spikes)
5. Generate a simple raster plot of spike times
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
import os

# Load the NWB file using lindi
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("=== Basic Metadata ===")
print(f"Session Description: {nwb.session_description}")
print(f"NWB Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Subject: {nwb.subject.subject_id}, Species: {nwb.subject.species}, Sex: {nwb.subject.sex}, Age: {nwb.subject.age}")

print("\n=== Trials Information ===")
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['id'][:])}")
print(f"Trial start times (first 5): {trials['start_time'][0:5]}")
print(f"Trial durations (first 5): {trials['stop_time'][0:5] - trials['start_time'][0:5]}")

print("\n=== Units Information ===")
units = nwb.units
unit_ids = units["id"].data
print(f"Number of units: {len(unit_ids)}")
print(f"Available columns: {units.colnames}")

# Print cell type distribution
celltypes = units["celltype_label"].data[:]
unique_celltypes, counts = np.unique(celltypes, return_counts=True)
print(f"Cell type distribution: {dict(zip(unique_celltypes, counts))}")

# Get spike times for a few example units (first 10)
print("\n=== Spike Information ===")
num_units_to_show = 10
for i in range(min(num_units_to_show, len(unit_ids))):
    spike_times = units["spike_times"][i]
    cell_type = "RSU" if units["celltype_label"].data[i] == 1 else "FSU"
    print(f"Unit {i} (Type: {cell_type}): {len(spike_times)} spikes")

# Create a raster plot of spike times around stimulus events
# Let's look at spikes for 10 units, around first 5 trials
fig, ax = plt.subplots(figsize=(10, 8))

# Plot duration before and after stimulus (in seconds)
pre_time = 0.5  
post_time = 1.5  

# Get the first 5 trial start times
trial_starts = trials["start_time"][0:5]

# For each of 10 units, plot spikes around each trial
for unit_idx in range(min(10, len(unit_ids))):
    spike_times = units["spike_times"][unit_idx]
    cell_type = units["celltype_label"].data[unit_idx]
    
    # Use different colors for different cell types
    color = 'blue' if cell_type == 1 else 'red'
    label = f"Unit {unit_idx} ({'RSU' if cell_type == 1 else 'FSU'})"
    
    for trial_idx, trial_start in enumerate(trial_starts):
        # Find spikes within window around this trial
        mask = (spike_times >= trial_start - pre_time) & (spike_times <= trial_start + post_time)
        spikes_in_window = spike_times[mask]
        
        # Convert to time relative to stimulus onset
        relative_times = spikes_in_window - trial_start
        
        # Plot spikes as dots in a raster plot
        y_pos = unit_idx
        ax.scatter(relative_times, [y_pos] * len(relative_times), 
                   s=5, color=color, alpha=0.7)
    
    # Add label for this unit (only once)
    ax.text(-pre_time - 0.1, unit_idx, label, ha='right', va='center', fontsize=8)

# Add a vertical line to indicate stimulus onset
ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Stimulus Onset')

# Add labels
ax.set_xlabel('Time relative to stimulus (s)')
ax.set_yticks([])
ax.set_title('Spike Raster Plot Around Stimulus Events')

# Add legend for cell types
ax.plot([], [], 'o', color='blue', label='RSU')
ax.plot([], [], 'o', color='red', label='FSU')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('raster_plot.png')
print("\nSaved raster plot to raster_plot.png")