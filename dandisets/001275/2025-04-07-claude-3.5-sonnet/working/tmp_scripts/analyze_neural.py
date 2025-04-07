"""
Analyze neural data from the mental navigation task:
- Plot example spike raster around trial events
- Show electrode locations
- Display firing rate distributions
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trials data and units
trials = nwb.intervals["trials"]
units = nwb.processing["ecephys"]["units"]
electrodes = nwb.electrodes

# Plot 1: Electrode locations
plt.figure(figsize=(8, 8))
x_pos = electrodes["rel_x"].data[:]
y_pos = electrodes["rel_y"].data[:]
plt.scatter(x_pos, y_pos)
plt.title('Electrode Locations')
plt.xlabel('Relative X Position')
plt.ylabel('Relative Y Position')
plt.savefig('tmp_scripts/electrode_locations.png')
plt.close()

# Plot 2: Firing rate distribution
firing_rates = units["fr"].data[:]
plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=20)
plt.title('Distribution of Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.savefig('tmp_scripts/firing_rates.png')
plt.close()

# Plot 3: Example spike raster
# Get spikes and trial events for a subset of trials
n_trials_to_plot = 20
trial_start_times = trials["start_time"][:n_trials_to_plot]
go_cue_times = trials["gocuettl"][:n_trials_to_plot]

# Select a few units with good firing rates
unit_indices = np.argsort(firing_rates)[-5:]  # Get 5 units with highest firing rates
plt.figure(figsize=(12, 8))

for i, unit_idx in enumerate(unit_indices):
    spike_times = units["spike_times"][unit_idx]
    
    # Plot spikes relative to trial start
    trial_spikes = []
    trial_numbers = []
    
    for trial_num, (start_time, go_time) in enumerate(zip(trial_start_times, go_cue_times)):
        # Get spikes in 2-second window around go cue
        mask = (spike_times >= go_time - 1) & (spike_times <= go_time + 1)
        trial_spikes.extend(spike_times[mask] - go_time)
        trial_numbers.extend([trial_num] * np.sum(mask))
    
    plt.scatter(trial_spikes, np.array(trial_numbers) + i, s=1, alpha=0.5)

plt.axvline(x=0, color='r', linestyle='--', label='Go Cue')
plt.title('Spike Raster Around Go Cue')
plt.xlabel('Time from Go Cue (s)')
plt.ylabel('Trial Number')
plt.legend()
plt.savefig('tmp_scripts/spike_raster.png')
plt.close()

# Plot 4: Unit quality distribution
quality_counts = {}
for q in units["quality"].data[:]:
    if q in quality_counts:
        quality_counts[q] += 1
    else:
        quality_counts[q] = 1

plt.figure(figsize=(8, 6))
plt.bar(quality_counts.keys(), quality_counts.values())
plt.title('Distribution of Unit Quality')
plt.ylabel('Number of Units')
plt.savefig('tmp_scripts/unit_quality.png')
plt.close()