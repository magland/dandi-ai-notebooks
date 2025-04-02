"""
This script provides a simplified analysis of the neural data, focusing on basic unit properties
and sampling a smaller subset of data to avoid timeout issues.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import time

# Start timing
start_time = time.time()

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the units data
units = nwb.processing["ecephys"]["units"]

# Get the quality information for units
quality = units["quality"].data[:]
print(f"\nUnit quality distribution:")
quality_counts = {}
for q in quality:
    if q in quality_counts:
        quality_counts[q] += 1
    else:
        quality_counts[q] = 1
        
for q, count in quality_counts.items():
    print(f"  {q}: {count} units")

# Extract basic information about all units
n_units = len(units["id"])
fr = units["fr"].data[:]  # firing rates
depths = units["depth"].data[:]  # recording depths
amplitudes = units["Amplitude"].data[:]
n_spikes = units["n_spikes"].data[:]

print(f"\nUnit statistics:")
print(f"  Number of units: {n_units}")
print(f"  Average firing rate: {np.mean(fr):.2f} Hz")
print(f"  Min firing rate: {np.min(fr):.2f} Hz")
print(f"  Max firing rate: {np.max(fr):.2f} Hz")
print(f"  Depth range: {np.min(depths):.2f} - {np.max(depths):.2f}")
print(f"  Average spike count per unit: {np.mean(n_spikes):.2f}")
print(f"  Average spike amplitude: {np.mean(amplitudes):.2f}")

# Plot firing rate distribution
plt.figure(figsize=(10, 6))
plt.hist(fr, bins=20)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.title('Distribution of Unit Firing Rates')
plt.savefig("firing_rate_distribution.png")

# Plot firing rate vs depth
plt.figure(figsize=(10, 6))
plt.scatter(depths, fr, alpha=0.7)
plt.xlabel('Depth')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rate vs Depth')
plt.savefig("firing_rate_vs_depth.png")

# Plot spike amplitude vs firing rate
plt.figure(figsize=(10, 6))
plt.scatter(fr, amplitudes, alpha=0.7)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Spike Amplitude')
plt.title('Spike Amplitude vs Firing Rate')
plt.savefig("amplitude_vs_firing_rate.png")

# Plot spike count vs firing rate
plt.figure(figsize=(10, 6))
plt.scatter(fr, n_spikes, alpha=0.7)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Spikes')
plt.title('Number of Spikes vs Firing Rate')
plt.savefig("spike_count_vs_firing_rate.png")

# Get a subset of units for spike time analysis (top 5 by firing rate)
top_indices = np.argsort(fr)[-5:]
unit_ids = units["id"][top_indices]

# Get trials data for a subset of the trials
trials = nwb.intervals["trials"]
max_trials = min(100, len(trials["id"]))
trial_indices = np.random.choice(range(len(trials["id"])), max_trials, replace=False)

print(f"\nAnalyzing spike timing for {len(top_indices)} units across {max_trials} sample trials")

# Sample spike histograms for the top units
for i, unit_idx in enumerate(top_indices):
    unit_id = unit_ids[i]
    print(f"  Processing unit {unit_id}...")
    
    # Get spike times for this unit
    spike_times = units["spike_times"][unit_idx]
    
    # Create a raster plot for sample trials
    plt.figure(figsize=(12, 8))
    
    # Plot spike times relative to trial start (limit to max_trials)
    for j, trial_idx in enumerate(trial_indices[:20]):  # Limit to 20 trials for clarity
        start_time = trials["start_time"][trial_idx]
        stop_time = trials["stop_time"][trial_idx]
        
        # Find spikes in this trial window
        trial_spikes = spike_times[(spike_times >= start_time) & (spike_times <= stop_time)]
        
        # Plot relative to trial start
        if len(trial_spikes) > 0:
            relative_times = trial_spikes - start_time
            plt.scatter(relative_times, np.ones_like(trial_spikes) * j, marker='|', s=30, color='black')
    
    plt.xlabel('Time from Trial Start (s)')
    plt.ylabel('Trial Number')
    plt.title(f'Unit {unit_id} Spike Raster Plot')
    plt.tight_layout()
    plt.savefig(f"unit_{unit_id}_raster.png")

# Print total execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nScript execution completed in {elapsed_time:.2f} seconds")