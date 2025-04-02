"""
This script explores the neural data in the NWB file, examining basic unit properties
and neural activity patterns in relation to behavior during the mental navigation task.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

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

print(f"\nUnit statistics:")
print(f"  Number of units: {n_units}")
print(f"  Average firing rate: {np.mean(fr):.2f} Hz")
print(f"  Min firing rate: {np.min(fr):.2f} Hz")
print(f"  Max firing rate: {np.max(fr):.2f} Hz")
print(f"  Depth range: {np.min(depths):.2f} - {np.max(depths):.2f}")

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

# Calculate the average firing rate for each unit during successful vs. failed trials
print("\nAnalyzing neural activity for successful vs. failed trials...")

# Get trial data
trials = nwb.intervals["trials"]
success = trials["succ"][:] == 1
failure = trials["succ"][:] == 0

# Create time bins for analyzing neural activity (pre, during, post-trial)
n_units = len(units["id"])
success_rates = np.zeros(n_units)
failure_rates = np.zeros(n_units)

# Get spiking data for each unit
for i in range(n_units):
    spike_times = units["spike_times"][i]
    
    # Count spikes during successful trials
    success_spike_count = 0
    success_duration = 0
    for j in range(len(success)):
        if success[j]:
            start_time = trials["start_time"][j]
            stop_time = trials["stop_time"][j]
            duration = stop_time - start_time
            success_duration += duration
            
            # Count spikes that fall within this trial's time window
            trial_spikes = np.sum((spike_times >= start_time) & (spike_times <= stop_time))
            success_spike_count += trial_spikes
    
    if success_duration > 0:
        success_rates[i] = success_spike_count / success_duration
    
    # Count spikes during failed trials
    failure_spike_count = 0
    failure_duration = 0
    for j in range(len(failure)):
        if failure[j]:
            start_time = trials["start_time"][j]
            stop_time = trials["stop_time"][j]
            duration = stop_time - start_time
            failure_duration += duration
            
            # Count spikes that fall within this trial's time window
            trial_spikes = np.sum((spike_times >= start_time) & (spike_times <= stop_time))
            failure_spike_count += trial_spikes
    
    if failure_duration > 0:
        failure_rates[i] = failure_spike_count / failure_duration

# Plot the comparison of firing rates during successful vs. failed trials
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(success_rates, failure_rates)
max_rate = max(np.max(success_rates), np.max(failure_rates)) * 1.1
plt.plot([0, max_rate], [0, max_rate], 'k--')
plt.xlabel('Firing Rate During Successful Trials (Hz)')
plt.ylabel('Firing Rate During Failed Trials (Hz)')
plt.title('Comparison of Firing Rates: Success vs. Failure')

plt.subplot(1, 2, 2)
diff = success_rates - failure_rates
plt.hist(diff, bins=20)
plt.xlabel('Difference in Firing Rate (Success - Failure) Hz')
plt.ylabel('Count')
plt.title('Distribution of Firing Rate Differences')
plt.tight_layout()
plt.savefig("success_vs_failure_firing_rates.png")

# Analyze firing rates by trial type
print("\nAnalyzing neural activity by trial type...")

trial_types = trials["trial_type"][:]
type1 = trial_types == 1  # linear map visible (NTS)
type3 = trial_types == 3  # fully occluded (MNAV)

# Create arrays to store firing rates by trial type
type1_rates = np.zeros(n_units)
type3_rates = np.zeros(n_units)

# Calculate firing rates for each trial type
for i in range(n_units):
    spike_times = units["spike_times"][i]
    
    # Type 1 trials
    type1_spike_count = 0
    type1_duration = 0
    for j in range(len(type1)):
        if type1[j]:
            start_time = trials["start_time"][j]
            stop_time = trials["stop_time"][j]
            duration = stop_time - start_time
            type1_duration += duration
            
            # Count spikes that fall within this trial's time window
            trial_spikes = np.sum((spike_times >= start_time) & (spike_times <= stop_time))
            type1_spike_count += trial_spikes
    
    if type1_duration > 0:
        type1_rates[i] = type1_spike_count / type1_duration
    
    # Type 3 trials
    type3_spike_count = 0
    type3_duration = 0
    for j in range(len(type3)):
        if type3[j]:
            start_time = trials["start_time"][j]
            stop_time = trials["stop_time"][j]
            duration = stop_time - start_time
            type3_duration += duration
            
            # Count spikes that fall within this trial's time window
            trial_spikes = np.sum((spike_times >= start_time) & (spike_times <= stop_time))
            type3_spike_count += trial_spikes
    
    if type3_duration > 0:
        type3_rates[i] = type3_spike_count / type3_duration

# Plot the comparison of firing rates by trial type
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(type1_rates, type3_rates)
max_rate = max(np.max(type1_rates), np.max(type3_rates)) * 1.1
plt.plot([0, max_rate], [0, max_rate], 'k--')
plt.xlabel('Firing Rate During Type 1 Trials (Hz)')
plt.ylabel('Firing Rate During Type 3 Trials (Hz)')
plt.title('Comparison of Firing Rates by Trial Type')

plt.subplot(1, 2, 2)
diff = type3_rates - type1_rates
plt.hist(diff, bins=20)
plt.xlabel('Difference in Firing Rate (Type 3 - Type 1) Hz')
plt.ylabel('Count')
plt.title('Distribution of Firing Rate Differences')
plt.tight_layout()
plt.savefig("trial_type_firing_rates.png")

# Analyze neural activity by landmark
print("\nAnalyzing neural activity by landmark...")

# Get landmark data
start_landmarks = trials["curr"][:]
target_landmarks = trials["target"][:]

# Select a subset of units to visualize (pick 5 units with highest firing rates)
top_unit_indices = np.argsort(fr)[-5:]
top_unit_ids = units["id"][top_unit_indices]

# Create a heatmap of firing rates for different start-target landmark combinations
# for each of these units
for unit_idx in range(len(top_unit_indices)):
    unit_index = top_unit_indices[unit_idx]
    unit_id = units["id"][unit_index]
    
    # Get spike times for this unit
    spike_times = units["spike_times"][unit_index]
    
    # Create a matrix to store firing rates for each start-target combination
    unique_landmarks = np.unique(np.concatenate([start_landmarks, target_landmarks]))
    landmark_matrix = np.zeros((len(unique_landmarks), len(unique_landmarks)))
    count_matrix = np.zeros((len(unique_landmarks), len(unique_landmarks)))
    
    # Iterate through trials
    for j in range(len(trials["id"])):
        start = start_landmarks[j]
        target = target_landmarks[j]
        
        # Find indices in the unique_landmarks array
        start_idx = np.where(unique_landmarks == start)[0][0]
        target_idx = np.where(unique_landmarks == target)[0][0]
        
        # Calculate firing rate for this trial
        start_time = trials["start_time"][j]
        stop_time = trials["stop_time"][j]
        duration = stop_time - start_time
        
        if duration > 0:
            # Count spikes that fall within this trial's time window
            trial_spikes = np.sum((spike_times >= start_time) & (spike_times <= stop_time))
            rate = trial_spikes / duration
            
            # Add to the matrices
            landmark_matrix[start_idx, target_idx] += rate
            count_matrix[start_idx, target_idx] += 1
    
    # Average the firing rates
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = np.divide(landmark_matrix, count_matrix)
        avg_matrix = np.nan_to_num(avg_matrix)
    
    # Create a custom colormap that goes from white to blue
    colors = [(1, 1, 1), (0, 0, 0.8)]
    cmap_name = 'white_blue'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_matrix, cmap=cm)
    plt.colorbar(label='Average Firing Rate (Hz)')
    plt.xticks(np.arange(len(unique_landmarks)), [f"{int(l)}" for l in unique_landmarks])
    plt.yticks(np.arange(len(unique_landmarks)), [f"{int(l)}" for l in unique_landmarks])
    plt.xlabel('Target Landmark')
    plt.ylabel('Start Landmark')
    plt.title(f'Unit {unit_id} Firing Rate by Landmark Combination')
    plt.savefig(f"unit_{unit_id}_landmark_firing_rates.png")

print("Analysis complete. Check the plots for detailed results.")