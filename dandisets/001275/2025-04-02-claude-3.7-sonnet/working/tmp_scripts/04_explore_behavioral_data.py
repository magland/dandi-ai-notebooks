"""
This script explores the behavioral data in the NWB file, specifically focusing on
joystick movements (hand position) during the mental navigation task.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the behavioral data (hand position)
print("Getting behavioral data...")
hand_position = nwb.processing["behavior"]["hand_position"]
hand_data = hand_position.data[:]
hand_timestamps = hand_position.timestamps[:]

print(f"Hand position data shape: {hand_data.shape}")
print(f"Hand position timestamps shape: {hand_timestamps.shape}")
print(f"Time range: {hand_timestamps[0]} to {hand_timestamps[-1]} seconds")

# Get trial information
trials = nwb.intervals["trials"]
trial_types = trials["trial_type"][:]
start_landmarks = trials["curr"][:]
target_landmarks = trials["target"][:]
success = trials["succ"][:]
start_times = trials["start_time"][:]
stop_times = trials["stop_time"][:]
joy_on_times = trials["joy1onttl"][:]
joy_off_times = trials["joy1offttl"][:]

# Sample a few successful trials of each type
print("Selecting sample trials for analysis...")
type1_success_indices = np.where((trial_types == 1) & (success == 1))[0]
type3_success_indices = np.where((trial_types == 3) & (success == 1))[0]

# Sample up to 5 trials of each type
max_samples = 5
type1_samples = type1_success_indices[:max_samples] if len(type1_success_indices) >= max_samples else type1_success_indices
type3_samples = type3_success_indices[:max_samples] if len(type3_success_indices) >= max_samples else type3_success_indices

print(f"Selected {len(type1_samples)} Type 1 trials and {len(type3_samples)} Type 3 trials")

# Function to extract hand position data for a given time window
def get_hand_data_for_window(start_time, stop_time, max_points=1000):
    # Find indices within the time window
    indices = np.where((hand_timestamps >= start_time) & (hand_timestamps <= stop_time))[0]
    
    # If too many points, subsample
    if len(indices) > max_points:
        indices = np.linspace(indices[0], indices[-1], max_points, dtype=int)
    
    return hand_timestamps[indices], hand_data[indices]

# Plot hand position trajectories for Type 1 trials (linear map visible)
plt.figure(figsize=(12, 8))
plt.suptitle('Hand Position Trajectories - Type 1 Trials (Linear Map Visible)', fontsize=16)

for i, trial_idx in enumerate(type1_samples):
    start_time = joy_on_times[trial_idx]  # Use joystick press time as start
    stop_time = joy_off_times[trial_idx]  # Use joystick release time as stop
    
    if start_time > 0 and stop_time > start_time:
        timestamps, position = get_hand_data_for_window(start_time, stop_time)
        
        # Normalize time to start at 0
        normalized_time = timestamps - timestamps[0]
        
        # Plot as a line with changing color to indicate time progression
        plt.subplot(len(type1_samples), 1, i+1)
        
        # When position is scalar, it's just magnitude, so we'll plot it directly
        if len(position.shape) == 1:
            sc = plt.scatter(normalized_time, position, c=normalized_time, cmap='viridis', 
                          s=20, alpha=0.7)
            plt.colorbar(sc, label='Time (s)')
            plt.ylabel('Position')
        else:
            # If position is 2D (x, y), we'll create a 2D plot
            sc = plt.scatter(position[:, 0], position[:, 1], c=normalized_time, cmap='viridis', 
                          s=20, alpha=0.7)
            plt.colorbar(sc, label='Time (s)')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
        
        plt.title(f"Trial {trial_idx}: Start={int(start_landmarks[trial_idx])}, Target={int(target_landmarks[trial_idx])}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("type1_hand_trajectories.png")

# Plot hand position trajectories for Type 3 trials (fully occluded)
plt.figure(figsize=(12, 8))
plt.suptitle('Hand Position Trajectories - Type 3 Trials (Fully Occluded)', fontsize=16)

for i, trial_idx in enumerate(type3_samples):
    start_time = joy_on_times[trial_idx]  # Use joystick press time as start
    stop_time = joy_off_times[trial_idx]  # Use joystick release time as stop
    
    if start_time > 0 and stop_time > start_time:
        timestamps, position = get_hand_data_for_window(start_time, stop_time)
        
        # Normalize time to start at 0
        normalized_time = timestamps - timestamps[0]
        
        # Plot as a line with changing color to indicate time progression
        plt.subplot(len(type3_samples), 1, i+1)
        
        # When position is scalar, it's just magnitude, so we'll plot it directly
        if len(position.shape) == 1:
            sc = plt.scatter(normalized_time, position, c=normalized_time, cmap='viridis', 
                          s=20, alpha=0.7)
            plt.colorbar(sc, label='Time (s)')
            plt.ylabel('Position')
        else:
            # If position is 2D (x, y), we'll create a 2D plot
            sc = plt.scatter(position[:, 0], position[:, 1], c=normalized_time, cmap='viridis', 
                          s=20, alpha=0.7)
            plt.colorbar(sc, label='Time (s)')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
        
        plt.title(f"Trial {trial_idx}: Start={int(start_landmarks[trial_idx])}, Target={int(target_landmarks[trial_idx])}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("type3_hand_trajectories.png")

# Calculate average duration by landmark distance
print("Analyzing joystick movement durations by landmark distance...")

# Calculate distances between landmarks (assuming linear arrangement 1-6)
distances = np.abs(target_landmarks - start_landmarks)

# Calculate movement durations (joystick press to release)
durations = joy_off_times - joy_on_times

# Filter out invalid durations
valid_indices = (durations > 0) & (joy_on_times > 0) & (joy_off_times > 0)
valid_distances = distances[valid_indices]
valid_durations = durations[valid_indices]
valid_types = trial_types[valid_indices]
valid_success = success[valid_indices]

# Group by distance and type
unique_distances = np.unique(valid_distances)
unique_types = np.unique(valid_types)

# Create figure for average durations
plt.figure(figsize=(10, 6))

for t in unique_types:
    avg_durations = []
    std_durations = []
    
    for d in unique_distances:
        # Get durations for successful trials with this distance and type
        indices = (valid_distances == d) & (valid_types == t) & (valid_success == 1)
        if np.sum(indices) > 0:
            avg_durations.append(np.mean(valid_durations[indices]))
            std_durations.append(np.std(valid_durations[indices]))
        else:
            avg_durations.append(0)
            std_durations.append(0)
    
    # Plot average durations by distance
    type_label = "Linear map visible" if t == 1 else "Fully occluded"
    plt.errorbar(unique_distances, avg_durations, yerr=std_durations, 
                 marker='o', linestyle='-', label=f"Type {int(t)}: {type_label}")

plt.xlabel('Landmark Distance')
plt.ylabel('Average Movement Duration (s)')
plt.title('Movement Duration by Landmark Distance and Trial Type')
plt.xticks(unique_distances)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("duration_by_distance.png")

print("Analysis complete. Check the output plots.")