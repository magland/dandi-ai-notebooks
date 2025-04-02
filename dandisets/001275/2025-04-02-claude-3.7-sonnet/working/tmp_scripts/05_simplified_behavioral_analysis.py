"""
This script provides a simplified analysis of behavioral data from the mental navigation task.
It focuses on extracting summary statistics without processing the entire dataset.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trial information
print("Extracting trial data...")
trials = nwb.intervals["trials"]
trial_types = trials["trial_type"][:]
start_landmarks = trials["curr"][:]
target_landmarks = trials["target"][:]
success = trials["succ"][:]
start_times = trials["start_time"][:]
stop_times = trials["stop_time"][:]
joy_on_times = trials["joy1onttl"][:]
joy_off_times = trials["joy1offttl"][:]

# Calculate movement durations (time between joystick press and release)
movement_durations = joy_off_times - joy_on_times

# Filter out invalid values (negative durations or missing timestamps)
valid_indices = (movement_durations > 0) & (joy_on_times > 0) & (joy_off_times > 0)
valid_durations = movement_durations[valid_indices]
valid_types = trial_types[valid_indices]
valid_start = start_landmarks[valid_indices]
valid_target = target_landmarks[valid_indices]
valid_success = success[valid_indices]

print(f"Total trials: {len(movement_durations)}")
print(f"Valid trials with joystick data: {np.sum(valid_indices)}")

# Calculate distance between landmarks (assuming linear arrangement)
distances = np.abs(valid_target - valid_start)

# Plot distribution of movement durations
plt.figure(figsize=(10, 6))
plt.hist(valid_durations, bins=30, alpha=0.7)
plt.xlabel('Movement Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Joystick Movement Durations')
plt.savefig('movement_duration_distribution.png')

# Create box plot of durations by trial type
plt.figure(figsize=(10, 6))
data = []
labels = []

# Get durations for each trial type
for t in np.unique(valid_types):
    indices = valid_types == t
    if np.sum(indices) > 0:
        data.append(valid_durations[indices])
        if t == 1:
            labels.append(f"Type {int(t)} (Linear Map Visible)")
        elif t == 3:
            labels.append(f"Type {int(t)} (Fully Occluded)")
        else:
            labels.append(f"Type {int(t)}")

plt.boxplot(data)
plt.xticks(range(1, len(labels)+1), labels)
plt.ylabel('Movement Duration (s)')
plt.title('Movement Duration by Trial Type')
plt.savefig('duration_by_trial_type.png')

# Plot success rate by distance between landmarks
plt.figure(figsize=(10, 6))
unique_distances = np.unique(distances)
success_rates = []

for dist in unique_distances:
    indices = distances == dist
    success_rate = np.mean(valid_success[indices]) * 100
    success_rates.append(success_rate)

plt.bar(unique_distances, success_rates)
plt.xlabel('Distance Between Landmarks')
plt.ylabel('Success Rate (%)')
plt.xticks(unique_distances)
plt.title('Success Rate by Landmark Distance')
plt.savefig('success_rate_by_distance.png')

# Plot average duration by distance between landmarks
plt.figure(figsize=(10, 6))
avg_durations = []
std_durations = []

for dist in unique_distances:
    indices = distances == dist
    avg_durations.append(np.mean(valid_durations[indices]))
    std_durations.append(np.std(valid_durations[indices]))

plt.errorbar(unique_distances, avg_durations, yerr=std_durations, marker='o', linestyle='-')
plt.xlabel('Distance Between Landmarks')
plt.ylabel('Average Movement Duration (s)')
plt.xticks(unique_distances)
plt.title('Average Movement Duration by Landmark Distance')
plt.grid(True, alpha=0.3)
plt.savefig('avg_duration_by_distance.png')

# Plot average duration by distance for each trial type
plt.figure(figsize=(10, 6))

for t in np.unique(valid_types):
    type_indices = valid_types == t
    avg_by_dist = []
    
    for dist in unique_distances:
        dist_indices = distances == dist
        combined_indices = type_indices & dist_indices
        if np.sum(combined_indices) > 0:
            avg_by_dist.append(np.mean(valid_durations[combined_indices]))
        else:
            avg_by_dist.append(np.nan)
    
    if t == 1:
        label = "Type 1 (Linear Map Visible)"
    elif t == 3:
        label = "Type 3 (Fully Occluded)"
    else:
        label = f"Type {int(t)}"
    
    plt.plot(unique_distances, avg_by_dist, marker='o', label=label)

plt.xlabel('Distance Between Landmarks')
plt.ylabel('Average Movement Duration (s)')
plt.xticks(unique_distances)
plt.legend()
plt.title('Average Movement Duration by Distance and Trial Type')
plt.grid(True, alpha=0.3)
plt.savefig('duration_by_distance_and_type.png')

print("Analysis complete. Check the output plots.")