"""
This script explores the basic structure of the NWB file and examines the trials data.
It provides an overview of the mental navigation task structure and parameters.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic dataset information
print("\nDataset Information:")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")

# Print subject information
print("\nSubject Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")

# Explore the trials data
print("\nTrials Data:")
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['id'])}")

# Let's look at the types of trials
trial_types = trials["trial_type"][:]
unique_trial_types = np.unique(trial_types)
print(f"Unique trial types: {unique_trial_types}")

# Count occurrences of each trial type
for trial_type in unique_trial_types:
    count = np.sum(trial_types == trial_type)
    if trial_type == 1:
        desc = "linear map visible (NTS)"
    elif trial_type == 2:
        desc = "centre visible, periphery occluded"
    elif trial_type == 3:
        desc = "fully occluded (MNAV)"
    else:
        desc = "unknown"
    print(f"Trial type {trial_type} ({desc}): {count} trials")

# Calculate success rate by trial type
print("\nSuccess rates by trial type:")
for trial_type in unique_trial_types:
    type_indices = trial_types == trial_type
    success_rate = np.mean(trials["succ"][:][type_indices]) * 100
    print(f"Trial type {trial_type}: {success_rate:.2f}% success rate")

# Create figure for success rate by trial type
plt.figure(figsize=(10, 6))
for i, trial_type in enumerate(unique_trial_types):
    type_indices = trial_types == trial_type
    success_rate = np.mean(trials["succ"][:][type_indices]) * 100
    plt.bar(i, success_rate)

plt.xticks(np.arange(len(unique_trial_types)), [f"Type {int(t)}" for t in unique_trial_types])
plt.ylabel("Success Rate (%)")
plt.title("Success Rate by Trial Type")
plt.savefig("success_rate_by_trial_type.png")

# Let's look at the distribution of start and target landmarks
print("\nLandmark distribution:")
start_landmarks = trials["curr"][:]
target_landmarks = trials["target"][:]

unique_start = np.unique(start_landmarks)
unique_target = np.unique(target_landmarks)

print(f"Unique start landmarks: {unique_start}")
print(f"Unique target landmarks: {unique_target}")

# Create a plot of the distribution of start and target combinations
start_target_pairs = np.zeros((len(unique_start), len(unique_target)))
for i in range(len(trials["id"])):
    start = start_landmarks[i]
    target = target_landmarks[i]
    start_idx = np.where(unique_start == start)[0][0]
    target_idx = np.where(unique_target == target)[0][0]
    start_target_pairs[start_idx, target_idx] += 1

plt.figure(figsize=(10, 8))
plt.imshow(start_target_pairs, cmap='viridis')
plt.colorbar(label='Number of trials')
plt.xticks(np.arange(len(unique_target)), [f"{int(t)}" for t in unique_target])
plt.yticks(np.arange(len(unique_start)), [f"{int(s)}" for s in unique_start])
plt.xlabel('Target Landmark')
plt.ylabel('Start Landmark')
plt.title('Distribution of Start-Target Landmark Combinations')
plt.savefig("start_target_combinations.png")

# Let's look at response times
print("\nResponse time analysis:")
rt = trials["rt"][:]
print(f"Average response time: {np.mean(rt):.4f}s")
print(f"Median response time: {np.median(rt):.4f}s")
print(f"Min response time: {np.min(rt):.4f}s")
print(f"Max response time: {np.max(rt):.4f}s")

# Plot distribution of response times
plt.figure(figsize=(10, 6))
plt.hist(rt, bins=30, alpha=0.7)
plt.xlabel('Response Time (s)')
plt.ylabel('Count')
plt.title('Distribution of Response Times')
plt.axvline(np.median(rt), color='r', linestyle='--', label=f'Median: {np.median(rt):.2f}s')
plt.legend()
plt.savefig("response_time_distribution.png")

# Look at the difference between target vector (ta) and produced vector (tp)
print("\nVector analysis:")
ta = trials["ta"][:]  # Actual vector (seconds)
tp = trials["tp"][:]  # Produced vector (seconds)
error = tp - ta
print(f"Average error: {np.mean(error):.4f}s")
print(f"Median error: {np.median(error):.4f}s")
print(f"Average absolute error: {np.mean(np.abs(error)):.4f}s")

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(error, bins=30, alpha=0.7)
plt.xlabel('Error (Produced - Actual) in seconds')
plt.ylabel('Count')
plt.title('Distribution of Errors in Time Navigation')
plt.axvline(0, color='r', linestyle='--', label='Zero Error')
plt.axvline(np.mean(error), color='g', linestyle='--', label=f'Mean Error: {np.mean(error):.2f}s')
plt.legend()
plt.savefig("error_distribution.png")