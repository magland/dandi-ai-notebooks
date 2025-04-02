# %% [markdown]
# # Analysis of Mental Navigation Data in Primate Posterior Parietal Cortex
# 
# ## **⚠️ DISCLAIMER ⚠️**
# **This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.**
# 
# This notebook explores Dandiset 001275: "Mental navigation primate PPC Neupane_Fiete_Jazayeri", which contains neurophysiology data collected from macaque monkeys during a mental navigation task. 
# 
# ### Dataset Background
# 
# This dataset is associated with the study published in Nature: https://doi.org/10.1038/s41586-024-07557-z. It contains recordings from the posterior parietal cortex (PPC) during a mental navigation task. A companion dataset with recordings from the entorhinal cortex is available at: https://doi.org/10.48324/dandi.000897/0.240605.1710.
#
# ### Task Description
# 
# In the mental navigation task, the subject (a macaque monkey) is presented with a start and target landmark, sampled from a linear map of 6 landmarks on a screen. After a delay, the subject is given a go cue to navigate from the start to the target landmark using a joystick. The subject must deflect the joystick in the proper direction and hold it until they think they've arrived at the target landmark. The visual drift and intervening landmarks are occluded from view, making this a mental navigation task.
# 
# The task has different conditions:
# - **Type 1**: Linear map visible (NTS)
# - **Type 3**: Fully occluded (MNAV)
# 
# ### Required Packages
# 
# Before running this notebook, ensure you have the following packages installed:
# - pynwb
# - lindi
# - numpy 
# - matplotlib
# - pandas
# - seaborn

# %% [markdown]
# ## 1. Loading and Exploring the Dataset
# 
# First, let's load the Dandiset information and list the available assets:

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())

print(f"Total number of assets: {len(assets)}")
print("\nFirst few assets:")
for asset in assets[:5]:
    print(f"- {asset.path} ({asset.size / 1e6:.2f} MB)")

# %% [markdown]
# The dataset contains NWB files from two subjects: "amadeus" and "mahler". Each session has two files:
# 1. A behavior+ecephys file (relatively small, ~300-590MB)
# 2. A separate ecephys file (very large, ~16-287GB)
# 
# For this analysis, we'll focus on one of the behavior+ecephys files, which contains both behavioral data and a subset of the neural recordings.

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the seaborn style for better visualization
sns.set_theme()

# %% [markdown]
# Let's select a specific file to analyze. We'll use the behavior+ecephys file from subject "amadeus" from the session on 01/04/2020:

# %%
# Function to find a specific asset by path pattern
def find_asset(assets, pattern):
    for asset in assets:
        if pattern in asset.path:
            return asset
    return None

# Find the behavior+ecephys file for amadeus, session 01042020
asset = find_asset(assets, "sub-amadeus/sub-amadeus_ses-01042020_behavior+ecephys.nwb")
if asset:
    print(f"Found asset: {asset.path}")
    print(f"Asset ID: {asset.identifier}")
    print(f"Size: {asset.size / 1e6:.2f} MB")
    asset_url = f"https://api.dandiarchive.org/api/assets/{asset.identifier}/download/"
    print(f"Download URL: {asset_url}")
else:
    print("Asset not found")

# %% [markdown]
# Now let's load the NWB file using lindi and pynwb:

# %%
# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(f"https://lindi.neurosift.org/dandi/dandisets/001275/assets/{asset.identifier}/nwb.lindi.json")
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

# %% [markdown]
# ## 2. Exploring the Trial Structure
# 
# Let's explore the trials table to understand the structure of the task:

# %%
# Get the trials data
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['id'])}")

# Look at the available columns in the trials table
print("\nAvailable trial data fields:")
for column in trials.colnames:
    print(f"- {column}")

# %% [markdown]
# Now let's examine the types of trials and success rates:

# %%
# Extract trial types
trial_types = trials["trial_type"][:]
unique_trial_types = np.unique(trial_types)

# Count occurrences of each trial type
print("Trial type distribution:")
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
plt.show()

# %% [markdown]
# Let's look at the distribution of start and target landmarks in the trial structure:

# %%
# Get landmark data
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
plt.show()

# %% [markdown]
# The heatmap reveals how trials were distributed across different start-target combinations. We can see that:
# 
# 1. Same-landmark combinations (diagonal) were avoided
# 2. There are preferred combinations in the experimental design
# 3. Some combinations (e.g., 1→6, 2→6) have many more trials than others
# 
# Now, let's analyze the success rate as a function of the distance between landmarks:

# %%
# Calculate distances between landmarks (assuming linear arrangement 1-6)
distances = np.abs(target_landmarks - start_landmarks)

# Calculate success rate by distance
unique_distances = np.unique(distances)
success_rates = []

for dist in unique_distances:
    indices = distances == dist
    success_rate = np.mean(trials["succ"][:][indices]) * 100
    success_rates.append(success_rate)
    print(f"Distance {int(dist)}: {success_rate:.2f}% success rate")

# Plot success rate by distance
plt.figure(figsize=(10, 6))
plt.bar(unique_distances, success_rates)
plt.xlabel('Distance Between Landmarks')
plt.ylabel('Success Rate (%)')
plt.xticks(unique_distances)
plt.title('Success Rate by Landmark Distance')
plt.show()

# %% [markdown]
# We can observe a clear trend where success rates decrease as the distance between landmarks increases. This suggests that mental navigation becomes more challenging with greater distances.
# 
# ## 3. Behavioral Data Analysis
# 
# Let's now look at the behavioral data, focusing on the joystick movements (hand position) and performance metrics.

# %%
# Get hand position data
hand_position = nwb.processing["behavior"]["hand_position"]
hand_data = hand_position.data[:]
hand_timestamps = hand_position.timestamps[:]

print(f"Hand position data shape: {hand_data.shape}")
print(f"Hand position timestamps shape: {hand_timestamps.shape}")
print(f"Time range: {hand_timestamps[0]} to {hand_timestamps[-1]} seconds")

# %% [markdown]
# Let's examine the joystick movement durations (time between joystick press and release) across trials:

# %%
# Calculate movement durations (time between joystick press and release)
joy_on_times = trials["joy1onttl"][:]
joy_off_times = trials["joy1offttl"][:]
movement_durations = joy_off_times - joy_on_times

# Filter out invalid values (negative durations or missing timestamps)
valid_indices = (movement_durations > 0) & (joy_on_times > 0) & (joy_off_times > 0)
valid_durations = movement_durations[valid_indices]
valid_types = trial_types[valid_indices]
valid_start = start_landmarks[valid_indices]
valid_target = target_landmarks[valid_indices]
valid_success = trials["succ"][:][valid_indices]

print(f"Total trials: {len(movement_durations)}")
print(f"Valid trials with joystick data: {np.sum(valid_indices)}")

# Plot distribution of movement durations
plt.figure(figsize=(10, 6))
plt.hist(valid_durations, bins=30, alpha=0.7)
plt.xlabel('Movement Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Joystick Movement Durations')
plt.show()

# %% [markdown]
# Now let's compare movement durations between different trial types:

# %%
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
plt.show()

# %% [markdown]
# Let's analyze how movement duration varies with distance and trial type:

# %%
# Calculate distance between landmarks for valid trials
valid_distances = np.abs(valid_target - valid_start)

# Plot average duration by distance and trial type
plt.figure(figsize=(10, 6))

for t in np.unique(valid_types):
    type_indices = valid_types == t
    avg_by_dist = []
    
    for dist in np.unique(valid_distances):
        dist_indices = valid_distances == dist
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
    
    plt.plot(np.unique(valid_distances), avg_by_dist, marker='o', label=label)

plt.xlabel('Distance Between Landmarks')
plt.ylabel('Average Movement Duration (s)')
plt.xticks(np.unique(valid_distances))
plt.legend()
plt.title('Average Movement Duration by Distance and Trial Type')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# This interaction between visibility and distance suggests different cognitive processes for mental navigation under different visual feedback conditions:
# 
# 1. For **Linear Map Visible** trials, durations increase linearly with distance.
# 2. For **Fully Occluded** trials, durations plateau at longer distances.
# 
# This pattern suggests that when visual information is available, navigation time scales with distance. In contrast, when navigating purely mentally (occluded condition), subjects might rely on different strategies or reach the limits of mental navigation precision at longer distances.

# %% [markdown]
# ## 4. Neural Data Analysis
# 
# Now let's examine the neural data recorded from the posterior parietal cortex (PPC) during the mental navigation task.

# %%
# Get the units data
units = nwb.processing["ecephys"]["units"]

# Basic information about all units
n_units = len(units["id"])
print(f"Number of units: {n_units}")

# Get unit quality information
quality = units["quality"].data[:]
quality_counts = {}
for q in quality:
    if q in quality_counts:
        quality_counts[q] += 1
    else:
        quality_counts[q] = 1

print("\nUnit quality distribution:")
for q, count in quality_counts.items():
    print(f"- {q}: {count} units")

# Get firing rates
fr = units["fr"].data[:]

print(f"\nFiring rate statistics:")
print(f"- Average firing rate: {np.mean(fr):.2f} Hz")
print(f"- Min firing rate: {np.min(fr):.2f} Hz")
print(f"- Max firing rate: {np.max(fr):.2f} Hz")

# Plot firing rate distribution
plt.figure(figsize=(10, 6))
plt.hist(fr, bins=20)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.title('Distribution of Unit Firing Rates')
plt.show()

# %% [markdown]
# Let's examine the distribution of recording depths of the units:

# %%
# Get unit depths
depths = units["depth"].data[:]

print(f"Depth range: {np.min(depths):.2f} - {np.max(depths):.2f}")

# Plot firing rate vs depth
plt.figure(figsize=(10, 6))
plt.scatter(depths, fr, alpha=0.7)
plt.xlabel('Depth')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rate vs Depth')
plt.show()

# %% [markdown]
# Now let's look at the spiking activity of a single unit during trials. This is computationally intensive, so we'll select just one unit and a small number of trials:

# %%
# Select a unit (choose one with a good firing rate)
unit_indices = np.argsort(fr)[-5:]  # Get the indices of the 5 units with highest firing rates
selected_unit_idx = unit_indices[0]
selected_unit_id = units["id"][selected_unit_idx]
selected_spike_times = units["spike_times"][selected_unit_idx]

print(f"Selected unit ID: {selected_unit_id}")
print(f"Firing rate: {fr[selected_unit_idx]:.2f} Hz")
print(f"Number of spikes: {len(selected_spike_times)}")

# Select a subset of trials for performance reasons
max_trials = 20
trial_indices = np.random.choice(range(len(trials["id"])), max_trials, replace=False)

# Create a raster plot
plt.figure(figsize=(12, 8))

# Plot spike times relative to trial start
for j, trial_idx in enumerate(trial_indices):
    start_time = trials["start_time"][trial_idx]
    stop_time = trials["stop_time"][trial_idx]
    
    # Find spikes in this trial window
    trial_spikes = selected_spike_times[(selected_spike_times >= start_time) & (selected_spike_times <= stop_time)]
    
    # Plot relative to trial start
    if len(trial_spikes) > 0:
        relative_times = trial_spikes - start_time
        plt.scatter(relative_times, np.ones_like(trial_spikes) * j, marker='|', s=30, color='black')

plt.xlabel('Time from Trial Start (s)')
plt.ylabel('Trial Number')
plt.title(f'Unit {selected_unit_id} Spike Raster Plot')
plt.tight_layout()
plt.show()

# %% [markdown]
# The raster plot shows the spiking activity of a single unit across multiple trials. Each vertical line represents a spike, and each row represents a different trial. This visualization helps identify whether the unit has consistent firing patterns related to specific task events or phases.

# %% [markdown]
# ## 5. Relationship Between Neural Activity and Behavior
# 
# Let's examine how neural activity relates to task performance by comparing firing rates between successful and failed trials:

# %%
# We'll use a sample of units to make computation faster
sample_size = min(10, n_units)
sample_indices = np.random.choice(range(n_units), sample_size, replace=False)

# Initialize arrays to store firing rates
success_rates = np.zeros(sample_size)
failure_rates = np.zeros(sample_size)

# Get indices of successful and failed trials
success_indices = trials["succ"][:] == 1
failure_indices = trials["succ"][:] == 0

# Function to calculate mean firing rate for a given unit during specified trials
def calculate_mean_firing_rate(spike_times, trial_start_times, trial_stop_times):
    if len(trial_start_times) == 0:
        return 0
    
    total_spikes = 0
    total_duration = 0
    
    for start, stop in zip(trial_start_times, trial_stop_times):
        duration = stop - start
        total_duration += duration
        
        # Count spikes in this time window
        spikes_in_window = np.sum((spike_times >= start) & (spike_times <= stop))
        total_spikes += spikes_in_window
    
    if total_duration > 0:
        return total_spikes / total_duration
    else:
        return 0

# Calculate firing rates for sample units during successful and failed trials
for i, unit_idx in enumerate(sample_indices):
    spike_times = units["spike_times"][unit_idx]
    
    # Calculate rate during successful trials
    success_starts = trials["start_time"][:][success_indices]
    success_stops = trials["stop_time"][:][success_indices]
    success_rates[i] = calculate_mean_firing_rate(spike_times, success_starts, success_stops)
    
    # Calculate rate during failed trials
    failure_starts = trials["start_time"][:][failure_indices]
    failure_stops = trials["stop_time"][:][failure_indices]
    failure_rates[i] = calculate_mean_firing_rate(spike_times, failure_starts, failure_stops)

# Plot comparison of firing rates
plt.figure(figsize=(10, 6))
plt.scatter(success_rates, failure_rates, alpha=0.7)
max_rate = max(np.max(success_rates), np.max(failure_rates)) * 1.1
plt.plot([0, max_rate], [0, max_rate], 'k--')
plt.xlabel('Firing Rate During Successful Trials (Hz)')
plt.ylabel('Firing Rate During Failed Trials (Hz)')
plt.title('Comparison of Neural Firing Rates: Success vs. Failure')
plt.axis('equal')
plt.tight_layout()
plt.show()

# %% [markdown]
# The scatter plot compares each unit's firing rate during successful trials versus failed trials. Points above the diagonal line indicate units that fire more during failed trials, while points below the line indicate units that fire more during successful trials. This can help identify neurons that might be particularly involved in successful task performance.

# %% [markdown]
# ## 6. Eye Tracking Data Analysis
# 
# This dataset also includes eye tracking data. Let's briefly explore it:

# %%
# Get eye position data
eye_position = nwb.processing["behavior"]["eye_position"]
eye_data = eye_position.data[:]
eye_timestamps = eye_position.timestamps[:]

print(f"Eye position data shape: {eye_data.shape}")
print(f"Eye position timestamps shape: {eye_timestamps.shape}")

# Plot a sample of eye position data
sample_size = 10000  # Take a small sample to avoid plotting too many points
sample_indices = np.linspace(0, len(eye_timestamps)-1, sample_size, dtype=int)

plt.figure(figsize=(10, 8))
plt.plot(eye_data[sample_indices, 0], eye_data[sample_indices, 1], 'k.', alpha=0.1)
plt.xlabel('Horizontal Eye Position')
plt.ylabel('Vertical Eye Position')
plt.title('Sample of Eye Movement Data')
plt.axis('equal')
plt.show()

# %% [markdown]
# ## 7. Conclusion and Further Analysis Directions
#
# In this notebook, we've explored Dandiset 001275, which contains neurophysiology data from the posterior parietal cortex (PPC) of macaque monkeys during a mental navigation task. We've:
#
# 1. Examined the task structure and trial distribution
# 2. Analyzed behavioral performance metrics 
# 3. Explored neural activity during the task
# 4. Compared neural activity between successful and failed trials
# 5. Briefly examined eye tracking data
# 
# ### Key findings:
# 
# - Performance (success rate) decreases with increasing distance between landmarks
# - There's a strong effect of visual feedback (occluded vs. visible) on navigation strategies
# - Units in the PPC show diverse firing patterns during the mental navigation task
# 
# ### Further analysis directions:
# 
# - Decode mental trajectory from neural activity
# - Analyze population dynamics during navigation
# - Compare activity between PPC and entorhinal cortex datasets
# - Examine error trials to understand failure modes
# - Analyze relationship between eye movements and navigation success
# 
# This dataset offers rich opportunities to study the neural basis of mental navigation and spatial cognition. The code in this notebook can serve as a starting point for more sophisticated analyses.

# %% [markdown]
# ## References
#
# 1. Dataset: https://dandiarchive.org/dandiset/001275
# 2. Associated paper: https://doi.org/10.1038/s41586-024-07557-z
# 3. Companion dataset (entorhinal cortex): https://doi.org/10.48324/dandi.000897/0.240605.1710