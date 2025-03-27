# %% [markdown]
# AI-generated Notebook for Dandiset 001275 with Human Supervision

**Important:** This notebook was AI-generated with human supervision and has not been fully verified. Use caution when interpreting the code or results.

# Introduction to Dandiset 001275

This notebook provides an introduction to Dandiset 001275, "Mental navigation primate PPC Neupane_Fiete_Jazayeri," which contains neurophysiology data collected from two primates during a mental navigation task. The data is associated with a previously published study (https://doi.org/10.1038/s41586-024-07557-z).

The dataset includes data from the entorhinal cortex and covers units, electrode groups, processing modules, spatial series, and electrical series.

## Key Metadata

- **Name:** Mental navigation primate PPC Neupane_Fiete_Jazayeri
- **Description:** This dataset contains neurophysiology data collected from two primates during a mental navigation task.
- **Contributor:** Neupane, Sujaya
- **Number of Subjects:** 2

## Accessing the Data

The data is open access and available on the DANDI archive: https://dandiarchive.org/dandiset/001275/draft

To begin, you'll need to install the necessary packages:
```bash
pip install pynwb lindi matplotlib seaborn
```

# %%
# Use the DANDI API to list all of the assets in the Dandiset
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())
assets

# %% [markdown]
# Exploring the Dataset Structure

# %%
# Load an NWB file and explore its contents
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
# Replace with the appropriate asset from the dandiset
nwb_file_url = "https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_file_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# %% [markdown]
# Dataset Structure and Contents

# %%
# Print basic information about the NWB file

print("Session description:", nwb.session_description)
print("Identifier:", nwb.identifier)
print("Session start time:", nwb.session_start_time)

# %% [markdown]
# Exploring Trials Data

# %%
# Explore trials data
trials = nwb.intervals["trials"]
print("Trials start time:", trials["start_time"].data.shape)
print("Trials stop time:", trials["stop_time"].data.shape)

# %% [markdown]
# Accessing and Visualizing Sample Data

# %%
# Access eye position data
behavior = nwb.processing["behavior"]
eye_position = behavior["eye_position"]
eye_position_data = eye_position.data[:100000]
eye_position_timestamps = eye_position.timestamps[:100000]

# Plot eye position
plt.figure(figsize=(10, 5))
plt.plot(eye_position_timestamps, eye_position_data[:, 0], label="X position[:100000]")
plt.plot(eye_position_timestamps, eye_position_data[:, 1], label="Y position[:100000]")
plt.xlabel("Time (s)")
plt.ylabel("Eye position")
plt.title("Eye position over time")
plt.legend()
plt.show()

# %% [markdown]
# Hand Position Data

# %%
# Access hand position data
hand_position = behavior["hand_position"]
hand_position_data = hand_position.data[:100000]
hand_position_timestamps = hand_position.timestamps[:100000]

# Plot hand position
plt.figure(figsize=(10, 5))
plt.plot(hand_position_timestamps, hand_position_data, label="Hand position[:100000]")
plt.xlabel("Time (s)")
plt.ylabel("Hand position")
plt.title("Hand position over time")
plt.legend()
plt.show()

# %% [markdown]
# Units Data

# %%
# Access units data
units = nwb.processing["ecephys"]["units"]
print("Column names:", units.colnames)

unit_ids = units["id"].data[:]
print("Number of units:", len(unit_ids))

# Plot histogram of spike times for a random unit
unit_index = np.random.randint(0, len(unit_ids))
spike_times = units["spike_times"][unit_index]
plt.figure()
plt.hist(spike_times, bins=50)
plt.xlabel("Spike times")
plt.ylabel("Count")
plt.title(f"Spike times for unit {unit_ids[unit_index]}")
plt.show()

# %% [markdown]
# Examples of Common Analyses

# %%
# Example: Calculate the mean firing rate for each unit
units = nwb.processing["ecephys"]["units"]
firing_rates = units["fr"].data[:]
print("Mean firing rate for each unit:", firing_rates)
print("Mean firing rate across units:", np.mean(firing_rates))