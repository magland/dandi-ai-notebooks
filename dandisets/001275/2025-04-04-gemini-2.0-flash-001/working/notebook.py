# %% [markdown]
# # AI-Generated DANDI Notebook: Caution Advised
#
# ***This notebook was AI-generated using `dandi-notebook-gen` and has not been fully verified. Use caution when interpreting the code or results.***
#
# This notebook provides a starting point for exploring and analyzing data from Dandiset 001275, "Mental navigation primate PPC Neupane_Fiete_Jazayeri."
#
# The purpose of this notebook is to illustrate how to access and visualize data, not to draw scientific conclusions.
#
# Before running this notebook, make sure you have the following packages installed:
# ```bash
# pip install pynwb lindi matplotlib seaborn dandi
# ```

# %%
# Import necessary libraries
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# %% [markdown]
# ## 1. Introduction to the Dandiset
#
# Dandiset 001275, named "Mental navigation primate PPC Neupane_Fiete_Jazayeri," contains neurophysiology data collected from two primates during a mental navigation task.
# The data is associated with a previously published study (https://doi.org/10.1038/s41586-024-07557-z).
#
# The dataset includes data from the entorhinal cortex.
#
# Key metadata:
# - **Name:** Mental navigation primate PPC Neupane_Fiete_Jazayeri
# - **Description:** This dataset contains neurophysiology data collected from two primates during a mental navigation task
# - **License:** CC-BY-4.0
# - **Contributors:** Neupane, Sujaya
# - **Measurement Technique:** multi electrode extracellular electrophysiology recording technique

# %% [markdown]
# ## 2. Dataset Structure Exploration
#
# First, let's use the DANDI API to list all of the assets in the Dandiset.

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001275")
assets = list(dandiset.get_assets())
print(f"Number of assets in the dandiset: {len(assets)}")
for asset in assets:
    print(asset.path)

# %% [markdown]
# ## 3. Accessing and Visualizing Sample Data from NWB Files
#
# We will load data from the NWB file: `sub-amadeus/sub-amadeus_ses-01042020_ecephys.nwb`. This file contains extracellular electrophysiology data.

# %%
# Load the NWB file
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/0bbd0039-0f40-4eb2-b67a-93802fb3b28d/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
except Exception as e:
    print(f"Error loading NWB file: {e}")

# %% [markdown]
# Let's inspect the contents of the NWB file.

# %%
nwb.session_description

# %%
nwb.identifier

# %%
nwb.session_start_time

# %% [markdown]
# Now we can load the ElectricalSeries data.

# %%
# Get the ElectricalSeries object
try:
    electrical_series = nwb.acquisition["ElectricalSeriesVP0"]
    data = electrical_series.data
    rate = electrical_series.rate
    electrodes = electrical_series.electrodes
    electrode_ids = electrodes["id"].data[:]

except Exception as e:
    print(f"Error accessing ElectricalSeries data: {e}")

# %% [markdown]
# The shape of the data is:

# %%
data.shape

# %% [markdown]
# The sampling rate is:

# %%
rate

# %% [markdown]
# The electrode IDs are:

# %%
electrode_ids

# %% [markdown]
# Let's plot a small subset of the data to visualize the neural activity. We will plot 4 channels for 1000 timepoints.

# %%
# Select a subset of channels and time points
num_channels = 4
num_timepoints = 1000
start_channel = 0
start_time = 0

# Extract the data subset
try:
    subset_channels = electrode_ids[start_channel:start_channel + num_channels]
    channel_indices = np.where(np.isin(electrodes["id"].data[:], subset_channels))[0]
    subset_data = data[start_time:start_time + num_timepoints, channel_indices]
    time = np.arange(start_time, start_time + num_timepoints) / rate
except Exception as e:
    print(f"Error extracting data subset: {e}")

# %%
# Plot the data
try:
    plt.figure(figsize=(10, 6))
    for i in range(num_channels):
        plt.plot(time, subset_data[:, i], label=f"Channel {subset_channels[i]}")

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title("Extracellular Electrophysiology Data")
    plt.legend()
    plt.show() # displaying plot in notebook
    # plt.savefig("tmp_scripts/ecephys_plot.png")
    plt.close()

except Exception as e:
    print(f"Error plotting data: {e}")

# %% [markdown]
# The plot shows the voltage for four channels over time. The voltage for each channel remains close to zero during this time.

# %% [markdown]
# ## 4. Example Analyses
#
# Here are some examples of common analyses that might be relevant to this dataset:
#
# - Spike sorting
# - Local field potential (LFP) analysis
# - Analysis of neural activity during the mental navigation task

# %% [markdown]
# This notebook provides a basic introduction to the Dandiset and demonstrates how to access and visualize the data.