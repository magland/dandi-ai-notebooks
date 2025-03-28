# %% [markdown]
# AI-Generated Notebook for DANDI:001363
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified.
# Please be cautious when interpreting the code or results.
#
# This notebook explores the data in DANDI Archive Dandiset 001363, "Neural Spiking Data in the Rat Somatosensory Cortex Using a Flexible Electrode Responding to Transcranial Focused Ultrasound".
#
# This dataset contains neural spiking data in the rat somatosensory cortex using a flexible electrode responding to transcranial focused ultrasound.
#
# The notebook will guide you through the following steps:
#
# 1.  Loading the Dandiset metadata
# 2.  Exploring the dataset structure
# 3.  Accessing and visualizing sample data from NWB files
# 4.  Example analyses that might be relevant to the dataset's content
#
# Before running this notebook, make sure you have the following packages installed:
#
# ```bash
# pip install dandi pynwb lindi matplotlib seaborn
# ```

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001363")
assets = list(dandiset.get_assets())
print(f"Number of assets: {len(assets)}")
assets

# %% [markdown]
# We can inspect the Dandiset metadata to learn more about the dataset.

# %%
dandiset.get_metadata()

# %% [markdown]
# We can see the description of the dataset:
#
# > In this study, we investigate the neuronal response to transcranial focused ultrasound stimulation (tFUS) on the somatosensory cortex using a 128-element array transducer and a chronically implanted flexible electrode. This flexible electrode allows us to study higher intensities of tFUS which are impossible with a rigid electrode due to the vibration artifacts that are created. Here we test 5 different levels of intensity including 100, 400, 700, 1000, and 1300 kPa. We then tested the effect of varying duty cycle while keeping the pulse repetition frequency (PRF) constant while using the highest intensity (1300 kPa), testing duty cycles of 0.6%, 6%, 30%, 60%, and 90% while holding PRF at 1500 Hz. Finally we tested the effect of varying PRF while holding duty cycle constant, testing PRFs of 30, 300, 1500, 3000, and 4500 Hz with a duty cycle of 30%. In each of these, the fundamental frequency of ultrasound was 1500 kHz, and the ultrasound duration was 67 ms, with trials performed every 2 seconds, with a jitter of 10%. Each recording has 505 trials.

# %% [markdown]
# Now let's explore the contents of the NWB file.
#
# This is how you would access data in this particular NWB file using lindi and pynwb.

# %%
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# %% [markdown]
# Load the NWB file

# %%
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001363/assets/b8de194c-d6ad-42e4-9e8f-bddffd2dc86b/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# %% [markdown]
# Information about the NWB file

# %%
nwb.session_description # (str) Rat Ultrasound Array Stimulation

# %%
nwb.identifier # (str) BH643_4500_67_50V

# %%
nwb.session_start_time # (datetime) 2024-12-20T19:53:03.000000-05:00

# %% [markdown]
# Access the ElectricalSeries data

# %%
electrical_series = nwb.acquisition["ElectricalSeries"]
data = electrical_series.data
rate = electrical_series.rate
electrodes = nwb.electrodes

# %% [markdown]
# Plot the first few seconds of data from a few channels

# %%
num_channels = 4
duration = 5  # seconds
num_samples = int(duration * rate)
channels = range(num_channels)

plt.figure(figsize=(10, 6))
for i, channel in enumerate(channels):
    plt.plot(np.linspace(0, duration, num_samples), data[:num_samples, channel] + i * 100, label=f"Channel {channel}")  # Add offset for clarity
plt.xlabel("Time (s)")
plt.ylabel("Voltage (uV) + Offset")
plt.title("Sample Data from ElectricalSeries")
plt.legend()
plt.savefig("tmp_scripts/electrical_series.png")
plt.show()

# %% [markdown]
# The plot shows constant voltage levels across multiple channels over time. This could indicate baseline data or a test signal. The plot is informative in that it shows the channel separation.

# %% [markdown]
# Plot electrode locations

# %%
electrode_x = electrodes["x"].data[:]
electrode_y = electrodes["y"].data[:]

plt.figure(figsize=(6, 6))
plt.scatter(electrode_x, electrode_y)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Electrode Locations")
plt.savefig("tmp_scripts/electrode_locations.png")
plt.show()

# %% [markdown]
# The electrode locations plot shows two vertical columns of data points, representing the spatial configuration of electrodes. The electrodes are placed on two vertical lines, evenly spaced.

# %% [markdown]
# Access trials data

# %%
trials = nwb.intervals["trials"]
start_time = trials["start_time"].data[:]
stop_time = trials["stop_time"].data[:]

# %% [markdown]
# Plot trials start and stop times

# %%
plt.figure(figsize=(10, 6))
plt.plot(start_time, label="Start Time")
plt.plot(stop_time, label="Stop Time")
plt.xlabel("Trial Index")
plt.ylabel("Time (s)")
plt.title("Trials Start and Stop Times")
plt.legend()
plt.savefig("tmp_scripts/trials_start_stop.png")
plt.show()

# %% [markdown]
# The trials start and stop times plot shows the start and stop times of trials, with overlapping lines indicating that the start and stop times are almost identical for each trial. The lines exhibit a linear trend, suggesting that both start and stop times increase uniformly with each trial.