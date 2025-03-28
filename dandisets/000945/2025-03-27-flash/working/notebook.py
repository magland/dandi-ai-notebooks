# %% [markdown]
# # DANDI Archive - Exploratory Analysis of Dandiset 000945 (AI-Generated)
#
# **Warning:** This notebook was AI-generated with human supervision and has not been fully verified. Use caution when interpreting the code or results.
#
# This notebook provides an exploratory analysis of Dandiset 000945, titled "Neural Spiking Data in the Awake Rat Somatosensory Cortex Responding to Trials of Transcranial Focused Ultrasound Stimulation." The dataset contains neural spiking data recorded from the somatosensory cortex of awake rats during transcranial focused ultrasound stimulation (tFUS) experiments. The experiments tested different pulse repetition frequencies (PRFs) of ultrasound stimulation using a 128-element random array ultrasound transducer. Chronic electrophysiological recordings were acquired using 32-channel NeuroNexus electrodes.
#
# The notebook will guide you through the following steps:
# 1.  Loading the Dandiset metadata
# 2.  Accessing and exploring the NWB file structure
# 3.  Visualizing electrode positions
# 4.  Examining sample unit spike times
#
# Before running this notebook, make sure you have the following packages installed:
# ```bash
# pip install dandi lindi pynwb matplotlib seaborn
# ```

# %%
# Load the Dandiset assets using the DANDI API
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())

# Print the available assets
for asset in assets:
    print(f"{asset.path}: {asset.identifier}")

# %% [markdown]
# ## Accessing and Exploring the NWB File
#
# We will access and explore the contents of the NWB file: `sub-BH497/sub-BH497_ses-20240310T143729_ecephys.nwb`.
#
# This section demonstrates how to load data from the NWB file using `lindi` and `pynwb`. Please note that you should use the `lindi` URL shown below rather than the direct download asset URL.

# %%
# Load the NWB file
import pynwb
import lindi

lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print some information about the NWB file
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")

# %% [markdown]
# ## Visualizing Electrode Positions
#
# This section visualizes the electrode positions in 3D space. The plot shows the spatial arrangement of the electrodes, which can be helpful for understanding the recording setup.

# %%
# Get electrode positions
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from mpl_toolkits.mplot3d import Axes3D

electrodes = nwb.ec_electrodes
electrode_ids = electrodes["id"].data[:]
x = electrodes["x"].data[:]
y = electrodes["y"].data[:]
z = electrodes["z"].data[:]

# Create a 3D scatter plot of electrode positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Electrode Positions")
plt.show()

# %% [markdown]
# ## Examining Sample Unit Spike Times
#
# This section displays the spike times for a few sample units. The raster plot shows the temporal distribution of neural spikes, allowing for analysis of firing patterns and synchrony among different units.

# %%
# Get spike times for a few units
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

units = nwb.units
unit_ids = units["id"].data[:]
num_units_to_plot = min(5, len(unit_ids))  # Plot up to 5 units

plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
for i in range(num_units_to_plot):
    spike_times = units["spike_times"][i]
    plt.vlines(spike_times, i - 0.4, i + 0.4, color='k', linewidth=0.8)  # Use vlines for clearer spike representation

plt.xlabel("Time (s)")
plt.ylabel("Unit ID")  # Changed ylabel to Unit Number
plt.title("Spike Times for Sample Units")
plt.yticks(range(num_units_to_plot), unit_ids[:num_units_to_plot])  # Use actual unit_ids as y-ticks and ensure they match

plt.xlim(0, 10)  # Limit x-axis to the first 10 seconds for clarity
plt.show()

# %% [markdown]
# ## Further Analysis
#
# This notebook provides a basic introduction to exploring the Dandiset. Further analysis could include:
#
# *   Analyzing trial-related activity
# *   Examining cell type labels
# *   Investigating the effects of different PRFs on neural spiking
# *   Performing more sophisticated analyses of spike patterns and synchrony