# Explore LFP data from the NWB file.
# This script will:
# 1. Load the NWB file.
# 2. Access the LFP data.
# 3. Plot a segment of the LFP data for a few electrodes.

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the LFP data
ecephys = nwb.processing["ecephys"]
LFP = ecephys["LFP"]
lfp_data = LFP.data
electrodes = LFP.electrodes

# Select a subset of electrodes and time points
num_electrodes = 4
num_timepoints = 1000
electrode_indices = np.arange(num_electrodes)  # Select the first few electrodes
start_time = 0
end_time = start_time + num_timepoints / LFP.rate

# Get the LFP data for the selected electrodes and time points
lfp_subset = lfp_data[start_time:int(end_time * LFP.rate), electrode_indices]

# Create a time axis
time = np.linspace(start_time, end_time, num_timepoints)

# Plot the LFP data
plt.figure(figsize=(10, 6))
for i in range(num_electrodes):
    plt.plot(time, lfp_subset[:, i], label=f"Electrode {electrodes['id'][i]}")

plt.xlabel("Time (s)")
plt.ylabel("LFP (uV)")
plt.title("LFP Data for Selected Electrodes")
plt.legend()
plt.savefig("tmp_scripts/lfp_plot.png")
plt.close() # important to prevent hanging