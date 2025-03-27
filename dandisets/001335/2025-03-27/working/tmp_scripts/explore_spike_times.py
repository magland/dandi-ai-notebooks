# Explore spike times from the NWB file.
# This script will:
# 1. Load the NWB file.
# 2. Access the units data (spike times).
# 3. Plot the spike times for a few units.

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the units data
units = nwb.units
spike_times = units["spike_times"]

# Select a subset of units to plot
num_units = 5
unit_indices = np.arange(num_units)

# Plot the spike times for each unit
plt.figure(figsize=(10, 6))
for i in unit_indices:
    spike_time_values = spike_times[i]
    plt.vlines(spike_time_values, i - 0.4, i + 0.4, linewidth=0.5, label=f"Unit {units['id'][i]}")

plt.xlabel("Time (s)")
plt.ylabel("Unit ID")
plt.title("Spike Times for Selected Units")
plt.legend()
plt.savefig("tmp_scripts/spike_times_plot.png")
plt.close() # important to prevent hanging