import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/65a7e913-45c7-48db-bf19-b9f5e910110a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the units data
units = nwb.units

# Select a few units to plot
unit_ids = [0, 1, 2]

# Plot the spike times for each unit
plt.figure(figsize=(10, 6))
for unit_id in unit_ids:
    spike_times = units["spike_times"][unit_id]
    plt.vlines(spike_times, unit_id - 0.4, unit_id + 0.4, label=f"Unit {unit_id}")

plt.xlabel("Time (s)")
plt.ylabel("Unit ID")
plt.title("Spike Times for Selected Units")
plt.legend()
plt.savefig("tmp_scripts/spike_times_plot.png")
print("Spike times plot saved to tmp_scripts/spike_times_plot.png")

print("Done!")