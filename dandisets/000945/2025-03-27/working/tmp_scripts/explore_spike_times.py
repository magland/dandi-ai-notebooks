import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get spike times for a few units
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

plt.savefig("tmp_scripts/spike_times.png")
plt.close()