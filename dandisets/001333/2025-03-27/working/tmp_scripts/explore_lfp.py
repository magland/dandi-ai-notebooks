import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import seaborn as sns
sns.set_theme()

# Load the NWB file
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001333/assets/5409700b-e080-44e6-a6db-1d3e8890cd6c/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the LFP data
ecephys = nwb.processing["ecephys"]
LFP = ecephys["LFP"]["LFP"]
lfp_data = LFP.data[:]  # Load all data into memory
lfp_rate = LFP.rate
electrodes = nwb.ec_electrodes

# Get electrode information
electrode_ids = electrodes["id"].data[:]
electrode_locations = electrodes["location"].data[:]

# Select all electrodes to plot
num_electrodes = len(electrode_ids)

plt.figure(figsize=(10, 6))

plt.plot(lfp_data)  # Plot LFP
plt.xlabel("Data Points")
plt.ylabel("LFP Signal")
plt.title("LFP Signal for All Electrodes")
plt.savefig("tmp_scripts/lfp_allelectrodes.png")

print("LFP Plotting script complete.")