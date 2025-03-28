"""
Script to explore and visualize the LFP data from the NWB file.
The script generates histograms and summary statistics of LFP data for selected electrodes.

Instructions for the plot:
- Generate descriptive statistics for LFP data.
- Plot data distribution for a sample of electrodes.
- Save plots as PNG images in the tmp_scripts directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import lindi
import pynwb

# Load the NWB file using LindiH5pyFile and pynwb
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access and subset LFP data
lfp = nwb.processing["ecephys"]["LFP"]
electrodes = lfp.electrodes["id"].data[:]
data_sample = lfp.data[:1000, :5]  # Load a smaller segment for simplicity 

# Plot LFP data: 1000 time points for first 5 electrodes
plt.figure(figsize=(15, 8))
for i in range(data_sample.shape[1]):
    plt.plot(data_sample[:, i], label=f'Electrode {electrodes[i]}')

plt.xlabel('Time Points')
plt.ylabel('Voltage (µV)')
plt.title('LFP Data Sample from First 5 Electrodes')
plt.legend()
plt.savefig('tmp_scripts/lfp_sample_plot.png')
plt.close()