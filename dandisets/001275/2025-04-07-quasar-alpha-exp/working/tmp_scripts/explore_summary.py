# This script loads the NWB file for Dandiset 001275 and generates a summary of trial statistics,
# unit properties, and electrode information. It creates several plots saved to PNG files
# in the tmp_scripts directory for use in the notebook.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import lindi
import os

# Ensure output directory exists
os.makedirs('tmp_scripts', exist_ok=True)

f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Basic metadata
print(f"Session description: {nwb.session_description}")
print(f"Number of trials: {len(nwb.intervals['trials']['id'])}")
print(f"Number of units: {len(nwb.processing['ecephys']['units']['id'].data)}")
print(f"Number of electrodes: {len(nwb.electrodes['id'].data)}")

trials = nwb.intervals["trials"]
units = nwb.processing["ecephys"]["units"]
electrodes = nwb.electrodes

# Trial success distribution
succ = trials["succ"][:]
plt.figure()
plt.hist(succ, bins=2)
plt.xlabel("Trial success (0=fail,1=success)")
plt.ylabel("Count")
plt.title("Trial Success Distribution")
plt.savefig("tmp_scripts/trial_success.png")
plt.close()

# Reaction time distribution
rt = trials["rt"][:]
plt.figure()
plt.hist(rt, bins=50)
plt.xlabel("Reaction time (s)")
plt.ylabel("Count")
plt.title("Reaction Time Distribution")
plt.savefig("tmp_scripts/reaction_time.png")
plt.close()

# Number of spikes per unit
n_spikes = units["n_spikes"].data[:]
plt.figure()
plt.hist(n_spikes, bins=30)
plt.xlabel("Number of spikes")
plt.ylabel("Units count")
plt.title("Number of Spikes per Unit")
plt.savefig("tmp_scripts/n_spikes_per_unit.png")
plt.close()

# Unit depths
depths = units["depth"].data[:]
plt.figure()
plt.hist(depths, bins=30)
plt.xlabel("Depth (microns)")
plt.ylabel("Number of units")
plt.title("Unit Depth Distribution")
plt.savefig("tmp_scripts/unit_depths.png")
plt.close()

# Electrode location scatter
rel_x = electrodes["rel_x"].data[:]
rel_y = electrodes["rel_y"].data[:]
plt.figure()
plt.scatter(rel_x, rel_y)
plt.xlabel("rel_x")
plt.ylabel("rel_y")
plt.title("Electrode Relative Coordinates")
plt.savefig("tmp_scripts/electrode_locations.png")
plt.close()