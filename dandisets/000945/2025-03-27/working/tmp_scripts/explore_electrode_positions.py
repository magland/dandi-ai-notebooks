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

# Get electrode positions
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

# Save the plot to a file
plt.savefig("tmp_scripts/electrode_positions.png")
plt.close()