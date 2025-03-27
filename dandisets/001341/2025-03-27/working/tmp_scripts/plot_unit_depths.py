import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/5738ae8a-dd82-425b-8966-bbcfd35649a7/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the depths of the units
depths = nwb.units["depth"].data[:]
unit_ids = nwb.units["id"].data[:]

# Plot the depths
plt.figure(figsize=(10, 5))
plt.bar(unit_ids, depths)
plt.xlabel("Unit ID")
plt.ylabel("Depth (microns)")
plt.title("Depths of Units")
plt.savefig("tmp_scripts/unit_depths.png")
plt.close()