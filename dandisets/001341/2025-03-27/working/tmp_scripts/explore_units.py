import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np

# Load the NWB file
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/5738ae8a-dd82-425b-8966-bbcfd35649a7/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

    # Get the units data
    units = nwb.units
    firing_rate = units["firing_rate"].data[:]
    depth = units["depth"].data[:]
    layer = units["layer"].data[:]

    # Create a scatter plot of firing_rate vs depth, colored by layer
    plt.figure(figsize=(10, 6))
    unique_layers = np.unique(layer)
    for l in unique_layers:
        idx = layer == l
        plt.scatter(depth[idx], firing_rate[idx], label=l)

    plt.xlabel("Depth (um)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("Firing Rate vs Depth by Cortical Layer")
    plt.legend()
    plt.grid(True)

    # Save the plot to a PNG file
    plt.savefig("tmp_scripts/firing_rate_depth_layer.png")
    print("Plot saved to tmp_scripts/firing_rate_depth_layer.png")

except Exception as e:
    print(f"Error: {e}")