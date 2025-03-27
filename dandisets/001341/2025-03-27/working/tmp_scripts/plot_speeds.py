import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/5738ae8a-dd82-425b-8966-bbcfd35649a7/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the forward and lateral speed data
speed_for = nwb.acquisition["speed_for"].data[:]
speed_lat = nwb.acquisition["speed_lat"].data[:]
timestamps = nwb.acquisition["speed_for"].timestamps[:]

# Plot the speed data
plt.figure(figsize=(10, 5))
plt.plot(timestamps, speed_for, label="Forward Speed")
plt.plot(timestamps, speed_lat, label="Lateral Speed")
plt.xlabel("Time (s)")
plt.ylabel("Speed (cm/s)")
plt.title("Forward and Lateral Speed")
plt.legend()
plt.savefig("tmp_scripts/speeds.png")
plt.close()