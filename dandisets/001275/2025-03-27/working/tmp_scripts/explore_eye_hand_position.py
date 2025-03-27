# Explore the eye and hand position data in the NWB file
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

behavior = nwb.processing["behavior"]

# Plot eye position
eye_position = behavior["eye_position"]
eye_position_data = eye_position.data[:100000]
eye_position_timestamps = eye_position.timestamps[:100000]

plt.figure(figsize=(10, 5))
plt.plot(eye_position_timestamps, eye_position_data[:, 0], label="X position[:100000]")
plt.plot(eye_position_timestamps, eye_position_data[:, 1], label="Y position[:100000]")
plt.xlabel("Time (s)")
plt.ylabel("Eye position")
plt.title("Eye position over time")
plt.legend()
plt.savefig("tmp_scripts/eye_position.png")
plt.close()

# Plot hand position
hand_position = behavior["hand_position"]
hand_position_data = hand_position.data[:100000]
hand_position_timestamps = hand_position.timestamps[:100000]

plt.figure(figsize=(10, 5))
plt.plot(hand_position_timestamps, hand_position_data, label="Hand position[:100000]")
plt.xlabel("Time (s)")
plt.ylabel("Hand position")
plt.title("Hand position over time")
plt.legend()
plt.savefig("tmp_scripts/hand_position.png")
plt.close()