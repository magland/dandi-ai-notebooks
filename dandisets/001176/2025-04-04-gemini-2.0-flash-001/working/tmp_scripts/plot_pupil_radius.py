# Plots the pupil radius over time.
import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

pupil_raw_radius = nwb.acquisition["PupilTracking"]["pupil_raw_radius"]
timestamps = pupil_raw_radius.timestamps[:]
data = pupil_raw_radius.data[:]

plt.figure(figsize=(10, 5))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel("Pupil Radius (pixels)")
plt.title("Pupil Radius Over Time")
plt.savefig("tmp_scripts/pupil_radius.png")
plt.close()