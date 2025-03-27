# Explore RoiResponseSeries1 and create a plot
import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get RoiResponseSeries1 data and timestamps
ophys = nwb.processing["ophys"]
Fluorescence = ophys["Fluorescence"]
RoiResponseSeries1 = Fluorescence["RoiResponseSeries1"]
roi_data = RoiResponseSeries1.data[:5000, 0]  # Load a subset of the data
roi_timestamps = RoiResponseSeries1.timestamps[:5000] # Load a subset of the timestamps

# Plot the ROI response over time
plt.figure(figsize=(10, 5))
plt.plot(roi_timestamps, roi_data)
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (AU)")
plt.title("ROI Response Over Time")
plt.savefig("tmp_scripts/roi_response.png")
plt.close()