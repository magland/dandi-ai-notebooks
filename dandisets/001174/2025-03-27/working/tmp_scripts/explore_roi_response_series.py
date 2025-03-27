import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/de07db56-e7f3-4809-9972-755c51598e8d/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

RoiResponseSeries = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]
data = RoiResponseSeries.data
rate = RoiResponseSeries.rate
num_rois = data.shape[1]

# Plot the activity of the first 3 ROIs over time
num_rois_to_plot = min(3, num_rois)  # Ensure we don't try to plot more ROIs than available
time = np.arange(0, data.shape[0] / rate, 1 / rate)
fig, axes = plt.subplots(num_rois_to_plot, 1, figsize=(15, 5 * num_rois_to_plot))

for i in range(num_rois_to_plot):
    if num_rois_to_plot == 1:
      ax = axes
    else:
      ax = axes[i]
    ax.plot(time, data[:, i])
    ax.set_title(f'ROI {i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescence')

plt.tight_layout()
plt.savefig("tmp_scripts/roi_response_series.png")
plt.close()