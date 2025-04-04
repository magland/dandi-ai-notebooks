# Plots the average and correlation images for channel 1.
import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

average = nwb.processing["ophys"]["SummaryImages_chan1"]["average"].data[:]
correlation = nwb.processing["ophys"]["SummaryImages_chan1"]["correlation"].data[:]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(average, cmap="gray")
plt.title("Average Image")

plt.subplot(1, 2, 2)
plt.imshow(correlation, cmap="gray")
plt.title("Correlation Image")

plt.savefig("tmp_scripts/summary_images.png")
plt.close()