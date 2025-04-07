# This script loads and visualizes two summary images from the NWB file:
# average image (mean pixel intensity) and correlation image (pixelwise temporal correlations).
# Both images are shown side by side and saved as tmp_scripts/summary_images.png

import matplotlib.pyplot as plt
import lindi
import pynwb
import numpy as np

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/be84b6ff-7016-4ed8-af63-aa0e07c02530/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwbfile = pynwb.NWBHDF5IO(file=f, mode='r').read()

avg_image = nwbfile.processing['ophys']['SummaryImages_chan1']['average'].data[:]
corr_image = nwbfile.processing['ophys']['SummaryImages_chan1']['correlation'].data[:]

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(avg_image, cmap='gray')
plt.title('Average Intensity')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(corr_image, cmap='gray')
plt.title('Correlation Image')
plt.axis('off')

plt.tight_layout()
plt.savefig("tmp_scripts/summary_images.png")