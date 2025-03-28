# %%
# This script loads the NWB file and plots the image data.
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the NWB file
file_path = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(file_path)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the image data
image_series = nwb.acquisition["SingleTimePointImaging"]
image_data = image_series.data[0, 9595-50:9595+50, 9595-50:9595+50]

# Plot the image
plt.figure(figsize=(8, 8))
plt.imshow(image_data, cmap='gray')
plt.title('DAPI Image (Central 100x100 pixels)')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.colorbar()

# Save the plot to a PNG file
plt.savefig('tmp_scripts/image.png')
plt.close()