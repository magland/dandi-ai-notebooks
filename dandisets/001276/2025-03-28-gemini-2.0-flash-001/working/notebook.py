# %% [markdown]
# AI-Generated Notebook for DANDI:001276
#
# **Warning:** This notebook was AI-generated using `dandi-notebook-gen` and has not been fully verified.
# Use caution when interpreting the code or results.
#
# Instructions for the user:
# Install the necessary packages using pip install lindi pynwb matplotlib seaborn

# %% [markdown]
# ## Introduction
#
# This notebook explores the Dandiset 001276, which contains data from the study:
# **NG-CANCAN Remote Targeting Electroporation: Impact of Burst Number Variation on Permeabilization Distribution in Confluent Cell Monolayers**
#
# The experiments focus on optimizing the CANCAN protocol and investigating the effect of varying burst numbers on permeabilization distribution across confluent cell monolayers.
# The CANCAN protocols utilized canceling pulses to minimize cell damage near the electrodes while targeting cells in the center of the electrode array.
#
# The Dandiset includes data from experiments conducted using a four-electrode array with an inter-electrode distance of 10.0 mm.
#
# This work was partially supported by NIH grant 1R21EY034258.
#
# Citation: Silkuniene, Giedre; Silkunas, Mantas; Pakhomov, Andrei (2025) NG-CANCAN Remote Targeting Electroporation: Impact of Burst Number Variation on Permeabilization Distribution in Confluent Cell Monolayers (Version draft) [Data set]. DANDI Archive. https://dandiarchive.org/dandiset/001276/draft

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001276")
assets = list(dandiset.get_assets())
print(assets)

# %% [markdown]
# ## Dataset Structure Exploration
#
# The Dandiset contains a number of NWB files, each associated with a specific subject and experimental condition.
# The following code lists the assets in the Dandiset:

# %% [markdown]
# ## Sample Data Access and Visualization
#
# We will now access and visualize sample data from one of the NWB files in the Dandiset.
# We will use the `lindi` and `pynwb` libraries to load the NWB file.
# In this example, we are loading from this NWB file: sub-P1-20240627-A2/sub-P1-20240627-A2_obj-1aoyzxh_image.nwb

# %%
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

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
plt.show()

# %% [markdown]
# The above code loads the NWB file and plots a 100x100 pixel subset of the DAPI image from the SingleTimePointImaging acquisition.

# %% [markdown]
# ## Examples of Common Analyses
#
# This section will provide examples of common analyses that might be relevant to the dataset's content.
#
# 1. Cell Counting: The DAPI-stained images can be used to count the number of cells in a given region.
# 2. Intensity Analysis: The intensity of the DAPI signal can be used to assess DNA content in the cells.
# 3. Permeabilization Analysis: YoPro-1 staining (if available) can measure membrane permeabilization induced by electroporation.