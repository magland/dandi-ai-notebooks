# %%
# This script loads the NWB file and attempts to plot all image masks superimposed on each other

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the NWB file
file_path = "https://lindi.neurosift.org/dandi/dandisets/001276/assets/95141d7a-82aa-4552-940a-1438a430a0d7/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(file_path)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print the contents of the SingleTimePointImaging to see what the image masks are called
print(nwb.acquisition["SingleTimePointImaging"])