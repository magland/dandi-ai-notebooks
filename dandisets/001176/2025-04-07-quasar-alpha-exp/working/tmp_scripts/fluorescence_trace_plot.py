# This script loads and plots calcium fluorescence trace (RoiResponseSeries1) from the NWB file.
# The plot shows the single ROI's activity over time to inspect calcium dynamics.
# Output saved to tmp_scripts/fluorescence_trace.png

import matplotlib.pyplot as plt
import lindi
import pynwb
import numpy as np

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/be84b6ff-7016-4ed8-af63-aa0e07c02530/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwbfile = pynwb.NWBHDF5IO(file=f, mode='r').read()

roi_response = nwbfile.processing['ophys']['Fluorescence']['RoiResponseSeries1']
fluorescence_data = roi_response.data[:]
timestamps = roi_response.timestamps[:]

plt.figure(figsize=(10, 4))
plt.plot(timestamps, fluorescence_data)
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence (a.u.)")
plt.title("Calcium Fluorescence Trace (ROI 1)")
plt.tight_layout()
plt.savefig("tmp_scripts/fluorescence_trace.png")