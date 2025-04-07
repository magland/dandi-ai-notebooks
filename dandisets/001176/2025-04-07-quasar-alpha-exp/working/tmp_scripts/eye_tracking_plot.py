# This script loads eye tracking and pupil size data from NWB file and plots:
# - Pupil radius over time
# - X and Y eye positions over time
# Plot is saved to tmp_scripts/pupil_eye_tracking.png

import matplotlib.pyplot as plt
import lindi
import pynwb
import numpy as np

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/be84b6ff-7016-4ed8-af63-aa0e07c02530/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwbfile = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Extract datasets
pupil_radius = nwbfile.acquisition['PupilTracking']['pupil_raw_radius']
eye_position = nwbfile.acquisition['EyeTracking']['eye_position']

pupil_radius_data = pupil_radius.data[:]
eye_position_data = eye_position.data[:]

plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(pupil_radius_data)
plt.ylabel("Pupil Radius (a.u.)")
plt.title("Pupil radius over time")

plt.subplot(3,1,2)
plt.plot(eye_position_data[:,0])
plt.ylabel("Eye X-pos (px units)")
plt.title("Eye X Position")

plt.subplot(3,1,3)
plt.plot(eye_position_data[:,1])
plt.xlabel("Frame")
plt.ylabel("Eye Y-pos (px units)")
plt.title("Eye Y Position")

plt.tight_layout()
plt.savefig("tmp_scripts/pupil_eye_tracking.png")