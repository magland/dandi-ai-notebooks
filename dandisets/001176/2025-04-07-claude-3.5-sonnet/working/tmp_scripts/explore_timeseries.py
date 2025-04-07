"""
This script creates time series plots of:
1. Fluorescence signal from ROIs
2. Pupil radius
3. Treadmill velocity

This will help us understand the temporal dynamics of neural activity and behavior.
"""

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/b22180d0-41dc-4091-a334-2e5bd4b5c548/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the fluorescence data - selecting first 1000 timepoints as example
fluorescence = nwb.processing['ophys']['Fluorescence']['RoiResponseSeries1']
fluor_data = fluorescence.data[:1000]
fluor_times = fluorescence.timestamps[:1000]

# Get pupil radius data - selecting corresponding timepoints
pupil = nwb.acquisition['PupilTracking']['pupil_raw_radius']
pupil_data = pupil.data[:2000]  # More timepoints since different sampling rate
pupil_times = pupil.timestamps[:2000]

# Get treadmill velocity - selecting corresponding timepoints
velocity = nwb.acquisition['treadmill_velocity']
vel_data = velocity.data[:10000]  # More timepoints since different sampling rate
vel_times = velocity.timestamps[:10000]

# Create figure
plt.figure(figsize=(12, 8))

# Plot fluorescence
plt.subplot(311)
plt.plot(fluor_times, fluor_data)
plt.title('Fluorescence Signal')
plt.ylabel('Fluorescence (a.u.)')

# Plot pupil radius
plt.subplot(312)
plt.plot(pupil_times, pupil_data)
plt.title('Pupil Radius')
plt.ylabel('Radius (pixels)')

# Plot velocity
plt.subplot(313)
plt.plot(vel_times, vel_data)
plt.title('Treadmill Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (a.u.)')

plt.tight_layout()
plt.savefig('tmp_scripts/timeseries.png', dpi=300, bbox_inches='tight')
plt.close()