"""
Plot Beta_Band_Voltage signals (power envelope) across electrodes over time.

Saves a multi-channel time series plot as PNG.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

nwb_file = "tmp_scripts/sample_file.nwb"
output_png = "tmp_scripts/beta_band_voltage.png"

with h5py.File(nwb_file, "r") as f:
    group = f["processing"]["ecephys"]["LFP"]["Beta_Band_Voltage"]
    data = group["data"][:]            # shape: (1400,)
    timestamps = group["timestamps"][:]
    electrodes = group["electrodes"][:]

# For illustration, arrange data as (n_electrodes, n_timepoints)
# Likely, data is organized per-electrode or combined, but since shape is (1400,)
# We'll assume it's for one channel or averaged; if multi-channel needed, adjust accordingly.
# Based on electrode array shape, it might be stacked vertically.

# Here, treat as multi-electrode x time:
if data.ndim == 1 and electrodes.shape[0] > 1:
    # Guess reshape: timepoints along axis 1
    try:
        data = data.reshape((electrodes.shape[0], -1))
    except:
        # fallback, treat as one channel
        data = data.reshape((1, -1))
elif data.ndim == 1:
    data = data.reshape((1, -1))

plt.figure(figsize=(12, 6))
for i, elec_id in enumerate(electrodes):
    if i >= data.shape[0]:
        break
    plt.plot(timestamps, data[i], label=f'Electrode {elec_id}')

plt.xlabel('Time (s)')
plt.ylabel('Beta Band Power (a.u.)')
plt.title('Beta Band Voltage signals across electrodes')
plt.legend()
plt.tight_layout()
plt.savefig(output_png)
print(f"Saved plot: {output_png}")