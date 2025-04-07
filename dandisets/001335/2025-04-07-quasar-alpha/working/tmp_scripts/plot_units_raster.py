# This script loads spike times from a few example units and generates raster plots
# over a short time window (~10 seconds) for inclusion in the notebook.
import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

units = nwb.units
unit_ids = units['id'].data[:]
n_units = len(unit_ids)

example_unit_idxs = np.arange(min(3, n_units))
time_window = [0, 10]  # seconds

plt.figure(figsize=(10, 4))
for idx, unit_idx in enumerate(example_unit_idxs):
    stimes = units['spike_times'][unit_idx]
    stimes_in_window = stimes[(stimes >= time_window[0]) & (stimes <= time_window[1])]
    plt.scatter(stimes_in_window, idx * np.ones_like(stimes_in_window), s=5, label=f'Unit {unit_ids[unit_idx]}')

plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.title('Spike times for example units (first 10 seconds)')
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/example_units_raster.png')
plt.close()