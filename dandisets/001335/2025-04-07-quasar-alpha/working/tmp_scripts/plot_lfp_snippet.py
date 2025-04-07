# This script plots a short snippet (2 seconds) of raw LFP data from 5 example electrodes.
# It generates a visualization of simultaneous activity traces for the notebook.
import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

LFP = nwb.processing["ecephys"]["LFP"]
rate = LFP.rate  # 2500 Hz
data = LFP.data

segment_duration_sec = 2
segment_samples = int(segment_duration_sec * rate)
start_idx = 0  # beginning of recording

# Select 5 electrode indices (0, 1, 2, 3, 4)
num_channels = data.shape[1]
channels_to_plot = np.arange(min(5, num_channels))

# Load data snippet [time x channels]
snippet = data[start_idx:start_idx+segment_samples, channels_to_plot]

time = np.arange(segment_samples) / rate

plt.figure(figsize=(10, 6))
offset = 0
for i, ch in enumerate(channels_to_plot):
    plt.plot(time, snippet[:, i] + offset, label=f'channel {ch}')
    offset += 500  # Offset for visualization

plt.xlabel('Time (s)')
plt.ylabel('Voltage + offset')
plt.title('LFP traces, first 2 seconds, 5 example electrodes')
plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/lfp_snippet.png")
plt.close()