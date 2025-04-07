# This script loads the NWB data and plots a small snippet around a randomly selected trial onset.
# It avoids loading excessive data by loading only a short window (~200 ms).
# The plot is saved as 'tmp_scripts/trial_snippet.png'.

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import random

url = "https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

electrical_series = nwb.acquisition['ElectricalSeries']
data = electrical_series.data
rate = electrical_series.rate

trials = nwb.intervals['trials']
starts = trials['start_time'][:]
selected_trial = random.choice(starts)

window = 0.1  # seconds before and after trial
start_index = int(max((selected_trial - window) * rate, 0))
end_index = int(min((selected_trial + window) * rate, data.shape[0]))

snippet = data[start_index:end_index, :]
time = np.arange(snippet.shape[0]) / rate + (start_index / rate - selected_trial)

plt.figure(figsize=(12, 8))
for ch in range(snippet.shape[1]):
    plt.plot(time * 1000, snippet[:, ch] * 1e6 + ch * 100, label=f'Ch {ch}')  # microvolt scaling with offset
plt.xlabel('Time (ms)')
plt.ylabel('Microvolts + offset')
plt.title('Signal snippet around a random trial onset')
plt.tight_layout()
plt.savefig('tmp_scripts/trial_snippet.png')
plt.close()
print("Saved plot to tmp_scripts/trial_snippet.png")