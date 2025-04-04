# tmp_scripts/explore_psd.py
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

ElectricalSeries = nwb.acquisition["ElectricalSeries"]
data = ElectricalSeries.data
rate = ElectricalSeries.rate
electrodes = ElectricalSeries.electrodes

electrode_ids = electrodes["id"].data[:]
electrode_locations = electrodes["location"].data[:]

# Plot the power spectral density of the signal from the first electrode
electrode_index = 0
electrode_id = electrode_ids[electrode_index]
electrode_location = electrode_locations[electrode_index]
raw_data = data[:10000, electrode_index]  # Load a subset of the data

# Calculate the power spectral density
frequencies, power_spectrum = plt.psd(raw_data, Fs=rate, NFFT=2048)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (V^2/Hz)")
plt.title(f"Power Spectral Density from electrode {electrode_id} at {electrode_location}")
plt.savefig("tmp_scripts/plot_psd_electrode_1.png")
plt.close()

print(f"Electrode ID: {electrode_id}")
print(f"Electrode Location: {electrode_location}")