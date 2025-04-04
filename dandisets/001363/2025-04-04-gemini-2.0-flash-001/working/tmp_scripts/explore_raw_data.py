# tmp_scripts/explore_raw_data.py
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

ElectricalSeries = nwb.acquisition["ElectricalSeries"]
data = ElectricalSeries.data
electrodes = ElectricalSeries.electrodes

electrode_ids = electrodes["id"].data[:]
electrode_locations = electrodes["location"].data[:]

# Plot the raw data from the first electrode
electrode_index = 0
electrode_id = electrode_ids[electrode_index]
electrode_location = electrode_locations[electrode_index]
raw_data = data[:1000, electrode_index]  # Load a subset of the data

plt.figure(figsize=(10, 5))
plt.plot(raw_data)
plt.xlabel("Time (samples)")
plt.ylabel("Voltage (uV)")
plt.title(f"Raw data from electrode {electrode_id} at {electrode_location}")
plt.savefig("tmp_scripts/plot_raw_electrode_1.png")
plt.close()

print(f"Electrode ID: {electrode_id}")
print(f"Electrode Location: {electrode_location}")