# Explore the units table in the NWB file
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

units = nwb.processing["ecephys"]["units"]

print("Column names:", units.colnames)

unit_ids = units["id"].data[:]
print("Number of units:", len(unit_ids))

# Print some statistics for each column
for col in units.colnames:
    try:
        data = units[col].data[:]
        print(f"Column: {col}")
        print(f"  dtype: {data.dtype}")
        if np.issubdtype(data.dtype, np.number):
            print(f"  min: {np.min(data)}")
            print(f"  max: {np.max(data)}")
            print(f"  mean: {np.mean(data)}")
            print(f"  std: {np.std(data)}")
        else:
            print(f"  Unique values: {np.unique(data)}")
    except Exception as e:
        print(f"  Could not get statistics: {e}")

# Plot histogram of spike times for a random unit
unit_index = np.random.randint(0, len(unit_ids))
spike_times = units["spike_times"][unit_index]
plt.figure()
plt.hist(spike_times, bins=50)
plt.xlabel("Spike times")
plt.ylabel("Count")
plt.title(f"Spike times for unit {unit_ids[unit_index]}")
plt.savefig("tmp_scripts/spike_times.png")
plt.close()