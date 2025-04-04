# tmp_scripts/explore_trials.py
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load https://api.dandiarchive.org/api/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001363/assets/59d1acbb-5ad5-45f1-b211-c2e311801824/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

trials = nwb.intervals["trials"]
start_time = trials["start_time"][:]
stop_time = trials["stop_time"][:]
trial_ids = trials["id"][:]

# Create a plot of the trial start and stop times
plt.figure(figsize=(10, 5))
plt.plot(trial_ids, start_time, label="Start Time")
plt.plot(trial_ids, stop_time, label="Stop Time")
plt.xlabel("Trial ID")
plt.ylabel("Time (s)")
plt.title("Trial Start and Stop Times")
plt.legend()
plt.savefig("tmp_scripts/plot_trial_times.png")
plt.close()

print(f"Number of trials: {len(trial_ids)}")