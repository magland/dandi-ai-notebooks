# Script to explore key aspects of the NWB file data
# Generates plots as PNG files for review

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import lindi

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Plot 1: Trial success rates by condition
trials = nwb.intervals["trials"]
trial_types = trials["trial_type"][:]
success = trials["succ"][:]

# Calculate success rates by trial type
unique_types = np.unique(trial_types)
success_rates = [np.mean(success[trial_types == t]) for t in unique_types]

type_labels = {
    1: 'Linear Map',
    2: 'Center Only',
    3: 'Fully Occluded'
}
current_labels = [type_labels.get(t, f'Type {t}') for t in unique_types]

plt.figure(figsize=(8, 5))
plt.bar(unique_types, success_rates)
plt.xlabel('Trial Type')
plt.ylabel('Success Rate')
plt.title('Success Rates by Trial Condition')
plt.xticks(unique_types, current_labels)
plt.savefig('tmp_scripts/trial_success.png')
plt.close()

# Plot 2: Example spike times from first 3 units
units = nwb.processing["ecephys"]["units"]
unit_ids = units["id"].data[:3]  # First 3 units

plt.figure(figsize=(10, 6))
for i, unit_id in enumerate(unit_ids):
    spike_times = units["spike_times"][i]
    plt.scatter(spike_times, [i] * len(spike_times), s=1, label=f'Unit {unit_id}')
plt.xlabel('Time (s)')
plt.ylabel('Unit')
plt.title('Example Spike Times')
plt.legend()
plt.savefig('tmp_scripts/spike_times.png')
plt.close()

# Plot 3: Response time distribution
rt = trials["rt"][:]
plt.figure(figsize=(8, 5))
plt.hist(rt, bins=50)
plt.xlabel('Response Time (s)')
plt.ylabel('Count')
plt.title('Distribution of Response Times')
plt.savefig('tmp_scripts/response_times.png')
plt.close()