"""
Analyze behavioral data from the mental navigation task:
- Trial types distribution
- Success rates
- Response times distribution
- Distance between start and target landmarks
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trials data
trials = nwb.intervals["trials"]
trial_types = trials["trial_type"][:]
success = trials["succ"][:]
rt = trials["rt"][:]
start_pos = trials["curr"][:]
target_pos = trials["target"][:]
distances = np.abs(target_pos - start_pos)

# Plot 1: Trial types distribution
plt.figure(figsize=(10, 6))
trial_type_counts = np.bincount(trial_types.astype(int))[1:]
plt.bar(['NTS', 'Center Visible', 'MNAV'], trial_type_counts)
plt.title('Distribution of Trial Types')
plt.ylabel('Number of Trials')
plt.savefig('tmp_scripts/trial_types.png')
plt.close()

# Plot 2: Success rate by trial type
plt.figure(figsize=(10, 6))
success_by_type = [np.mean(success[trial_types == i]) for i in [1, 2, 3]]
plt.bar(['NTS', 'Center Visible', 'MNAV'], success_by_type)
plt.title('Success Rate by Trial Type')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.savefig('tmp_scripts/success_rates.png')
plt.close()

# Plot 3: Response time distribution
plt.figure(figsize=(10, 6))
rt_by_type = [rt[trial_types == i] for i in [1, 2, 3]]
plt.boxplot(rt_by_type, labels=['NTS', 'Center Visible', 'MNAV'])
plt.title('Response Times by Trial Type')
plt.xlabel('Trial Type')
plt.ylabel('Response Time (s)')
plt.savefig('tmp_scripts/response_times.png')
plt.close()

# Plot 4: Success rate vs distance
dist_bins = np.linspace(0, np.max(distances), 6)
dist_centers = (dist_bins[1:] + dist_bins[:-1]) / 2
success_by_dist = []
for i in range(len(dist_bins)-1):
    mask = (distances >= dist_bins[i]) & (distances < dist_bins[i+1])
    success_by_dist.append(np.mean(success[mask]))

plt.figure(figsize=(10, 6))
plt.plot(dist_centers, success_by_dist, 'o-')
plt.title('Success Rate vs Distance Between Landmarks')
plt.xlabel('Distance Between Start and Target')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.savefig('tmp_scripts/success_vs_distance.png')
plt.close()