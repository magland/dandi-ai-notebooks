"""
This script examines the odor presentation timeline during experimental blocks.
It will visualize when different odors were presented during Block 1.
"""

import numpy as np
import pynwb
import lindi
import matplotlib.pyplot as plt

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get Block 1 time interval
block1 = nwb.intervals["Block 1"]
block1_start = block1["start_time"].data[0]
block1_end = block1["stop_time"].data[0]

print(f"Block 1 timing: Start={block1_start:.2f}s, End={block1_end:.2f}s, Duration={(block1_end-block1_start):.2f}s")

# Create a plot showing the odor presentation timeline for Block 1
fig, ax = plt.subplots(figsize=(12, 6))

odors = ["Odor A ON", "Odor B ON", "Odor C ON"]  # Block 1 has ABC
colors = ['red', 'blue', 'green']

# Get the first few presentations for each odor to analyze timing patterns
print("\nFirst 5 presentations of each odor in Block 1:")
for i, odor_name in enumerate(odors):
    odor = nwb.intervals[odor_name]
    start_times = odor["start_time"].data[:]
    stop_times = odor["stop_time"].data[:]
    
    # Filter to only show events in Block 1
    mask = (start_times >= block1_start) & (start_times <= block1_end)
    filtered_starts = start_times[mask]
    filtered_stops = stop_times[mask]
    
    # Print the first 5 presentations
    if len(filtered_starts) >= 5:
        print(f"\n{odor_name}:")
        for j in range(5):
            print(f"  Presentation {j+1}: Start={filtered_starts[j]:.2f}s, Duration={(filtered_stops[j]-filtered_starts[j]):.2f}s")
    
    # Plot all presentations in Block 1
    for start, stop in zip(filtered_starts, filtered_stops):
        ax.axvspan(start, stop, alpha=0.3, color=colors[i], label=f"{odor_name}" if start == filtered_starts[0] else "")
    
    # Calculate and print statistics for this odor
    durations = filtered_stops - filtered_starts
    intervals = np.diff(filtered_starts)
    
    print(f"\nStatistics for {odor_name} in Block 1:")
    print(f"  Number of presentations: {len(filtered_starts)}")
    print(f"  Average duration: {np.mean(durations):.2f}s")
    if len(intervals) > 0:
        print(f"  Average interval between presentations: {np.mean(intervals):.2f}s")

# Add block boundaries
ax.axvline(block1_start, color='black', linestyle='--', label="Block 1 Start")
ax.axvline(block1_end, color='black', linestyle=':', label="Block 1 End")

# Set time limits to show the whole Block 1
ax.set_xlim(block1_start-60, block1_start+300)  # Show start of Block 1 plus 5 minutes

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Odor Presentations')
ax.set_title('Odor Presentation Timeline for Start of Block 1')
plt.tight_layout()
plt.savefig('tmp_scripts/block1_odor_timeline.png')

# Create a wider view showing more of the block
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Plot all presentations in Block 1 (same as before)
for i, odor_name in enumerate(odors):
    odor = nwb.intervals[odor_name]
    start_times = odor["start_time"].data[:]
    stop_times = odor["stop_time"].data[:]
    
    # Filter to only show events in Block 1
    mask = (start_times >= block1_start) & (start_times <= block1_end)
    filtered_starts = start_times[mask]
    filtered_stops = stop_times[mask]
    
    for start, stop in zip(filtered_starts, filtered_stops):
        ax2.axvspan(start, stop, alpha=0.3, color=colors[i], label=f"{odor_name}" if start == filtered_starts[0] else "")

# Add block boundaries
ax2.axvline(block1_start, color='black', linestyle='--', label="Block 1 Start")
ax2.axvline(block1_end, color='black', linestyle=':', label="Block 1 End")

# Set time limits to show the whole Block 1
ax2.set_xlim(block1_start, block1_end)

handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Odor Presentations')
ax2.set_title('Complete Odor Presentation Timeline for Block 1')
plt.tight_layout()
plt.savefig('tmp_scripts/block1_full_timeline.png')