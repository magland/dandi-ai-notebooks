"""
This script examines the structure of a single NWB file in more detail,
looking for task-related information and examining the neural activity patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file we've confirmed works
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic metadata
print("\nNWB File Metadata:")
print(f"Session description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Session start time: {nwb.session_start_time}")

# Examine available data interfaces
print("\nAvailable acquisition data:")
for name in nwb.acquisition:
    print(f"  - {name}")

print("\nAvailable processing modules:")
for module_name in nwb.processing:
    module = nwb.processing[module_name]
    print(f"  - {module_name}")
    for interface_name in module.data_interfaces:
        print(f"    - {interface_name}")

# Check if there's any behavioral data or task-related information
has_behavior = 'behavior' in nwb.processing
print(f"\nHas behavior module: {has_behavior}")

# Extract and plot some example activity data
print("\nExtracting calcium activity data...")

# Get fluorescence traces
ophys = nwb.processing["ophys"]
fluor = ophys["Fluorescence"]["RoiResponseSeries"]
print(f"Fluorescence data shape: {fluor.data.shape}")
print(f"Number of ROIs: {fluor.data.shape[1]}")
print(f"Sampling rate: {fluor.rate} Hz")
print(f"Duration: {fluor.data.shape[0] / fluor.rate:.2f} seconds")

# Get event amplitudes
event_amp = ophys["EventAmplitude"]
print(f"Event amplitude data shape: {event_amp.data.shape}")

# Calculate activity statistics for each ROI
print("\nCalculating activity statistics...")
n_rois = min(fluor.data.shape[1], 10)  # Limit to 10 ROIs for performance
roi_stats = []

for i in range(n_rois):
    # Get fluorescence trace for this ROI
    trace = fluor.data[:, i]
    
    # Calculate statistics
    mean_activity = np.mean(trace)
    max_activity = np.max(trace)
    std_activity = np.std(trace)
    
    # Get event amplitudes for this ROI
    events = event_amp.data[:, i]
    num_events = np.sum(events > 0)
    
    roi_stats.append({
        'roi_id': i,
        'mean_activity': mean_activity,
        'max_activity': max_activity,
        'std_activity': std_activity,
        'num_events': num_events
    })

# Plot activity statistics
print("Plotting activity statistics...")

# Plot mean activity for each ROI
plt.figure(figsize=(10, 6))
roi_ids = [stat['roi_id'] for stat in roi_stats]
mean_activities = [stat['mean_activity'] for stat in roi_stats]
std_activities = [stat['std_activity'] for stat in roi_stats]

plt.bar(roi_ids, mean_activities, yerr=std_activities, alpha=0.7)
plt.xlabel('ROI ID')
plt.ylabel('Mean Fluorescence')
plt.title('Mean Activity by ROI')
plt.grid(alpha=0.3)
plt.savefig('tmp_scripts/roi_mean_activity.png', dpi=300)
plt.close()

# Plot number of detected events per ROI
plt.figure(figsize=(10, 6))
num_events = [stat['num_events'] for stat in roi_stats]

plt.bar(roi_ids, num_events, alpha=0.7)
plt.xlabel('ROI ID')
plt.ylabel('Number of Events')
plt.title('Number of Detected Calcium Events by ROI')
plt.grid(alpha=0.3)
plt.savefig('tmp_scripts/roi_num_events.png', dpi=300)
plt.close()

print("\nScript completed successfully!")