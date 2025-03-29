"""
Script to analyze event-related activity in the calcium imaging data.
This script:
1. Detects significant calcium events more robustly
2. Analyzes event timing and synchronization
3. Visualizes the relationship between neurons
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import find_peaks
import seaborn as sns

# Set seaborn style for better visualizations
sns.set_theme()

# Load the NWB file
url = "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
print(f"Loading NWB file from {url}")
f = lindi.LindiH5pyFile.from_lindi_file(url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the fluorescence data
ophys = nwb.processing["ophys"]
fluorescence = ophys["Fluorescence"]["RoiResponseSeries"]
print(f"Fluorescence data shape: {fluorescence.data.shape}")

# To avoid memory issues, we'll use a subset of the data (first 3000 timepoints)
max_timepoints = 3000
if fluorescence.data.shape[0] > max_timepoints:
    print(f"Using first {max_timepoints} timepoints for analysis")
    fluor_data = fluorescence.data[:max_timepoints, :]
else:
    fluor_data = fluorescence.data[:]

n_rois = fluor_data.shape[1]
time_vector = np.arange(fluor_data.shape[0]) / fluorescence.rate

print("\n1. Detecting calcium events with better method...")
# Detect events using z-scoring and peak finding
z_scored_data = np.zeros_like(fluor_data)
for i in range(n_rois):
    z_scored_data[:, i] = zscore(fluor_data[:, i])

# Find peaks for each ROI with height threshold of 2 standard deviations
events = np.zeros_like(fluor_data, dtype=bool)
event_peak_indices = []
for i in range(n_rois):
    # Find peaks in z-scored data
    peaks, _ = find_peaks(z_scored_data[:, i], height=2.0)
    events[peaks, i] = True
    event_peak_indices.append(peaks)

# Count events per ROI
event_counts = np.sum(events, axis=0)

# Plot improved event counts
plt.figure(figsize=(10, 6))
plt.bar(np.arange(n_rois), event_counts)
plt.xlabel('ROI #')
plt.ylabel('Number of Calcium Events')
plt.title('Improved Calcium Event Count per ROI')
plt.tight_layout()
plt.savefig('improved_calcium_event_counts.png', dpi=150)
plt.close()

print("\n2. Analyzing event timing and synchronization...")
# Create a raster plot of calcium events
plt.figure(figsize=(12, 8))
for i in range(n_rois):
    event_times = time_vector[events[:, i]]
    plt.scatter(event_times, np.ones_like(event_times) * i, marker='|', s=100, color='k')

plt.xlabel('Time (s)')
plt.ylabel('ROI #')
plt.title('Calcium Event Raster Plot')
plt.tight_layout()
plt.savefig('calcium_event_raster.png', dpi=150)
plt.close()

# Calculate event synchronization
# For each timepoint, count how many neurons are active simultaneously
event_synchronization = np.sum(events, axis=1)

# Plot event synchronization over time
plt.figure(figsize=(12, 6))
plt.plot(time_vector, event_synchronization)
plt.xlabel('Time (s)')
plt.ylabel('Number of Simultaneously Active ROIs')
plt.title('Neuronal Synchronization')
plt.tight_layout()
plt.savefig('neuronal_synchronization.png', dpi=150)
plt.close()

print("\n3. Visualizing active periods...")
# Find timepoints with high synchronization (many neurons active)
highly_sync_threshold = np.percentile(event_synchronization, 95)  # Top 5% of synchronized activity
highly_sync_timepoints = event_synchronization >= highly_sync_threshold
highly_sync_indices = np.where(highly_sync_timepoints)[0]

if len(highly_sync_indices) > 0:
    print(f"Found {len(highly_sync_indices)} timepoints with high synchronization")

    # Choose a few timepoints with high synchronization to examine
    num_examples = min(3, len(highly_sync_indices))
    selected_indices = highly_sync_indices[:num_examples]

    # Show neural activity around these timepoints
    window_size = int(2 * fluorescence.rate)  # 2 seconds before and after
    
    # Create a figure with subplots for each example
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4*num_examples), sharex=True)
    if num_examples == 1:
        axes = [axes]  # Make it iterable for the loop

    for i, idx in enumerate(selected_indices):
        # Define window boundaries
        start_idx = max(0, idx - window_size)
        end_idx = min(len(time_vector), idx + window_size)
        
        # Adjust time to be centered around the event
        window_time = time_vector[start_idx:end_idx] - time_vector[idx]
        
        # Calculate mean activity for each ROI in the window
        for roi in range(n_rois):
            axes[i].plot(window_time, z_scored_data[start_idx:end_idx, roi], alpha=0.5)
        
        # Add vertical line at event time
        axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[i].set_title(f'Activity around high synchronization at t={time_vector[idx]:.2f}s')
        axes[i].set_ylabel('Z-scored activity')
    
    plt.xlabel('Time relative to synchronization (s)')
    plt.tight_layout()
    plt.savefig('activity_around_synchronization.png', dpi=150)
    plt.close()
else:
    print("No timepoints with high synchronization found")

print("\n4. Analyzing neuronal coactivation patterns...")
# Create co-activation matrix (how often pairs of neurons are active together)
coactivation = np.zeros((n_rois, n_rois))
for i in range(n_rois):
    for j in range(n_rois):
        if i <= j:
            # Count how many times both neurons are active at the same time
            coactivation[i, j] = np.sum(np.logical_and(events[:, i], events[:, j]))
            # Make the matrix symmetric
            coactivation[j, i] = coactivation[i, j]

# Plot coactivation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(coactivation, cmap='viridis', 
            xticklabels=np.arange(n_rois), yticklabels=np.arange(n_rois))
plt.title('ROI Co-activation Matrix')
plt.xlabel('ROI #')
plt.ylabel('ROI #')
plt.tight_layout()
plt.savefig('roi_coactivation_matrix.png', dpi=150)
plt.close()

print("Event-related activity analysis complete.")