"""
Script to analyze neural activity patterns in the calcium imaging data.
This script:
1. Computes correlations between ROIs to identify functional networks
2. Detects calcium transients to quantify activity
3. Creates heatmaps of neural activity
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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

print("\n1. Computing correlation matrix between ROIs...")
# Compute correlation matrix between ROIs
corr_matrix = np.zeros((n_rois, n_rois))
for i in range(n_rois):
    for j in range(n_rois):
        if i <= j:  # Only compute upper triangle (matrix is symmetric)
            corr, _ = pearsonr(fluor_data[:, i], fluor_data[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Fill in lower triangle

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=np.arange(n_rois), yticklabels=np.arange(n_rois))
plt.title('Correlation Matrix Between ROIs')
plt.xlabel('ROI #')
plt.ylabel('ROI #')
plt.tight_layout()
plt.savefig('roi_correlation_matrix.png', dpi=150)
plt.close()

print("\n2. Detecting calcium events and analyzing activity...")
# Simple calcium event detection (threshold-based)
# Typically, more sophisticated algorithms would be used
def detect_events(trace, threshold_std=2):
    """Basic event detection based on threshold crossing"""
    baseline = np.median(trace)
    threshold = baseline + threshold_std * np.std(trace)
    events = trace > threshold
    return events

# Detect events for all ROIs
events = np.zeros_like(fluor_data, dtype=bool)
for i in range(n_rois):
    events[:, i] = detect_events(fluor_data[:, i])

# Count events per ROI
event_counts = np.sum(events, axis=0)

# Plot event counts
plt.figure(figsize=(10, 6))
plt.bar(np.arange(n_rois), event_counts)
plt.xlabel('ROI #')
plt.ylabel('Number of Calcium Events')
plt.title('Calcium Event Count per ROI')
plt.tight_layout()
plt.savefig('calcium_event_counts.png', dpi=150)
plt.close()

# Create activity heatmap
plt.figure(figsize=(12, 8))
# Sort ROIs by their total activity
roi_order = np.argsort(-np.sum(fluor_data, axis=0))
sorted_data = fluor_data[:, roi_order]

# Create heatmap with seaborn
ax = sns.heatmap(sorted_data.T, cmap='viridis', 
                 xticklabels=np.arange(0, len(time_vector), 500))
ax.set_xlabel('Time (samples)')
ax.set_ylabel('ROI (sorted by activity)')
ax.set_title('Neural Activity Heatmap')
plt.tight_layout()
plt.savefig('neural_activity_heatmap.png', dpi=150)
plt.close()

# Plot temporal profile of neural population
plt.figure(figsize=(12, 6))
mean_activity = np.mean(fluor_data, axis=1)
plt.plot(time_vector, mean_activity)
plt.xlabel('Time (s)')
plt.ylabel('Mean Fluorescence (a.u.)')
plt.title('Population Average Activity')
plt.tight_layout()
plt.savefig('population_average_activity.png', dpi=150)
plt.close()

print("Neural activity analysis complete.")