"""
Analyze calcium activity traces from the fluorescence and event data.
This script will load the fluorescence and event amplitude data for ROIs
and visualize the activity patterns over time.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Load the NWB file using lindi
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Access the fluorescence data
fluor = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]
fluorescence_data = fluor.data[:]  # Load the entire dataset
sampling_rate = fluor.rate  # Hz

# Access the event amplitude data
events = nwb.processing["ophys"]["EventAmplitude"]
event_data = events.data[:]  # Load the entire dataset

print(f"Fluorescence data shape: {fluorescence_data.shape}")
print(f"Event data shape: {event_data.shape}")
print(f"Sampling rate: {sampling_rate} Hz")

# Calculate the timestamps (in seconds)
num_samples = fluorescence_data.shape[0]
timestamps = np.arange(num_samples) / sampling_rate
duration_min = timestamps[-1] / 60  # Convert to minutes
print(f"Recording duration: {duration_min:.2f} minutes")

# Select a subset of ROIs for visualization (to avoid overcrowding)
num_rois = fluorescence_data.shape[1]
roi_subset = np.linspace(0, num_rois-1, min(10, num_rois), dtype=int)  # Up to 10 ROIs evenly spaced

print(f"Total ROIs: {num_rois}")
print(f"Selected ROIs for visualization: {roi_subset}")

# Create a figure for the selected ROI traces
plt.figure(figsize=(14, 10))

# Plot fluorescence traces for the selected ROIs
for i, roi_id in enumerate(roi_subset):
    # Normalize the trace to better visualize patterns
    trace = fluorescence_data[:, roi_id]
    trace_norm = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
    
    # Offset the trace for visualization
    trace_offset = trace_norm + i
    
    # Plot the trace
    plt.plot(timestamps / 60, trace_offset, lw=1, label=f'ROI {roi_id}')

plt.xlabel('Time (minutes)')
plt.ylabel('Normalized Fluorescence (offset for clarity)')
plt.title('Fluorescence Traces for Selected ROIs')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tmp_scripts/fluorescence_traces.png', dpi=300)

# Create a figure to compare fluorescence and event data for a few selected ROIs
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Choose 3 ROIs that appear to have interesting activity patterns
interesting_rois = [roi_subset[0], roi_subset[len(roi_subset)//2], roi_subset[-1]]
print(f"Selected ROIs for detailed comparison: {interesting_rois}")

for i, roi_id in enumerate(interesting_rois):
    ax = axes[i]
    
    # Plot fluorescence trace
    fluor_trace = fluorescence_data[:, roi_id]
    ax.plot(timestamps / 60, fluor_trace, 'b-', lw=1, label='Fluorescence')
    
    # Plot event amplitudes
    event_trace = event_data[:, roi_id]
    ax.plot(timestamps / 60, event_trace, 'r-', lw=1, label='Event Amplitude')
    
    ax.set_ylabel(f'ROI {roi_id}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (minutes)')
plt.suptitle('Comparison of Fluorescence and Event Amplitude Traces')
plt.tight_layout()
plt.savefig('tmp_scripts/fluor_vs_events.png', dpi=300)

# Analyze activity patterns over time
# Create a raster plot of event activity
plt.figure(figsize=(14, 8))

# Create a binary event matrix (1 if event, 0 if no event)
# Use a threshold based on percentile of event amplitudes
event_threshold = np.percentile(event_data, 90)  # Top 10% are considered events
event_binary = event_data > event_threshold
print(f"Event threshold: {event_threshold:.4f}")
print(f"Number of detected events: {np.sum(event_binary)}")

# Create a raster plot
plt.imshow(event_binary.T, aspect='auto', cmap='binary', 
           extent=[0, duration_min, 0, num_rois])

plt.xlabel('Time (minutes)')
plt.ylabel('ROI ID')
plt.title('Raster Plot of Neural Activity (Event Detection)')
plt.colorbar(label='Event', ticks=[0, 1])
plt.grid(False)
plt.tight_layout()
plt.savefig('tmp_scripts/raster_plot.png', dpi=300)

# Calculate and plot cross-correlations between some ROIs
plt.figure(figsize=(12, 8))

# Choose a few ROIs for correlation analysis
corr_rois = interesting_rois
max_lag = int(10 * sampling_rate)  # 10 seconds max lag
lags = np.arange(-max_lag, max_lag + 1) / sampling_rate  # Convert to seconds

for i in range(len(corr_rois)):
    for j in range(i+1, len(corr_rois)):
        roi1_id = corr_rois[i]
        roi2_id = corr_rois[j]
        
        # Get the traces
        trace1 = fluorescence_data[:, roi1_id] - np.mean(fluorescence_data[:, roi1_id])
        trace2 = fluorescence_data[:, roi2_id] - np.mean(fluorescence_data[:, roi2_id])
        
        # Calculate cross-correlation
        xcorr = signal.correlate(trace1, trace2, mode='full')
        xcorr /= np.sqrt(np.sum(trace1**2) * np.sum(trace2**2))  # Normalize
        
        # Plot
        plt.plot(lags, xcorr[len(trace1)-max_lag-1:len(trace1)+max_lag], 
                 label=f'ROI {roi1_id} vs ROI {roi2_id}')

plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Lag (seconds)')
plt.ylabel('Correlation')
plt.title('Cross-correlation between ROI pairs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tmp_scripts/cross_correlation.png', dpi=300)

print("Calcium activity analysis complete.")