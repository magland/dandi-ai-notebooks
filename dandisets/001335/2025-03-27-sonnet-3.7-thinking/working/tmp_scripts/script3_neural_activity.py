"""
This script analyzes neural activity (spike times) in relation to odor presentations.
It will look at how neurons respond during odor stimulation.
"""

import numpy as np
import pynwb
import lindi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get Odor A presentation times within Block 1
block1 = nwb.intervals["Block 1"]
block1_start = block1["start_time"].data[0]
block1_end = block1["stop_time"].data[0]

odorA = nwb.intervals["Odor A ON"]
odorA_starts = odorA["start_time"].data[:]
odorA_stops = odorA["stop_time"].data[:]
mask_odorA = (odorA_starts >= block1_start) & (odorA_starts <= block1_end)
odorA_starts_block1 = odorA_starts[mask_odorA]
odorA_stops_block1 = odorA_stops[mask_odorA]

# Collect neural data for analysis
units = nwb.units
unit_ids = units["id"].data[:]
n_units = len(unit_ids)

# Let's select a subset of units (10) to analyze
N_SAMPLE_UNITS = 10
np.random.seed(42)  # For reproducibility
sampled_indices = np.random.choice(n_units, size=N_SAMPLE_UNITS, replace=False)

# Print information about the sampled units
print("Selected Units for Analysis:")
for i, idx in enumerate(sampled_indices):
    unit_id = unit_ids[idx]
    unit_depth = units["depth"].data[idx]
    unit_hemisphere = units["hemisphere"].data[idx]
    n_spikes = len(units["spike_times"][idx])
    print(f"Unit {i+1}: ID={unit_id}, Depth={unit_depth:.2f} μm, Hemisphere={unit_hemisphere}, Spike Count={n_spikes}")

# Analyze neural responses to Odor A
# We'll look at a window around each odor presentation: 1s before to 3s after stimulus onset
pre_window = 1.0  # seconds before stimulus
post_window = 3.0  # seconds after stimulus (covers stimulus duration)

# Function to count spikes in a time window for a given unit
def count_spikes_in_window(spike_times, start_time, end_time):
    return np.sum((spike_times >= start_time) & (spike_times <= end_time))

# Calculate firing rates before and during odor presentation
print("\nOdor A Response Analysis:")
response_ratios = np.zeros(N_SAMPLE_UNITS)

for i, idx in enumerate(sampled_indices):
    spike_times = units["spike_times"][idx]
    
    # Count spikes in windows before and during odor presentation
    pre_counts = []
    during_counts = []
    
    for start, stop in zip(odorA_starts_block1[:20], odorA_stops_block1[:20]):  # Analyze first 20 presentations
        pre_start = start - pre_window
        pre_end = start
        
        n_pre = count_spikes_in_window(spike_times, pre_start, pre_end)
        n_during = count_spikes_in_window(spike_times, start, stop)
        
        pre_counts.append(n_pre)
        during_counts.append(n_during)
    
    # Calculate mean spike counts and firing rates
    mean_pre = np.mean(pre_counts)
    mean_during = np.mean(during_counts)
    
    # Calculate firing rates (spikes per second)
    pre_rate = mean_pre / pre_window
    during_rate = mean_during / (odorA_stops_block1[0] - odorA_starts_block1[0])
    
    # Calculate response ratio (during/pre)
    ratio = during_rate / pre_rate if pre_rate > 0 else float('inf')
    response_ratios[i] = ratio
    
    print(f"Unit {i+1}: Pre-odor rate={pre_rate:.2f} Hz, During-odor rate={during_rate:.2f} Hz, Ratio={ratio:.2f}")

# Create a raster plot for a responsive unit
# Find the unit with the highest response ratio
most_responsive_idx = np.argmax(response_ratios) if not np.any(np.isinf(response_ratios)) else 0
unit_idx = sampled_indices[most_responsive_idx]
unit_spike_times = units["spike_times"][unit_idx]
unit_id = unit_ids[unit_idx]

print(f"\nCreating raster plot for most responsive unit: Unit index {unit_idx}, ID={unit_id}")

# Create a figure for the raster plot
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

# Raster plot
ax1 = fig.add_subplot(gs[0])

# Show spikes for each odor presentation
for trial, (start, stop) in enumerate(zip(odorA_starts_block1[:15], odorA_stops_block1[:15])):  # First 15 trials
    # Window around odor presentation
    window_start = start - pre_window
    window_end = stop + 1.0  # 1 second after odor offset
    
    # Find spikes in this window
    mask = (unit_spike_times >= window_start) & (unit_spike_times <= window_end)
    spikes_in_window = unit_spike_times[mask]
    
    # Plot spikes relative to odor onset
    relative_times = spikes_in_window - start
    ax1.scatter(relative_times, np.ones_like(relative_times) * trial, color='black', s=2)
    
    # Mark odor presentation
    ax1.axvspan(0, stop-start, alpha=0.2, color='red', ymin=trial/15, ymax=(trial+1)/15)

ax1.axvline(0, linestyle='--', color='blue', label='Odor Onset')
ax1.set_xlabel('Time relative to odor onset (s)')
ax1.set_ylabel('Trial')
ax1.set_title(f'Unit {unit_id} Spike Raster Plot Relative to Odor A Onset')
ax1.legend()

# PSTH (Peri-Stimulus Time Histogram)
ax2 = fig.add_subplot(gs[1])

# Combine spikes from all trials
all_relative_times = []
for start in odorA_starts_block1[:20]:  # First 20 trials
    window_start = start - pre_window
    window_end = start + post_window
    
    mask = (unit_spike_times >= window_start) & (unit_spike_times <= window_end)
    spikes_in_window = unit_spike_times[mask]
    
    relative_times = spikes_in_window - start
    all_relative_times.extend(relative_times)

# Create histogram
bins = np.arange(-pre_window, post_window, 0.1)  # 100ms bins
ax2.hist(all_relative_times, bins=bins, alpha=0.7, color='green')
ax2.axvline(0, linestyle='--', color='blue', label='Odor Onset')
ax2.axvspan(0, 2, alpha=0.2, color='red', label='Odor On')  # Assuming 2s odor presentation
ax2.set_xlabel('Time relative to odor onset (s)')
ax2.set_ylabel('Spike Count')
ax2.set_title('Peri-Stimulus Time Histogram (PSTH)')
ax2.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/unit_response_to_odorA.png')

# Create a figure showing response ratios for all sampled units
plt.figure(figsize=(10, 6))
bar_colors = ['green' if ratio > 1.2 else 'blue' if ratio > 0.8 else 'red' for ratio in response_ratios]
bars = plt.bar(range(N_SAMPLE_UNITS), response_ratios, color=bar_colors)

plt.axhline(1.0, linestyle='--', color='black', alpha=0.7)
plt.xlabel('Unit Index')
plt.ylabel('Response Ratio (During Odor / Pre-Odor)')
plt.title('Odor A Response Magnitude for Sample Units')
plt.xticks(range(N_SAMPLE_UNITS), [f"Unit {i+1}" for i in range(N_SAMPLE_UNITS)])
plt.tight_layout()
plt.savefig('tmp_scripts/odorA_response_ratios.png')

# Now let's examine the LFP data around odor presentations
print("\nExtracting LFP data around Odor A presentations...")

# Get LFP data 
LFP = nwb.processing["ecephys"]["LFP"]
lfp_data = LFP.data  # This is a large dataset, so we need to extract only what we need
lfp_rate = LFP.rate  # Sampling rate in Hz

# Select a single channel for visualization
# Get electrode information
electrodes = nwb.electrodes
electrode_ids = electrodes['id'].data[:]
electrode_locations = electrodes['location'].data[:]

# Print some electrode information
print("\nElectrode information:")
for i in range(min(5, len(electrode_ids))):
    print(f"Electrode {i+1}: ID={electrode_ids[i]}, Location={electrode_locations[i]}")

# Select a single channel (use the first electrode)
channel_idx = 0
print(f"Using channel index {channel_idx} for LFP analysis")

# Calculate time points for a short segment around an odor presentation
# Let's look at the 5th odor presentation to ensure it's well within the recording
odor_idx = 4
odor_start = odorA_starts_block1[odor_idx]
odor_stop = odorA_stops_block1[odor_idx]

# Define a window around the odor presentation
pre_time = 2.0  # 2 seconds before odor onset
post_time = 4.0  # 4 seconds after odor onset

window_start = odor_start - pre_time
window_end = odor_start + post_time

# Convert times to sample indices
start_sample = int((window_start - LFP.starting_time) * lfp_rate)
end_sample = int((window_end - LFP.starting_time) * lfp_rate)

# Make sure we don't go out of bounds
start_sample = max(0, start_sample)
end_sample = min(lfp_data.shape[0], end_sample)

print(f"Extracting LFP samples from {start_sample} to {end_sample} (window around odor presentation)")

# Extract the LFP data for this time window and channel
lfp_segment = lfp_data[start_sample:end_sample, channel_idx]

# Create a time vector (in seconds relative to odor onset)
time_vector = np.arange(len(lfp_segment)) / lfp_rate + (window_start - odor_start)

# Plot the LFP data
plt.figure(figsize=(12, 6))
plt.plot(time_vector, lfp_segment)
plt.axvline(0, color='blue', linestyle='--', label='Odor Onset')
plt.axvspan(0, odor_stop - odor_start, color='red', alpha=0.2, label='Odor On')
plt.xlabel('Time relative to odor onset (s)')
plt.ylabel('LFP Amplitude (µV)')
plt.title(f'LFP Activity around Odor A Presentation (Channel {channel_idx})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tmp_scripts/lfp_during_odorA.png')