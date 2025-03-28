"""
Script to analyze neural responses to different odor presentations in the dataset.
This script will:
1. Extract timing information for odor presentations
2. Analyze LFP activity during odor presentations
3. Analyze spike rates for neurons during different odor presentations
4. Create a raster plot for neural activity aligned to odor onset
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
print("NWB file loaded successfully!")

# Extract timing information for each experimental block
blocks = {}
for block_name in ["Block 1", "Block 2", "Block 3"]:
    block = nwb.intervals[block_name]
    start = block["start_time"].data[:][0]
    stop = block["stop_time"].data[:][0]
    blocks[block_name] = {"start": start, "stop": stop}
    print(f"{block_name}: {start:.2f}s to {stop:.2f}s")

# Extract odor presentation timing information
odor_intervals = {}
for odor_name in ["Odor A ON", "Odor B ON", "Odor C ON", "Odor D ON", "Odor E ON", "Odor F ON"]:
    odor_interval = nwb.intervals[odor_name]
    starts = odor_interval["start_time"].data[:]
    stops = odor_interval["stop_time"].data[:]
    durations = stops - starts
    odor_intervals[odor_name] = {"starts": starts, "stops": stops, "durations": durations}
    print(f"{odor_name}: {len(starts)} presentations, avg duration: {np.mean(durations):.2f}s")

# Function to determine which block an odor presentation belongs to
def get_block_for_time(time):
    for block_name, block_info in blocks.items():
        if block_info["start"] <= time < block_info["stop"]:
            return block_name
    return "None"  # If the time doesn't fall within any block

# Count odor presentations per block
odor_per_block = {}
for odor_name, intervals in odor_intervals.items():
    if odor_name not in odor_per_block:
        odor_per_block[odor_name] = {"Block 1": 0, "Block 2": 0, "Block 3": 0, "None": 0}
    
    for start_time in intervals["starts"]:
        block = get_block_for_time(start_time)
        odor_per_block[odor_name][block] += 1

# Plot odor presentations per block
odors = list(odor_per_block.keys())
block_names = ["Block 1", "Block 2", "Block 3"]
x = np.arange(len(odors))
width = 0.25

plt.figure(figsize=(12, 6))
for i, block in enumerate(block_names):
    counts = [odor_per_block[odor][block] for odor in odors]
    plt.bar(x + (i - 1) * width, counts, width, label=block)

plt.ylabel('Number of Presentations')
plt.title('Odor Presentations per Block')
plt.xticks(x, [odor.replace(" ON", "") for odor in odors])
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/odor_per_block.png')

# Get LFP data
lfp = nwb.processing["ecephys"]["LFP"]
lfp_rate = lfp.rate  # Sampling rate in Hz

# Select a subset of electrodes for analysis
electrode_indices = [0, 15, 30, 45, 60]
if len(electrode_indices) > lfp.data.shape[1]:
    electrode_indices = list(range(min(5, lfp.data.shape[1])))

# Function to extract LFP around an event
def get_lfp_around_event(event_time, before=0.5, after=2.5, electrode_idx=0):
    """
    Extract LFP data around an event.
    
    Parameters:
    - event_time: Time of the event in seconds
    - before: Time before event in seconds
    - after: Time after event in seconds
    - electrode_idx: Index of the electrode
    
    Returns:
    - times: Time array centered around the event
    - lfp_snippet: LFP data around the event
    """
    # Calculate sample indices
    start_idx = max(0, int((event_time - before) * lfp_rate))
    end_idx = min(lfp.data.shape[0], int((event_time + after) * lfp_rate))
    
    # Extract LFP data
    lfp_snippet = lfp.data[start_idx:end_idx, electrode_idx]
    times = np.arange(len(lfp_snippet)) / lfp_rate - before
    
    return times, lfp_snippet

# Analyze LFP responses to different odors
# Select one electrode for this analysis
electrode_idx = 15  # Changed from the earlier example

# For each odor, get average LFP response in Block 1
plt.figure(figsize=(15, 10))
odor_names = ["Odor A ON", "Odor B ON", "Odor C ON"]
colors = ['blue', 'orange', 'green']

for i, odor_name in enumerate(odor_names):
    # Get odor presentations in Block 1
    starts = odor_intervals[odor_name]["starts"]
    block1_starts = [t for t in starts if get_block_for_time(t) == "Block 1"]
    
    if len(block1_starts) > 0:
        # Limit to first 30 presentations for efficiency and to avoid timeout
        block1_starts = block1_starts[:30]
        
        # Get LFP data for each presentation
        all_lfp_data = []
        for start_time in block1_starts:
            times, lfp_data = get_lfp_around_event(start_time, before=0.5, after=2.5, electrode_idx=electrode_idx)
            all_lfp_data.append(lfp_data)
        
        # Calculate mean and std
        if all_lfp_data:
            all_lfp_data = np.vstack(all_lfp_data)
            mean_lfp = np.mean(all_lfp_data, axis=0)
            std_lfp = np.std(all_lfp_data, axis=0)
            
            # Plot mean and std
            plt.plot(times, mean_lfp, color=colors[i], label=f"{odor_name}")
            plt.fill_between(times, mean_lfp - std_lfp, mean_lfp + std_lfp, color=colors[i], alpha=0.3)

plt.axvline(x=0, color='black', linestyle='--', label='Odor Onset')
plt.axvline(x=2, color='black', linestyle=':', label='Approx. Odor Offset')
plt.xlabel('Time relative to Odor Onset (s)')
plt.ylabel('LFP Amplitude (µV)')
plt.title(f'Mean LFP Response to Different Odors in Block 1 (Electrode {electrode_idx})')
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('tmp_scripts/mean_lfp_response_block1.png')

# Now analyze spiking activity
units = nwb.units

# Select a few units for analysis based on high spike counts
spike_counts = []
for i in range(len(units["id"].data[:])):
    spike_times = units["spike_times"][i]
    spike_counts.append(len(spike_times))

# Find indices of top 5 units by spike count
top_unit_indices = np.argsort(spike_counts)[-5:]
print(f"\nSelected unit indices for spike analysis: {top_unit_indices}")

# Function to calculate firing rate around events
def calc_firing_rate(spike_times, event_times, before=0.5, after=2.5, bin_size=0.1):
    """
    Calculate firing rate around events.
    
    Parameters:
    - spike_times: Array of spike times
    - event_times: Array of event times
    - before: Time before event in seconds
    - after: Time after event in seconds
    - bin_size: Size of time bins in seconds
    
    Returns:
    - bin_centers: Centers of time bins
    - rate: Firing rate in each bin (Hz)
    """
    bins = np.arange(-before, after + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    count = np.zeros_like(bin_centers)
    
    for event_time in event_times:
        # Align spike times to event
        aligned_spikes = spike_times - event_time
        # Count spikes in each bin
        hist, _ = np.histogram(aligned_spikes, bins=bins)
        count += hist
    
    # Convert count to rate (Hz)
    rate = count / (len(event_times) * bin_size)
    return bin_centers, rate

# Analyze firing rate for each odor in Block 1
plt.figure(figsize=(15, 12))
gs = GridSpec(len(top_unit_indices), 1, figure=plt.gcf())

odor_names = ["Odor A ON", "Odor B ON", "Odor C ON"]
colors = ['blue', 'orange', 'green']

for i, unit_idx in enumerate(top_unit_indices):
    ax = plt.subplot(gs[i])
    
    unit_id = units["global_id"].data[unit_idx]
    spike_times = units["spike_times"][unit_idx]
    
    for j, odor_name in enumerate(odor_names):
        # Get odor presentations in Block 1
        starts = odor_intervals[odor_name]["starts"]
        block1_starts = [t for t in starts if get_block_for_time(t) == "Block 1"]
        
        if len(block1_starts) > 0:
            # Limit to first 30 presentations for efficiency
            block1_starts = block1_starts[:30]
            
            # Calculate firing rate
            bin_centers, rate = calc_firing_rate(spike_times, block1_starts, before=0.5, after=2.5, bin_size=0.1)
            ax.plot(bin_centers, rate, color=colors[j], label=f"{odor_name}" if i == 0 else "")
    
    ax.axvline(x=0, color='black', linestyle='--', label='Odor Onset' if i == 0 else "")
    ax.axvline(x=2, color='black', linestyle=':', label='Approx. Odor Offset' if i == 0 else "")
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Unit {unit_id}')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if i == len(top_unit_indices) - 1:
        ax.set_xlabel('Time relative to Odor Onset (s)')
    
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/firing_rates_block1.png')

# Create a raster plot for one unit around odor onset
selected_unit_idx = top_unit_indices[0]  # Use the unit with highest spike count
unit_id = units["global_id"].data[selected_unit_idx]
spike_times = units["spike_times"][selected_unit_idx]

plt.figure(figsize=(12, 8))

# Plot raster for Odor A, B, and C in Block 1
for i, odor_name in enumerate(["Odor A ON", "Odor B ON", "Odor C ON"]):
    # Get odor presentations in Block 1
    starts = odor_intervals[odor_name]["starts"]
    block1_starts = [t for t in starts if get_block_for_time(t) == "Block 1"]
    
    if len(block1_starts) > 0:
        # Limit to first 20 presentations for clarity
        block1_starts = block1_starts[:20]
        
        # For each presentation, plot spikes
        for j, start_time in enumerate(block1_starts):
            # Find spikes around this presentation
            mask = (spike_times >= start_time - 0.5) & (spike_times <= start_time + 2.5)
            presentation_spikes = spike_times[mask]
            
            # Align to odor onset
            aligned_spikes = presentation_spikes - start_time
            
            # Plot raster - create array of same y value for each spike
            y_values = np.ones_like(aligned_spikes) * (j + i * 25)
            plt.scatter(aligned_spikes, y_values, s=5, c=colors[i], marker='|')

# Add vertical lines for odor onset and offset
plt.axvline(x=0, color='black', linestyle='--', label='Odor Onset')
plt.axvline(x=2, color='black', linestyle=':', label='Approx. Odor Offset')

plt.xlabel('Time relative to Odor Onset (s)')
plt.ylabel('Trial Number')
plt.title(f'Spike Raster Plot for Unit {unit_id} during Odor Presentations in Block 1')
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('tmp_scripts/spike_raster_block1.png')

print("Analysis complete. Check the generated plots.")