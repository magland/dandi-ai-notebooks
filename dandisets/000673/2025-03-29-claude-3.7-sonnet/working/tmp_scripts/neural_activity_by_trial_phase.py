"""
This script examines neural activity in relation to different phases of the trial
and compares activity between memory load conditions.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# Make sure output directory exists for plots
os.makedirs('tmp_scripts', exist_ok=True)

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/9fdbe18f-158f-47c5-ba67-4c56118d6cf5/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trial data
trials = nwb.intervals["trials"]
loads = trials["loads"][:]
probe_in_out = trials["probe_in_out"][:]
response_accuracy = trials["response_accuracy"][:]
enc1_times = trials["timestamps_Encoding1"][:]
enc1_end_times = trials["timestamps_Encoding1_end"][:]
maint_times = trials["timestamps_Maintenance"][:]
probe_times = trials["timestamps_Probe"][:]
response_times = trials["timestamps_Response"][:]

# Get unit data
units = nwb.units
unit_ids = units["id"]
electrodes = nwb.electrodes
electrode_locations = electrodes["location"].data[:]

print(f"Analyzing neural activity across {len(unit_ids)} units")
print(f"Number of trials: {len(loads)}")

# Function to get spikes in a window around an event
def get_spike_counts_around_event(spike_times, event_times, window_size=1.0, bin_width=0.05):
    """
    Get spike counts in time bins around events
    
    Parameters:
    -----------
    spike_times : array
        Array of spike times for a unit
    event_times : array
        Array of event times
    window_size : float
        Size of the window around the event (in seconds) - each side
    bin_width : float
        Width of each bin (in seconds)
        
    Returns:
    --------
    time_bins : array
        Array of time bins (centered at 0 = event time)
    spike_counts : array
        Array of spike counts in each bin, averaged across events
    """
    num_bins = int(2 * window_size / bin_width)
    time_bins = np.linspace(-window_size, window_size, num_bins+1)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    all_counts = np.zeros((len(event_times), num_bins))
    
    for i, event_time in enumerate(event_times):
        start_time = event_time - window_size
        end_time = event_time + window_size
        
        # Get spikes in the window
        window_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
        
        # Shift spikes to be relative to the event
        relative_spikes = window_spikes - event_time
        
        # Bin the spikes
        counts, _ = np.histogram(relative_spikes, bins=time_bins)
        all_counts[i, :] = counts
    
    # Average across events
    mean_counts = np.mean(all_counts, axis=0)
    return bin_centers, mean_counts / bin_width  # Convert to firing rate (spikes/second)

# Select a subset of units (choose 5 random units)
np.random.seed(42)
num_units_to_plot = min(5, len(unit_ids))
unit_indices = np.random.choice(len(unit_ids), num_units_to_plot, replace=False)

# Define event types and corresponding times
events = {
    'Encoding Start': enc1_times,
    'Maintenance Start': maint_times,
    'Probe': probe_times,
    'Response': response_times
}

# Extract unit information
unit_locations = []
for i in range(len(unit_ids)):
    electrode_idx = units["electrodes"].data[i]
    unit_locations.append(electrode_locations[electrode_idx])

print("\nUnit locations for the selected units:")
for idx in unit_indices:
    print(f"Unit {unit_ids[idx]}: {unit_locations[idx]}")

# Plot neural activity for each selected unit around different trial events
for idx in unit_indices:
    unit_spike_times = units["spike_times"][idx]
    unit_id = unit_ids[idx]
    unit_location = unit_locations[idx]
    cluster_id = units["clusterID_orig"].data[idx]
    
    print(f"\nAnalyzing Unit {unit_id} (Cluster ID: {cluster_id}) in {unit_location}")
    print(f"  Number of spikes: {len(unit_spike_times)}")
    
    plt.figure(figsize=(15, 10))
    
    # Plot for different trial events
    for i, (event_name, event_times) in enumerate(events.items()):
        plt.subplot(2, 2, i+1)
        
        # Split by memory load
        load1_event_times = event_times[loads == 1]
        load3_event_times = event_times[loads == 3]
        
        # Get spike counts for each load condition
        time_bins_load1, counts_load1 = get_spike_counts_around_event(unit_spike_times, load1_event_times)
        time_bins_load3, counts_load3 = get_spike_counts_around_event(unit_spike_times, load3_event_times)
        
        # Apply smoothing
        smooth_counts_load1 = gaussian_filter1d(counts_load1, sigma=2)
        smooth_counts_load3 = gaussian_filter1d(counts_load3, sigma=2)
        
        plt.plot(time_bins_load1, smooth_counts_load1, 'b-', label='Load 1')
        plt.plot(time_bins_load3, smooth_counts_load3, 'r-', label='Load 3')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Time from event (s)')
        plt.ylabel('Firing Rate (spikes/s)')
        plt.title(f'{event_name}')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.suptitle(f'Unit {unit_id} ({unit_location}) response to task events', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'tmp_scripts/unit_{unit_id}_event_responses.png')
    plt.close()

# Calculate average firing rates by memory load
print("\nCalculating average firing rates by memory load...")

def calculate_epoch_firing_rate(spike_times, start_times, end_times):
    """Calculate firing rates during specified epochs"""
    total_spikes = 0
    total_time = 0
    
    for start, end in zip(start_times, end_times):
        # Count spikes in this window
        epoch_spikes = spike_times[(spike_times >= start) & (spike_times <= end)]
        total_spikes += len(epoch_spikes)
        total_time += (end - start)
    
    if total_time > 0:
        return total_spikes / total_time
    else:
        return 0

# Get maintenance periods
maint_start_times = trials["timestamps_Maintenance"][:]
probe_times = trials["timestamps_Probe"][:]

# Create lists to store firing rates for different units and conditions
load1_indices = np.where(loads == 1)[0]
load3_indices = np.where(loads == 3)[0]

unit_maint_rates_load1 = np.zeros(len(unit_ids))
unit_maint_rates_load3 = np.zeros(len(unit_ids))

for i in range(len(unit_ids)):
    unit_spike_times = units["spike_times"][i]
    
    # Calculate maintenance period firing rates by load
    maint_rate_load1 = calculate_epoch_firing_rate(unit_spike_times, 
                                                 maint_start_times[load1_indices], 
                                                 probe_times[load1_indices])
    
    maint_rate_load3 = calculate_epoch_firing_rate(unit_spike_times, 
                                                 maint_start_times[load3_indices], 
                                                 probe_times[load3_indices])
    
    unit_maint_rates_load1[i] = maint_rate_load1
    unit_maint_rates_load3[i] = maint_rate_load3

# Calculate the modulation index (normalized difference in firing rates)
modulation_index = (unit_maint_rates_load3 - unit_maint_rates_load1) / (unit_maint_rates_load3 + unit_maint_rates_load1 + 1e-10)

# Plot modulation index by unit (change in firing rate between load conditions)
plt.figure(figsize=(12, 6))
plt.bar(range(len(unit_ids)), modulation_index)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Unit Index')
plt.ylabel('Modulation Index\n(Load 3 - Load 1) / (Load 3 + Load 1)')
plt.title('Change in Firing Rate During Maintenance Period by Memory Load')
plt.tight_layout()
plt.savefig('tmp_scripts/load_modulation_index.png')
plt.close()

# Plot averages across units by brain region
regions = np.unique(unit_locations)
region_mod_indices = {}

for region in regions:
    region_units = [i for i, loc in enumerate(unit_locations) if loc == region]
    region_mod = modulation_index[region_units]
    region_mod_indices[region] = region_mod

plt.figure(figsize=(10, 6))
box_data = [region_mod_indices[region] for region in regions]
plt.boxplot(box_data, labels=regions)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Brain Region')
plt.ylabel('Modulation Index\n(Load 3 - Load 1) / (Load 3 + Load 1)')
plt.title('Change in Firing Rate During Maintenance by Brain Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('tmp_scripts/region_load_modulation.png')
plt.close()

# Compare firing rates in different trial phases by load
trial_phases = [
    ('Encoding', enc1_times, maint_times),
    ('Maintenance', maint_times, probe_times),
    ('Response', probe_times, response_times)
]

# Average across units for each region
region_phase_rates_load1 = {region: np.zeros(len(trial_phases)) for region in regions}
region_phase_rates_load3 = {region: np.zeros(len(trial_phases)) for region in regions}

for region in regions:
    region_units = [i for i, loc in enumerate(unit_locations) if loc == region]
    
    for i, (phase_name, start_times, end_times) in enumerate(trial_phases):
        # Calculate average firing rate for this phase and region
        region_rates_load1 = []
        region_rates_load3 = []
        
        for unit_idx in region_units:
            unit_spike_times = units["spike_times"][unit_idx]
            
            # Calculate for load 1
            rate_load1 = calculate_epoch_firing_rate(unit_spike_times, 
                                                   start_times[load1_indices], 
                                                   end_times[load1_indices])
            region_rates_load1.append(rate_load1)
            
            # Calculate for load 3
            rate_load3 = calculate_epoch_firing_rate(unit_spike_times, 
                                                   start_times[load3_indices], 
                                                   end_times[load3_indices])
            region_rates_load3.append(rate_load3)
        
        region_phase_rates_load1[region][i] = np.mean(region_rates_load1)
        region_phase_rates_load3[region][i] = np.mean(region_rates_load3)

# Plot average firing rates by trial phase for each anatomical region
for region in regions:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(trial_phases))
    width = 0.35
    
    plt.bar(x - width/2, region_phase_rates_load1[region], width, label='Load 1', color='blue')
    plt.bar(x + width/2, region_phase_rates_load3[region], width, label='Load 3', color='red')
    
    plt.xlabel('Trial Phase')
    plt.ylabel('Average Firing Rate (spikes/s)')
    plt.title(f'Average Firing Rate by Trial Phase and Memory Load: {region}')
    plt.xticks(x, [phase[0] for phase in trial_phases])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'tmp_scripts/{region.replace(" ", "_")}_phase_firing_rates.png')
    plt.close()

print("Neural activity analysis complete. Plots saved to tmp_scripts directory.")