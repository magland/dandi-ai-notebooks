"""
This script analyzes neural responses to focused ultrasound stimulation.
It creates peri-stimulus time histograms (PSTHs) to visualize how neurons
respond to the stimulation events.
"""
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
print("File loaded successfully")

# Get trial start times (stimulation onset)
trials = nwb.intervals["trials"]
stim_times = trials['start_time'][:]
print(f"Analyzing {len(stim_times)} stimulation trials")

# Get unit data
units = nwb.units
num_units = len(units['id'].data)
print(f"File contains {num_units} units")

# Parameters for PSTH
window_size = 2.0  # seconds around stimulation
bin_size = 0.05    # 50 ms bins
time_bins = np.arange(-window_size, window_size + bin_size, bin_size)
bin_centers = time_bins[:-1] + bin_size/2

# Information about cell types
cell_types = units['celltype_label'].data[:]
rsu_indices = np.where(cell_types == 1.0)[0]
fsu_indices = np.where(cell_types == 2.0)[0]
print(f"Number of RSUs: {len(rsu_indices)}")
print(f"Number of FSUs: {len(fsu_indices)}")

# Sample a few units from each type for analysis
num_samples = 5
rsu_samples = np.random.choice(rsu_indices, size=min(num_samples, len(rsu_indices)), replace=False)
fsu_samples = np.random.choice(fsu_indices, size=min(num_samples, len(fsu_indices)), replace=False)

# Get a random subset of trials to speed up processing
trial_subset = np.random.choice(len(stim_times), size=min(100, len(stim_times)), replace=False)
stim_times_subset = stim_times[trial_subset]

# Function to compute PSTH
def compute_psth(spike_times, event_times, window_size=2.0, bin_size=0.05):
    time_bins = np.arange(-window_size, window_size + bin_size, bin_size)
    psth = np.zeros(len(time_bins) - 1)
    
    # Loop through each stimulus event
    for event_time in event_times:
        # Find spikes within the window around this event
        window_spikes = spike_times[(spike_times >= event_time - window_size) & 
                                  (spike_times <= event_time + window_size)]
        
        # Convert spike times to times relative to event
        relative_times = window_spikes - event_time
        
        # Bin the spikes
        hist, _ = np.histogram(relative_times, bins=time_bins)
        psth += hist
    
    # Normalize by number of trials and bin size to get firing rate in Hz
    psth = psth / (len(event_times) * bin_size)
    return psth

# Compute and plot PSTHs for sample RSUs
plt.figure(figsize=(12, 8))
for i, unit_idx in enumerate(rsu_samples):
    # Get spike times for this unit
    spike_times = units['spike_times'][unit_idx][:]
    
    # Compute PSTH
    psth = compute_psth(spike_times, stim_times_subset, window_size, bin_size)
    
    # Plot
    plt.subplot(len(rsu_samples), 1, i+1)
    plt.bar(bin_centers, psth, width=bin_size, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.axvspan(0, 0.067, alpha=0.2, color='red')  # Highlight 67ms stimulation duration
    plt.ylabel(f"Unit {units['id'].data[unit_idx]}\nFiring Rate (Hz)")
    
    if i == 0:
        plt.title("RSU Responses to Focused Ultrasound Stimulation")
    
    if i == len(rsu_samples) - 1:
        plt.xlabel("Time relative to stimulation onset (s)")

plt.tight_layout()
plt.savefig('tmp_scripts/rsu_responses.png')

# Compute and plot PSTHs for sample FSUs
plt.figure(figsize=(12, 8))
for i, unit_idx in enumerate(fsu_samples):
    # Get spike times for this unit
    spike_times = units['spike_times'][unit_idx][:]
    
    # Compute PSTH
    psth = compute_psth(spike_times, stim_times_subset, window_size, bin_size)
    
    # Plot
    plt.subplot(len(fsu_samples), 1, i+1)
    plt.bar(bin_centers, psth, width=bin_size, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.axvspan(0, 0.067, alpha=0.2, color='red')  # Highlight 67ms stimulation duration
    plt.ylabel(f"Unit {units['id'].data[unit_idx]}\nFiring Rate (Hz)")
    
    if i == 0:
        plt.title("FSU Responses to Focused Ultrasound Stimulation")
    
    if i == len(fsu_samples) - 1:
        plt.xlabel("Time relative to stimulation onset (s)")

plt.tight_layout()
plt.savefig('tmp_scripts/fsu_responses.png')

# Calculate average response across all RSUs and FSUs
print("Calculating average response for all units...")

# Function to compute average PSTH for a set of units
def compute_avg_psth(unit_indices, stim_times):
    all_psths = []
    for unit_idx in unit_indices:
        spike_times = units['spike_times'][unit_idx][:]
        psth = compute_psth(spike_times, stim_times, window_size, bin_size)
        all_psths.append(psth)
    return np.mean(all_psths, axis=0) if all_psths else np.zeros(len(time_bins)-1)

# Use a smaller subset for all units to avoid timeout
smaller_subset = np.random.choice(len(stim_times), size=min(50, len(stim_times)), replace=False)
smaller_stim_subset = stim_times[smaller_subset]

# Sample fewer units for the average to speed up processing
rsu_avg_sample = np.random.choice(rsu_indices, size=min(10, len(rsu_indices)), replace=False)
fsu_avg_sample = np.random.choice(fsu_indices, size=min(10, len(fsu_indices)), replace=False)

# Compute average responses
avg_rsu_psth = compute_avg_psth(rsu_avg_sample, smaller_stim_subset)
avg_fsu_psth = compute_avg_psth(fsu_avg_sample, smaller_stim_subset)

# Plot average responses
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, avg_rsu_psth, width=bin_size, alpha=0.7, label='RSU (n={})'.format(len(rsu_avg_sample)))
plt.bar(bin_centers, avg_fsu_psth, width=bin_size, alpha=0.7, label='FSU (n={})'.format(len(fsu_avg_sample)))
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.axvspan(0, 0.067, alpha=0.2, color='red')  # Highlight 67ms stimulation duration
plt.ylabel("Average Firing Rate (Hz)")
plt.xlabel("Time relative to stimulation onset (s)")
plt.title("Average Neural Responses to Focused Ultrasound Stimulation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tmp_scripts/average_responses.png')

print("Analysis complete - plots saved to tmp_scripts directory")