"""
Exploratory script to analyze neural responses to ultrasound stimulation.
This script will:
1. Compute peri-stimulus time histograms (PSTHs) for different cell types
2. Compare responses between Regular Spiking Units (RSU) and Fast Spiking Units (FSU)
3. Visualize how neural firing changes in response to the stimulus
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
import os

# Load the NWB file using lindi
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("Analyzing neural responses to ultrasound stimulation...")
print(f"NWB File: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")

# Get trials and units data
trials = nwb.intervals["trials"]
units = nwb.units

# Extract trials start times
trial_start_times = trials["start_time"][:]
print(f"Number of trials: {len(trial_start_times)}")

# Extract unit information
unit_ids = units["id"].data
celltypes = units["celltype_label"].data[:]
print(f"Number of units: {len(unit_ids)}")
print(f"Cell types: RSU = {np.sum(celltypes == 1)}, FSU = {np.sum(celltypes == 2)}")

# Define window for PSTH calculation
pre_time = 1.0  # 1 second before stimulus
post_time = 2.0  # 2 seconds after stimulus
bin_size = 0.02  # 20 ms bins
bins = np.arange(-pre_time, post_time + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Separate units by cell type
rsu_indices = np.where(celltypes == 1)[0]
fsu_indices = np.where(celltypes == 2)[0]

print(f"Computing PSTHs for {len(rsu_indices)} RSU units and {len(fsu_indices)} FSU units...")

# Function to compute PSTH for a single unit across all trials
def compute_unit_psth(unit_idx, trial_times, pre_time, post_time, bins):
    spike_times = units["spike_times"][unit_idx]
    all_trial_counts = []
    
    # Loop through each trial
    for trial_start in trial_times:
        # Find spikes in window around this trial
        window_start = trial_start - pre_time
        window_end = trial_start + post_time
        
        # Get spikes in this window
        mask = (spike_times >= window_start) & (spike_times <= window_end)
        spikes_in_window = spike_times[mask]
        
        # Convert to time relative to stimulus onset
        relative_times = spikes_in_window - trial_start
        
        # Bin the spikes
        counts, _ = np.histogram(relative_times, bins=bins)
        all_trial_counts.append(counts)
    
    # Average across trials and convert to firing rate (spikes/second)
    mean_counts = np.mean(all_trial_counts, axis=0)
    firing_rate = mean_counts / bin_size
    
    return firing_rate

# Compute PSTH for each unit (sampling a subset of trials for speed)
# Using 100 random trials instead of all 500 for faster computation
np.random.seed(42)
sampled_trials = np.random.choice(trial_start_times, size=100, replace=False)

# Number of units to analyze per type (to keep processing time reasonable)
n_units_per_type = 10

# Compute PSTHs for sample units
rsu_psths = [compute_unit_psth(idx, sampled_trials, pre_time, post_time, bins) 
             for idx in rsu_indices[:n_units_per_type]]
fsu_psths = [compute_unit_psth(idx, sampled_trials, pre_time, post_time, bins) 
             for idx in fsu_indices[:n_units_per_type]]

# Calculate average PSTH for each cell type
mean_rsu_psth = np.mean(rsu_psths, axis=0)
mean_fsu_psth = np.mean(fsu_psths, axis=0)
sem_rsu_psth = np.std(rsu_psths, axis=0) / np.sqrt(len(rsu_psths))
sem_fsu_psth = np.std(fsu_psths, axis=0) / np.sqrt(len(fsu_psths))

# Plot average PSTHs for each cell type
plt.figure(figsize=(10, 6))

# Plot RSU average
plt.plot(bin_centers, mean_rsu_psth, 'b-', label='RSU (n={})'.format(len(rsu_psths)))
plt.fill_between(bin_centers, 
                 mean_rsu_psth - sem_rsu_psth, 
                 mean_rsu_psth + sem_rsu_psth, 
                 color='b', alpha=0.2)

# Plot FSU average
plt.plot(bin_centers, mean_fsu_psth, 'r-', label='FSU (n={})'.format(len(fsu_psths)))
plt.fill_between(bin_centers, 
                 mean_fsu_psth - sem_fsu_psth, 
                 mean_fsu_psth + sem_fsu_psth, 
                 color='r', alpha=0.2)

# Add vertical line for stimulus onset
plt.axvline(0, color='k', linestyle='--', label='Stimulus Onset')

plt.xlabel('Time relative to stimulus (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Average Neuronal Responses to Ultrasound Stimulation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('average_psth.png')
print("Saved average PSTH to average_psth.png")

# Plot heatmap of responses for individual units
plt.figure(figsize=(12, 10))

# Prepare data for heatmap
all_psths = np.vstack(rsu_psths + fsu_psths)
unit_labels = np.array(['RSU']*len(rsu_psths) + ['FSU']*len(fsu_psths))
unit_indices = list(rsu_indices[:n_units_per_type]) + list(fsu_indices[:n_units_per_type])

# Sort units by their peak response time
peak_times = np.argmax(all_psths[:, bin_centers > 0], axis=1)
sorted_indices = np.argsort(peak_times)

sorted_psths = all_psths[sorted_indices]
sorted_labels = unit_labels[sorted_indices]
sorted_unit_indices = [unit_indices[i] for i in sorted_indices]

# Normalize each unit's PSTH by its max value for better visualization
normalized_psths = np.zeros_like(sorted_psths)
for i, psth in enumerate(sorted_psths):
    # Add small value to avoid division by zero
    normalized_psths[i] = psth / (np.max(psth) + 1e-6)

# Create heatmap
plt.imshow(normalized_psths, aspect='auto', 
           extent=[bin_centers[0], bin_centers[-1], len(sorted_psths)-0.5, -0.5],
           cmap='viridis')

# Add vertical line for stimulus onset
plt.axvline(0, color='w', linestyle='--', linewidth=2)

# Add y-axis labels for unit types
for i, (label, idx) in enumerate(zip(sorted_labels, sorted_unit_indices)):
    color = 'white'
    plt.text(bin_centers[0] - 0.1, i, f"{label} (Unit {idx})", 
             va='center', ha='right', fontsize=8, color=color)

plt.colorbar(label='Normalized Firing Rate')
plt.xlabel('Time relative to stimulus (s)')
plt.ylabel('Neuron')
plt.title('Individual Neuronal Responses to Ultrasound Stimulation')
plt.tight_layout()
plt.savefig('psth_heatmap.png')
print("Saved PSTH heatmap to psth_heatmap.png")

# Calculate response indices (ratio of post-stim to pre-stim activity)
stim_window = 0.5  # 500 ms after stimulus
baseline_mask = (bin_centers >= -0.5) & (bin_centers < 0)
response_mask = (bin_centers >= 0) & (bin_centers < stim_window)

# Calculate response indices for RSU and FSU units
rsu_response_indices = []
fsu_response_indices = []

# Process units
for psth, unit_idx in zip(rsu_psths, rsu_indices[:n_units_per_type]):
    baseline = np.mean(psth[baseline_mask])
    response = np.mean(psth[response_mask])
    # Check if baseline is zero
    if baseline > 0:
        response_idx = response / baseline
    else:
        response_idx = np.nan
    rsu_response_indices.append((unit_idx, response_idx))

for psth, unit_idx in zip(fsu_psths, fsu_indices[:n_units_per_type]):
    baseline = np.mean(psth[baseline_mask])
    response = np.mean(psth[response_mask])
    # Check if baseline is zero
    if baseline > 0:
        response_idx = response / baseline
    else:
        response_idx = np.nan
    fsu_response_indices.append((unit_idx, response_idx))

# Create bar plot of response indices
plt.figure(figsize=(12, 6))

# Set up positions for bars
rsu_positions = np.arange(len(rsu_response_indices))
fsu_positions = np.arange(len(fsu_response_indices)) + len(rsu_response_indices) + 1

# Extract response indices
rsu_values = [r[1] for r in rsu_response_indices]
fsu_values = [r[1] for r in fsu_response_indices]

# Plot bars
rsu_bars = plt.bar(rsu_positions, rsu_values, color='blue', alpha=0.7)
fsu_bars = plt.bar(fsu_positions, fsu_values, color='red', alpha=0.7)

# Add reference line at y=1 (no change from baseline)
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No Change')

# Add labels and title
plt.xlabel('Unit')
plt.ylabel('Response Index (Post/Pre Ratio)')
plt.title('Ultrasound Response Strength by Unit Type')
plt.xticks(np.concatenate([rsu_positions, fsu_positions]), 
           [f"RSU {idx}" for idx, _ in rsu_response_indices] + 
           [f"FSU {idx}" for idx, _ in fsu_response_indices], 
           rotation=90)

plt.legend([rsu_bars[0], fsu_bars[0]], ['RSU', 'FSU'])
plt.tight_layout()
plt.savefig('response_index.png')
print("Saved response index plot to response_index.png")