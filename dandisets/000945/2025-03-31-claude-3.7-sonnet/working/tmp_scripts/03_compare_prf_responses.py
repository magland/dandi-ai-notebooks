"""
Exploratory script to compare neural responses to different Pulse Repetition Frequencies (PRFs).
This script will:
1. Load two NWB files with different PRFs (3000 Hz and 1500 Hz)
2. Compute PSTHs for both files
3. Compare response strength between the different PRFs
4. Visualize the differences in neural responses
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the two NWB files with different PRFs
print("Loading NWB files for different PRFs...")

# 3000 Hz PRF file
f_3000 = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb_3000 = pynwb.NWBHDF5IO(file=f_3000, mode='r').read()

# 1500 Hz PRF file
f_1500 = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/526c681d-0c50-44e1-92be-9c0134c71fd8/nwb.lindi.json")
nwb_1500 = pynwb.NWBHDF5IO(file=f_1500, mode='r').read()

# Print basic information
print(f"File 1: {nwb_3000.identifier} - PRF 3000 Hz")
print(f"File 2: {nwb_1500.identifier} - PRF 1500 Hz")
print(f"Subject: {nwb_3000.subject.subject_id}")
print(f"Institution: {nwb_3000.institution}")

# Define the function to compute PSTHs
def compute_unit_psth(units, unit_idx, trial_times, pre_time, post_time, bins):
    """Compute PSTH for a single unit across multiple trials."""
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
    firing_rate = mean_counts / (bins[1] - bins[0])
    
    return firing_rate

# Set up parameters for PSTH computation
pre_time = 0.3  # Time before stimulus onset (seconds)
post_time = 0.5  # Time after stimulus onset (seconds)
bin_size = 0.025  # 25 ms bins
bins = np.arange(-pre_time, post_time + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Get trials and units
trials_3000 = nwb_3000.intervals["trials"]
units_3000 = nwb_3000.units
trial_starts_3000 = trials_3000["start_time"][:]

trials_1500 = nwb_1500.intervals["trials"]
units_1500 = nwb_1500.units
trial_starts_1500 = trials_1500["start_time"][:]

# Sample a subset of trials (for faster computation)
np.random.seed(42)
n_trials = 50
sampled_trials_3000 = np.random.choice(trial_starts_3000, size=n_trials, replace=False)
sampled_trials_1500 = np.random.choice(trial_starts_1500, size=n_trials, replace=False)

# Get cell types
celltypes_3000 = units_3000["celltype_label"].data[:]
celltypes_1500 = units_1500["celltype_label"].data[:]

# Find units that are RSUs in both files
rsu_indices_3000 = np.where(celltypes_3000 == 1)[0]
rsu_indices_1500 = np.where(celltypes_1500 == 1)[0]

# Compute PSTHs for the first 5 RSU units in each file
n_units_to_analyze = 5
rsu_psths_3000 = []
rsu_psths_1500 = []

print("Computing PSTHs for RSU units...")
for i in range(min(n_units_to_analyze, len(rsu_indices_3000))):
    idx_3000 = rsu_indices_3000[i]
    idx_1500 = rsu_indices_1500[i]
    
    psth_3000 = compute_unit_psth(units_3000, idx_3000, sampled_trials_3000, pre_time, post_time, bins)
    psth_1500 = compute_unit_psth(units_1500, idx_1500, sampled_trials_1500, pre_time, post_time, bins)
    
    rsu_psths_3000.append(psth_3000)
    rsu_psths_1500.append(psth_1500)

# Compute mean PSTHs across units
mean_rsu_psth_3000 = np.mean(rsu_psths_3000, axis=0)
mean_rsu_psth_1500 = np.mean(rsu_psths_1500, axis=0)
sem_rsu_psth_3000 = np.std(rsu_psths_3000, axis=0) / np.sqrt(len(rsu_psths_3000))
sem_rsu_psth_1500 = np.std(rsu_psths_1500, axis=0) / np.sqrt(len(rsu_psths_1500))

# Find FSU units in both files
fsu_indices_3000 = np.where(celltypes_3000 == 2)[0]
fsu_indices_1500 = np.where(celltypes_1500 == 2)[0]

# Compute PSTHs for FSU units
fsu_psths_3000 = []
fsu_psths_1500 = []

print("Computing PSTHs for FSU units...")
for i in range(min(n_units_to_analyze, len(fsu_indices_3000))):
    idx_3000 = fsu_indices_3000[i]
    idx_1500 = fsu_indices_1500[i]
    
    psth_3000 = compute_unit_psth(units_3000, idx_3000, sampled_trials_3000, pre_time, post_time, bins)
    psth_1500 = compute_unit_psth(units_1500, idx_1500, sampled_trials_1500, pre_time, post_time, bins)
    
    fsu_psths_3000.append(psth_3000)
    fsu_psths_1500.append(psth_1500)

# Compute mean PSTHs for FSU units
mean_fsu_psth_3000 = np.mean(fsu_psths_3000, axis=0)
mean_fsu_psth_1500 = np.mean(fsu_psths_1500, axis=0)
sem_fsu_psth_3000 = np.std(fsu_psths_3000, axis=0) / np.sqrt(len(fsu_psths_3000))
sem_fsu_psth_1500 = np.std(fsu_psths_1500, axis=0) / np.sqrt(len(fsu_psths_1500))

# Plot mean PSTHs for both PRFs - RSU units
plt.figure(figsize=(10, 6))

# Plot RSU PSTHs for both PRFs
plt.subplot(2, 1, 1)
plt.plot(bin_centers, mean_rsu_psth_3000, 'b-', label='3000 Hz PRF')
plt.fill_between(bin_centers, 
                 mean_rsu_psth_3000 - sem_rsu_psth_3000,
                 mean_rsu_psth_3000 + sem_rsu_psth_3000,
                 color='b', alpha=0.2)

plt.plot(bin_centers, mean_rsu_psth_1500, 'g-', label='1500 Hz PRF')
plt.fill_between(bin_centers, 
                 mean_rsu_psth_1500 - sem_rsu_psth_1500,
                 mean_rsu_psth_1500 + sem_rsu_psth_1500,
                 color='g', alpha=0.2)

# Add vertical line for stimulus onset
plt.axvline(0, color='k', linestyle='--', label='Stimulus Onset')

plt.title('RSU Responses to Different PRFs')
plt.xlabel('Time relative to stimulus (s)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot FSU PSTHs for both PRFs
plt.subplot(2, 1, 2)
plt.plot(bin_centers, mean_fsu_psth_3000, 'r-', label='3000 Hz PRF')
plt.fill_between(bin_centers, 
                 mean_fsu_psth_3000 - sem_fsu_psth_3000,
                 mean_fsu_psth_3000 + sem_fsu_psth_3000,
                 color='r', alpha=0.2)

plt.plot(bin_centers, mean_fsu_psth_1500, 'orange', label='1500 Hz PRF')
plt.fill_between(bin_centers, 
                 mean_fsu_psth_1500 - sem_fsu_psth_1500,
                 mean_fsu_psth_1500 + sem_fsu_psth_1500,
                 color='orange', alpha=0.2)

# Add vertical line for stimulus onset
plt.axvline(0, color='k', linestyle='--', label='Stimulus Onset')

plt.title('FSU Responses to Different PRFs')
plt.xlabel('Time relative to stimulus (s)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prf_comparison_psth.png')
print("Saved PRF comparison PSTH to prf_comparison_psth.png")

# Calculate response indices (ratio of post-stim to pre-stim activity)
baseline_window = (-0.3, -0.05)  # 50-300 ms before stimulus
response_window = (0.05, 0.3)    # 50-300 ms after stimulus

# Function to calculate response index based on PSTH
def calculate_response_index(psth, bin_centers, baseline_window, response_window):
    baseline_mask = (bin_centers >= baseline_window[0]) & (bin_centers <= baseline_window[1])
    response_mask = (bin_centers >= response_window[0]) & (bin_centers <= response_window[1])
    
    baseline_activity = np.mean(psth[baseline_mask])
    response_activity = np.mean(psth[response_mask])
    
    # Avoid division by zero
    if baseline_activity == 0:
        return np.nan
    
    return response_activity / baseline_activity

# Calculate response indices for RSUs
rsu_response_idx_3000 = []
rsu_response_idx_1500 = []

for psth in rsu_psths_3000:
    idx = calculate_response_index(psth, bin_centers, baseline_window, response_window)
    rsu_response_idx_3000.append(idx)

for psth in rsu_psths_1500:
    idx = calculate_response_index(psth, bin_centers, baseline_window, response_window)
    rsu_response_idx_1500.append(idx)

# Calculate response indices for FSUs
fsu_response_idx_3000 = []
fsu_response_idx_1500 = []

for psth in fsu_psths_3000:
    idx = calculate_response_index(psth, bin_centers, baseline_window, response_window)
    fsu_response_idx_3000.append(idx)

for psth in fsu_psths_1500:
    idx = calculate_response_index(psth, bin_centers, baseline_window, response_window)
    fsu_response_idx_1500.append(idx)

# Create response index comparison plot
plt.figure(figsize=(12, 6))

# Create bar positions
x_pos = np.arange(len(rsu_response_idx_3000))
width = 0.35

# Plot RSU response indices
plt.subplot(1, 2, 1)
plt.bar(x_pos - width/2, rsu_response_idx_3000, width, color='blue', alpha=0.7, label='3000 Hz PRF')
plt.bar(x_pos + width/2, rsu_response_idx_1500, width, color='green', alpha=0.7, label='1500 Hz PRF')

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
plt.xlabel('Unit Number')
plt.ylabel('Response Index (Post/Pre Ratio)')
plt.title('RSU Response to Different PRFs')
plt.xticks(x_pos, [str(i) for i in range(len(rsu_response_idx_3000))])
plt.legend()

# Plot FSU response indices
plt.subplot(1, 2, 2)
plt.bar(x_pos - width/2, fsu_response_idx_3000, width, color='red', alpha=0.7, label='3000 Hz PRF')
plt.bar(x_pos + width/2, fsu_response_idx_1500, width, color='orange', alpha=0.7, label='1500 Hz PRF')

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
plt.xlabel('Unit Number')
plt.ylabel('Response Index (Post/Pre Ratio)')
plt.title('FSU Response to Different PRFs')
plt.xticks(x_pos, [str(i) for i in range(len(fsu_response_idx_3000))])
plt.legend()

plt.tight_layout()
plt.savefig('prf_comparison_response_index.png')
print("Saved PRF comparison response index to prf_comparison_response_index.png")

# Calculate mean response indices for each cell type and PRF
mean_rsu_idx_3000 = np.nanmean(rsu_response_idx_3000)
mean_rsu_idx_1500 = np.nanmean(rsu_response_idx_1500)
mean_fsu_idx_3000 = np.nanmean(fsu_response_idx_3000)
mean_fsu_idx_1500 = np.nanmean(fsu_response_idx_1500)

# Calculate standard error
sem_rsu_idx_3000 = np.nanstd(rsu_response_idx_3000) / np.sqrt(np.sum(~np.isnan(rsu_response_idx_3000)))
sem_rsu_idx_1500 = np.nanstd(rsu_response_idx_1500) / np.sqrt(np.sum(~np.isnan(rsu_response_idx_1500)))
sem_fsu_idx_3000 = np.nanstd(fsu_response_idx_3000) / np.sqrt(np.sum(~np.isnan(fsu_response_idx_3000)))
sem_fsu_idx_1500 = np.nanstd(fsu_response_idx_1500) / np.sqrt(np.sum(~np.isnan(fsu_response_idx_1500)))

# Create a summary bar plot
plt.figure(figsize=(8, 6))

x_labels = ['RSU', 'FSU']
x_pos = np.arange(len(x_labels))
width = 0.35

# Plot mean response indices
plt.bar(x_pos - width/2, [mean_rsu_idx_3000, mean_fsu_idx_3000], width, 
        yerr=[sem_rsu_idx_3000, sem_fsu_idx_3000],
        color=['blue', 'red'], alpha=0.7, label='3000 Hz PRF')

plt.bar(x_pos + width/2, [mean_rsu_idx_1500, mean_fsu_idx_1500], width,
        yerr=[sem_rsu_idx_1500, sem_fsu_idx_1500],
        color=['green', 'orange'], alpha=0.7, label='1500 Hz PRF')

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
plt.xlabel('Cell Type')
plt.ylabel('Mean Response Index (Post/Pre Ratio)')
plt.title('Effect of PRF on Neural Responses')
plt.xticks(x_pos, x_labels)
plt.legend()

plt.tight_layout()
plt.savefig('prf_summary.png')
print("Saved PRF response summary to prf_summary.png")

print("Analysis complete!")