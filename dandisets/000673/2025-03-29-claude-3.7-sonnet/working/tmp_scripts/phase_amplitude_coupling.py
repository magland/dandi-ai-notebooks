"""
This script explores phase-amplitude coupling (PAC) in the hippocampal neurons.
It analyzes how theta phase modulates gamma amplitude in relation to working memory load.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import os

# Make sure output directory exists for plots
os.makedirs('tmp_scripts', exist_ok=True)

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/9fdbe18f-158f-47c5-ba67-4c56118d6cf5/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trial data
trials = nwb.intervals["trials"]
loads = trials["loads"][:]
maint_start_times = trials["timestamps_Maintenance"][:]
probe_times = trials["timestamps_Probe"][:]
load1_indices = np.where(loads == 1)[0]
load3_indices = np.where(loads == 3)[0]

# Get units and electrode information
units = nwb.units
unit_ids = units["id"]
electrodes = nwb.electrodes
electrode_locations = electrodes["location"].data[:]

print("Analyzing phase-amplitude coupling in hippocampal neurons")
print(f"Number of trials with load 1: {len(load1_indices)}")
print(f"Number of trials with load 3: {len(load3_indices)}")

# Create a function to extract spike times for specified trials
def get_trial_spikes(unit_idx, trial_indices, start_times, end_times):
    
    unit_spike_times = units["spike_times"][unit_idx]
    trial_spikes = []
    
    for i in trial_indices:
        start_time = start_times[i]
        end_time = end_times[i]
        mask = (unit_spike_times >= start_time) & (unit_spike_times <= end_time)
        spikes_in_trial = unit_spike_times[mask]
        normalized_spikes = (spikes_in_trial - start_time) / (end_time - start_time)
        trial_spikes.append(normalized_spikes)
    
    return trial_spikes

# Filter functions for extracting different frequency bands
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Function to compute modulation index
def compute_modulation_index(phase, amplitude, n_bins=18):
    """
    Compute modulation index (MI) to quantify the phase-amplitude coupling
    
    Parameters:
    -----------
    phase : array
        Phase values in radians (-π to π)
    amplitude : array
        Amplitude values
    n_bins : int
        Number of phase bins
        
    Returns:
    --------
    MI : float
        Modulation index value
    mean_amplitude : array
        Mean amplitude in each phase bin
    bin_centers : array
        Centers of phase bins in radians
    """
    # Create bins from -pi to pi
    phase_bins = np.linspace(-np.pi, np.pi, n_bins+1)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    
    # Compute mean amplitude in each phase bin
    bin_indices = np.digitize(phase, phase_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins-1)  # Ensure indices are valid
    
    mean_amplitude = np.zeros(n_bins)
    for i in range(n_bins):
        if np.sum(bin_indices == i) > 0:
            mean_amplitude[i] = np.mean(amplitude[bin_indices == i])
    
    # Normalize mean amplitude (sum to 1)
    if np.sum(mean_amplitude) > 0:
        mean_amplitude = mean_amplitude / np.sum(mean_amplitude)
        
    # Compute Kullback-Leibler divergence from uniform distribution
    uniform_dist = np.ones(n_bins) / n_bins
    KL = np.sum(mean_amplitude * np.log(mean_amplitude / uniform_dist + 1e-10))
    
    # Modulation index = normalized KL divergence
    MI = KL / np.log(n_bins)
    
    return MI, mean_amplitude, bin_centers

# Select hippocampal units only
hippocampal_units = []
for i in range(len(unit_ids)):
    electrode_idx = units["electrodes"].data[i]
    if "hippocampus" in electrode_locations[electrode_idx]:
        hippocampal_units.append(i)

print(f"Number of hippocampal units: {len(hippocampal_units)}")

# Sample rate information - assuming 1000 Hz (typical for these recordings)
sampling_rate = 1000.0  # Hz

# Define the frequency bands of interest
theta_band = (4, 8)     # 4-8 Hz (theta)
gamma_band = (30, 80)   # 30-80 Hz (gamma)

# Process a subset of trials to analyze PAC
# We'll focus on the maintenance period where working memory is actively engaged
n_trials_to_analyze = 10

# Get a few hippocampal units to analyze
n_units_to_analyze = min(3, len(hippocampal_units))
units_to_analyze = hippocampal_units[:n_units_to_analyze]

# Analyze modulation index for each unit and memory load
modulation_indices_load1 = np.zeros(len(units_to_analyze))
modulation_indices_load3 = np.zeros(len(units_to_analyze))

phase_amp_histograms_load1 = []
phase_amp_histograms_load3 = []

for unit_idx_idx, unit_idx in enumerate(units_to_analyze):
    unit_id = unit_ids[unit_idx]
    electrode_idx = units["electrodes"].data[unit_idx]
    brain_region = electrode_locations[electrode_idx]
    
    print(f"\nAnalyzing Unit {unit_id} in {brain_region}")
    
    # Get spikes during maintenance periods for each memory load
    load1_trials_to_analyze = load1_indices[:n_trials_to_analyze]
    load3_trials_to_analyze = load3_indices[:n_trials_to_analyze]
    
    load1_spikes = get_trial_spikes(unit_idx, load1_trials_to_analyze, 
                                  maint_start_times, probe_times)
    load3_spikes = get_trial_spikes(unit_idx, load3_trials_to_analyze, 
                                  maint_start_times, probe_times)
    
    # Now we'll create a pseudo-LFP from the spike train to analyze PAC
    # This is a simplified approach since we don't have direct LFP recordings
    
    # Convert spikes to a time series for each load condition
    temps_load1 = np.linspace(0, 1, int(sampling_rate))  # 1 second normalized time
    temps_load3 = np.linspace(0, 1, int(sampling_rate))  # 1 second normalized time
    
    # Create pseudo-signals from spike density
    signal_load1 = np.zeros(len(temps_load1))
    signal_load3 = np.zeros(len(temps_load3))
    
    for trial_spikes in load1_spikes:
        for spike_time in trial_spikes:
            if spike_time < 1.0:  # Only use spikes within our normalized window
                idx = int(spike_time * sampling_rate)
                if 0 <= idx < len(signal_load1):
                    # Add a kernel around each spike
                    window_size = 50  # samples, 50ms at 1000Hz
                    for i in range(max(0, idx - window_size), min(len(signal_load1), idx + window_size + 1)):
                        signal_load1[i] += np.exp(-0.5 * ((i - idx) / (window_size/5))**2)
    
    for trial_spikes in load3_spikes:
        for spike_time in trial_spikes:
            if spike_time < 1.0:  # Only use spikes within our normalized window
                idx = int(spike_time * sampling_rate)
                if 0 <= idx < len(signal_load3):
                    window_size = 50  # samples
                    for i in range(max(0, idx - window_size), min(len(signal_load3), idx + window_size + 1)):
                        signal_load3[i] += np.exp(-0.5 * ((i - idx) / (window_size/5))**2)
    
    # Filter the signals in theta and gamma bands
    theta_load1 = apply_bandpass(signal_load1, theta_band[0], theta_band[1], sampling_rate)
    gamma_load1 = apply_bandpass(signal_load1, gamma_band[0], gamma_band[1], sampling_rate)
    
    theta_load3 = apply_bandpass(signal_load3, theta_band[0], theta_band[1], sampling_rate)
    gamma_load3 = apply_bandpass(signal_load3, gamma_band[0], gamma_band[1], sampling_rate)
    
    # Extract phase of theta and amplitude of gamma using Hilbert transform
    theta_phase_load1 = np.angle(hilbert(theta_load1))
    gamma_amp_load1 = np.abs(hilbert(gamma_load1))
    
    theta_phase_load3 = np.angle(hilbert(theta_load3))
    gamma_amp_load3 = np.abs(hilbert(gamma_load3))
    
    # Compute modulation indices
    MI_load1, mean_amp_load1, bin_centers = compute_modulation_index(theta_phase_load1, gamma_amp_load1)
    MI_load3, mean_amp_load3, _ = compute_modulation_index(theta_phase_load3, gamma_amp_load3)
    
    modulation_indices_load1[unit_idx_idx] = MI_load1
    modulation_indices_load3[unit_idx_idx] = MI_load3
    
    phase_amp_histograms_load1.append(mean_amp_load1)
    phase_amp_histograms_load3.append(mean_amp_load3)
    
    print(f"  Modulation Index (Load 1): {MI_load1:.4f}")
    print(f"  Modulation Index (Load 3): {MI_load3:.4f}")
    
    # Plot PAC results for this unit
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.plot(temps_load1, signal_load1, 'k-', alpha=0.5)
    plt.title(f'Raw Signal - Load 1')
    plt.xlabel('Normalized Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(222)
    plt.plot(temps_load3, signal_load3, 'k-', alpha=0.5)
    plt.title(f'Raw Signal - Load 3')
    plt.xlabel('Normalized Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(223)
    plt.bar(bin_centers, mean_amp_load1, width=2*np.pi/len(bin_centers), 
            alpha=0.7, color='blue', label=f'MI = {MI_load1:.4f}')
    plt.xlabel('Theta Phase (rad)')
    plt.ylabel('Normalized Gamma Amplitude')
    plt.title('Theta-Gamma PAC - Load 1')
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.legend()
    
    plt.subplot(224)
    plt.bar(bin_centers, mean_amp_load3, width=2*np.pi/len(bin_centers), 
            alpha=0.7, color='red', label=f'MI = {MI_load3:.4f}')
    plt.xlabel('Theta Phase (rad)')
    plt.ylabel('Normalized Gamma Amplitude')
    plt.title('Theta-Gamma PAC - Load 3')
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               ['-π', '-π/2', '0', 'π/2', 'π'])
    plt.legend()
    
    plt.suptitle(f'Phase-Amplitude Coupling: Unit {unit_id} ({brain_region})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'tmp_scripts/PAC_unit_{unit_id}.png')
    plt.close()

# Plot comparison of modulation indices by memory load
plt.figure(figsize=(10, 6))
ind = np.arange(len(units_to_analyze))
width = 0.35

plt.bar(ind - width/2, modulation_indices_load1, width, label='Load 1', color='blue')
plt.bar(ind + width/2, modulation_indices_load3, width, label='Load 3', color='red')

unit_labels = [f'Unit {unit_ids[idx]}' for idx in units_to_analyze]
plt.xlabel('Unit')
plt.ylabel('Modulation Index')
plt.title('PAC Modulation Index Comparison by Memory Load')
plt.xticks(ind, unit_labels)
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/modulation_index_comparison.png')
plt.close()

# Average phase-amplitude histograms across units
avg_phase_amp_load1 = np.mean(phase_amp_histograms_load1, axis=0)
avg_phase_amp_load3 = np.mean(phase_amp_histograms_load3, axis=0)

plt.figure(figsize=(10, 6))
plt.bar(bin_centers, avg_phase_amp_load1, width=2*np.pi/len(bin_centers), 
        alpha=0.6, color='blue', label='Load 1')
plt.bar(bin_centers, avg_phase_amp_load3, width=2*np.pi/len(bin_centers), 
        alpha=0.6, color='red', label='Load 3')
plt.xlabel('Theta Phase (rad)')
plt.ylabel('Normalized Gamma Amplitude')
plt.title('Average Theta-Gamma PAC Across Units')
plt.xlim(-np.pi, np.pi)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           ['-π', '-π/2', '0', 'π/2', 'π'])
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/avg_pac_comparison.png')
plt.close()

print("\nPhase-amplitude coupling analysis complete. Plots saved to tmp_scripts directory.")