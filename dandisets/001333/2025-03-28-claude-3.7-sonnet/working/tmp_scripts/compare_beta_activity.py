"""
This script compares beta band activity between different electrodes and files in the dataset.
We'll calculate and visualize beta power across different electrodes and compare 
the beta band representation in the simulated beta and LFP data files.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simpson

# Function to calculate beta band power (13-30 Hz)
def calculate_beta_power(data, fs):
    f, Pxx = signal.welch(data, fs, nperseg=min(1024, len(data)))
    # Extract beta frequency range (13-30 Hz)
    beta_mask = (f >= 13) & (f <= 30)
    beta_freqs = f[beta_mask]
    beta_power = Pxx[beta_mask]
    # Calculate total power in beta band using Simpson's rule for integration
    total_beta_power = simpson(beta_power, beta_freqs)
    return total_beta_power

# Function to load a file and extract beta band power from all electrodes
def analyze_file(file_url, is_beta_file=False):
    print(f"\nAnalyzing file: {file_url.split('/')[-2]}")
    f = lindi.LindiH5pyFile.from_lindi_file(file_url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    
    # Get LFP or Beta data
    ecephys = nwb.processing["ecephys"]
    
    if is_beta_file:
        # Beta band files have Beta_Band_Voltage data
        data_series = ecephys["LFP"]["Beta_Band_Voltage"]
        # Get the data and timestamps
        data = data_series.data[:]
        timestamps = data_series.timestamps[:]
        fs = 1.0 / (timestamps[1] - timestamps[0])  # Sampling frequency
        print(f"Beta band data shape: {data.shape}, sampling rate: {fs:.2f} Hz")
        # Calculate beta power
        beta_power = calculate_beta_power(data, fs)
        print(f"Total beta band power: {beta_power:.6e} V²")
        return {"data": data, "fs": fs, "beta_power": beta_power, "subject_type": "beta"}
    
    else:
        # LFP files have LFP data
        data_series = ecephys["LFP"]["LFP"]
        # Due to large size, sample a subset (first 10000 samples)
        sample_size = 10000
        data = data_series.data[:sample_size]
        fs = data_series.rate  # Sampling frequency
        print(f"LFP data shape: {data.shape}, sampling rate: {fs:.2f} Hz")
        # Calculate beta power
        beta_power = calculate_beta_power(data, fs)
        print(f"Total beta band power: {beta_power:.6e} V²")
        return {"data": data, "fs": fs, "beta_power": beta_power, "subject_type": "lfp"}

# Load and analyze data from different files
beta_file = "https://lindi.neurosift.org/dandi/dandisets/001333/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/nwb.lindi.json"
lfp_file = "https://lindi.neurosift.org/dandi/dandisets/001333/assets/00df5264-001b-4bb0-a987-0ddfb6058961/nwb.lindi.json"

beta_results = analyze_file(beta_file, is_beta_file=True)
lfp_results = analyze_file(lfp_file, is_beta_file=False)

# Compare beta power between the two file types
plt.figure(figsize=(10, 6))
plt.bar(['Beta Band File', 'LFP File'], 
        [beta_results['beta_power'], lfp_results['beta_power']], 
        color=['blue', 'orange'])
plt.title('Comparison of Beta Band Power (13-30 Hz) Between File Types')
plt.ylabel('Total Power (V²)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('beta_power_comparison.png')

# Compare frequency spectra
plt.figure(figsize=(12, 6))

# Calculate PSD for beta file data
f_beta, Pxx_beta = signal.welch(beta_results['data'], beta_results['fs'], nperseg=min(1024, len(beta_results['data'])))
mask_beta = f_beta <= 40  # Limit to 0-40 Hz for better visualization
plt.subplot(1, 2, 1)
plt.semilogy(f_beta[mask_beta], Pxx_beta[mask_beta])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V²/Hz)')
plt.title('Power Spectrum - Beta Band File')
plt.axvspan(13, 30, alpha=0.2, color='red', label='Beta Band')
plt.grid(True)
plt.legend()

# Calculate PSD for LFP file data
f_lfp, Pxx_lfp = signal.welch(lfp_results['data'], lfp_results['fs'], nperseg=min(1024, len(lfp_results['data'])))
mask_lfp = f_lfp <= 40  # Limit to 0-40 Hz for better visualization
plt.subplot(1, 2, 2)
plt.semilogy(f_lfp[mask_lfp], Pxx_lfp[mask_lfp])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V²/Hz)')
plt.title('Power Spectrum - LFP File')
plt.axvspan(13, 30, alpha=0.2, color='red', label='Beta Band')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('spectrum_comparison.png')