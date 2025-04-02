'''
This script compares LFP signals between healthy and Parkinson's subjects.
It loads LFP data from both subject types and plots:
1. Raw LFP signals for both subjects
2. Spectrograms to visualize frequency content over time
3. Power spectral densities to quantify frequency content
'''

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Set the plotting style
import seaborn as sns
sns.set_theme()

# Configure figure size and DPI for better visibility
plt.figure(figsize=(12, 10), dpi=100)

# Load a healthy subject LFP file
healthy_f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/00df5264-001b-4bb0-a987-0ddfb6058961/nwb.lindi.json")
healthy_nwb = pynwb.NWBHDF5IO(file=healthy_f, mode='r').read()

# Load a Parkinson's subject LFP file
parkinsons_f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/5535e23a-9029-43c5-80fb-0fb596541a81/nwb.lindi.json")
parkinsons_nwb = pynwb.NWBHDF5IO(file=parkinsons_f, mode='r').read()

# Print basic information 
print("Dataset Information:")
print(f"Healthy subject ID: {healthy_nwb.subject.subject_id}")
print(f"Parkinson's subject ID: {parkinsons_nwb.subject.subject_id}")

# Access LFP data
healthy_lfp = healthy_nwb.processing["ecephys"]["LFP"]["LFP"]
parkinsons_lfp = parkinsons_nwb.processing["ecephys"]["LFP"]["LFP"]

# Get sampling rate
sampling_rate = healthy_lfp.rate
print(f"Sampling rate: {sampling_rate} Hz")

# Select a subset of data (first 5 seconds) to avoid loading too much data
duration = 5  # seconds
n_samples = int(duration * sampling_rate)
n_samples = min(n_samples, healthy_lfp.data.shape[0])

# Load a subset of data for first electrode (shank0_elec0)
healthy_data = healthy_lfp.data[:n_samples]
parkinsons_data = parkinsons_lfp.data[:n_samples]

print(f"Data shape: {healthy_data.shape}")

# Create time array
time = np.arange(n_samples) / sampling_rate

# Plot raw LFP signals for comparison
plt.figure(figsize=(10, 6))
plt.plot(time, healthy_data, label="Healthy Subject", alpha=0.8)
plt.plot(time, parkinsons_data, label="Parkinson's Subject", alpha=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("LFP Voltage")
plt.title("Comparison of LFP Signals: Healthy vs. Parkinson's")
plt.legend()
plt.grid(True)
plt.savefig("lfp_comparison.png")
plt.close()

# Calculate and plot spectrogram to visualize frequency content over time
def plot_spectrogram(data, sampling_rate, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Calculate spectrogram 
    f, t, Sxx = signal.spectrogram(data, fs=sampling_rate, nperseg=256, noverlap=128)
    
    # Plot spectrogram (log scale for better visualization)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='Power/frequency (dB/Hz)')
    plt.ylim(0, 100)  # Limit to 0-100 Hz range
    plt.savefig(filename)
    plt.close()

# Plot spectrograms
plot_spectrogram(healthy_data, sampling_rate, 
                "Spectrogram: Healthy Subject", "healthy_spectrogram.png")
plot_spectrogram(parkinsons_data, sampling_rate, 
                "Spectrogram: Parkinson's Subject", "parkinsons_spectrogram.png")

# Calculate and plot power spectral density
def plot_psd(healthy_data, parkinsons_data, sampling_rate, filename):
    plt.figure(figsize=(10, 6))
    
    # Calculate PSD for both datasets
    f_healthy, Pxx_healthy = signal.welch(healthy_data, fs=sampling_rate, nperseg=1024)
    f_parkinsons, Pxx_parkinsons = signal.welch(parkinsons_data, fs=sampling_rate, nperseg=1024)
    
    # Plot PSDs
    plt.semilogy(f_healthy, Pxx_healthy, label="Healthy Subject")
    plt.semilogy(f_parkinsons, Pxx_parkinsons, label="Parkinson's Subject")
    
    # Highlight beta frequency band (13-30 Hz)
    plt.axvspan(13, 30, alpha=0.2, color='red', label='Beta Band (13-30 Hz)')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.title('Power Spectral Density Comparison')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 100)  # Limit to 0-100 Hz range
    plt.savefig(filename)
    plt.close()

# Plot PSD
plot_psd(healthy_data, parkinsons_data, sampling_rate, "psd_comparison.png")