"""
This script explores the LFP data from a sub-healthy-simulated-lfp file.
We'll examine the structure, time series, and spectral characteristics of
the LFP signals across different electrodes.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load LFP NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/00df5264-001b-4bb0-a987-0ddfb6058961/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the LFP data
ecephys = nwb.processing["ecephys"]
lfp_data = ecephys["LFP"]["LFP"]

# Print basic info about the LFP data
print("=== LFP Data Information ===")
print(f"Data shape: {lfp_data.data.shape}")
print(f"Sampling rate: {lfp_data.rate} Hz")
print(f"Starting time: {lfp_data.starting_time} seconds")
print(f"Electrode count: {lfp_data.electrodes['id'].data[:].size}")

# Get electrode information
electrodes = nwb.ec_electrodes
electrode_ids = electrodes["id"].data[:]
locations = electrodes["location"].data[:]
labels = electrodes["label"].data[:]

# Due to the large size, we'll just look at a subset of data
# for visualization (first 10,000 samples, ~5 seconds at 2000 Hz)
subset_size = 10000
subset_data = lfp_data.data[:subset_size]

# Calculate time array for the subset
time_array = np.arange(subset_size) / lfp_data.rate + lfp_data.starting_time

# Plot LFP data from a subset of electrodes (first 4 for clarity)
plt.figure(figsize=(12, 8))
for i in range(min(4, len(electrode_ids))):
    plt.subplot(4, 1, i+1)
    plt.plot(time_array, subset_data, label=f'Electrode {electrode_ids[i]} ({labels[i]})')
    plt.xlabel('Time (s)' if i == 3 else '')
    plt.ylabel('Voltage')
    plt.title(f'LFP Signal - Electrode {electrode_ids[i]} ({labels[i]})')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.savefig('lfp_timeseries.png')

# Compute and plot power spectral density for one electrode
fs = lfp_data.rate  # Sampling frequency
print(f"Computing PSD with sampling frequency: {fs} Hz")

# Compute Power Spectral Density
f, Pxx = signal.welch(subset_data, fs, nperseg=1024)

# Focus on the 0-40 Hz range for better visualization
mask = f <= 40
f = f[mask]
Pxx = Pxx[mask]

plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title('Power Spectral Density of LFP Signal')
plt.axvspan(13, 30, alpha=0.2, color='red', label='Beta Band (13-30 Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lfp_spectrum.png')

# Let's also look at a spectrogram to see how frequency content changes over time
plt.figure(figsize=(12, 6))
f, t, Sxx = signal.spectrogram(subset_data, fs, nperseg=256, noverlap=128)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of LFP Signal')
plt.colorbar(label='Power Spectral Density [dB/Hz]')
plt.axhline(y=13, color='r', linestyle='--', label='Beta Band Lower Bound (13 Hz)')
plt.axhline(y=30, color='r', linestyle='--', label='Beta Band Upper Bound (30 Hz)')
plt.legend()
plt.ylim(0, 40)  # Limit y-axis to 0-40 Hz for better visualization
plt.tight_layout()
plt.savefig('lfp_spectrogram.png')