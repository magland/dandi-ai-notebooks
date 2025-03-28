"""
This script explores the Beta_Band_Voltage data from a sub-healthy-simulated-beta file.
We'll plot the beta band signal and its power spectrum.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load beta band NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the Beta Band Voltage data
ecephys = nwb.processing["ecephys"]
beta_band = ecephys["LFP"]["Beta_Band_Voltage"]

# Print basic info about the Beta Band data
print("=== Beta Band Voltage Data ===")
print(f"Shape: {beta_band.data.shape}")
print(f"Electrode count: {beta_band.electrodes['id'].data[:].size}")

# Get the data and timestamps
beta_data = beta_band.data[:]
timestamps = beta_band.timestamps[:]

print(f"Data time range: {timestamps[0]} to {timestamps[-1]} seconds")
print(f"Data length: {len(beta_data)} samples")

# Plot the beta band voltage data
plt.figure(figsize=(12, 5))
plt.plot(timestamps, beta_data)
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.title('Beta Band Voltage Signal')
plt.grid(True)
plt.tight_layout()
plt.savefig('beta_band_signal.png')

# Compute and plot power spectral density
fs = 1.0 / (timestamps[1] - timestamps[0])  # Sampling frequency
print(f"Sampling frequency: {fs} Hz")

# Compute Power Spectral Density
f, Pxx = signal.welch(beta_data, fs, nperseg=256)

# Focus on the 0-40 Hz range for better visualization
mask = f <= 40
f = f[mask]
Pxx = Pxx[mask]

plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title('Power Spectral Density of Beta Band Signal')
plt.axvspan(13, 30, alpha=0.2, color='red', label='Beta Band (13-30 Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('beta_band_spectrum.png')