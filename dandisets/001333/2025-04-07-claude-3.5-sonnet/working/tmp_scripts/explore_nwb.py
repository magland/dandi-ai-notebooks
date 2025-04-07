"""
Initial exploration of NWB file structure and content focusing on LFP data
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dandi.dandiapi import DandiAPIClient
import requests
import os

# Set up plotting style
sns.set_theme()

# Get a sample NWB file URL using DANDI API
client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
assets = list(dandiset.get_assets())

# Use the first healthy simulated beta file
sample_asset = assets[0]
file_url = sample_asset.get_content_url(follow_redirects=1)

# Download the file
local_path = 'tmp_scripts/sample.nwb'
print(f"Downloading file from {file_url}")
response = requests.get(file_url)
with open(local_path, 'wb') as f:
    f.write(response.content)
print("Download complete")

# Open the NWB file
with h5py.File(local_path, 'r') as f:
    # Try to access LFP data
    try:
        # Get LFP data
        lfp_data = f['processing/ecephys/LFP/LFP/data'][:]
        electrodes = f['processing/ecephys/LFP/LFP/electrodes'][:]
        start_time = f['processing/ecephys/LFP/LFP/starting_time'][()]
        
        # Calculate time points (assuming 1000 Hz sampling rate based on dataset description)
        sampling_rate = 1000  # Hz
        duration = len(lfp_data) / sampling_rate
        time_points = np.linspace(start_time, start_time + duration, len(lfp_data))
        
        print("\n=== LFP Data Information ===")
        print(f"Number of timepoints: {len(lfp_data)}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Starting time: {start_time}")
        print(f"Electrodes: {electrodes}")
        
        # Plot a sample of the LFP data (first 1 second)
        plt.figure(figsize=(12, 6))
        plt.plot(time_points[:1000], lfp_data[:1000])
        plt.xlabel('Time (s)')
        plt.ylabel('LFP Signal (μV)')
        plt.title('Sample LFP Signal (First 1 second)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('tmp_scripts/lfp_sample.png')
        plt.close()
        
        # Create a spectrogram
        plt.figure(figsize=(12, 6))
        Pxx, freqs, bins, im = plt.specgram(lfp_data[:10000], Fs=sampling_rate, 
                                          NFFT=256, noverlap=128,
                                          cmap='viridis')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('LFP Spectrogram')
        plt.colorbar(im, label='Power (dB)')
        plt.ylim(0, 100)  # Focus on frequencies up to 100 Hz
        plt.tight_layout()
        plt.savefig('tmp_scripts/lfp_spectrogram.png')
        plt.close()
        
        # Calculate and plot power spectrum
        plt.figure(figsize=(12, 6))
        f, Pxx = plt.psd(lfp_data, Fs=sampling_rate, NFFT=1024, 
                        noverlap=512, scale_by_freq=True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.title('LFP Power Spectrum')
        plt.grid(True)
        plt.xlim(0, 100)  # Focus on frequencies up to 100 Hz
        plt.tight_layout()
        plt.savefig('tmp_scripts/lfp_psd.png')
        plt.close()
            
    except Exception as e:
        print(f"Error accessing LFP data: {e}")

# Clean up
os.remove(local_path)