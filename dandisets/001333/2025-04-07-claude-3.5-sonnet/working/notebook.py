# %% [markdown]
# # Exploring the Parkinson's Electrophysiological Signal Dataset (PESD)
# 
# **Note: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.**
# 
# This notebook demonstrates how to access and analyze data from DANDI:001333, a dataset containing electrophysiological signals from both healthy and parkinsonian subjects. The dataset includes two types of signals:
# 
# 1. Beta Average Rectified Voltage (ARV) - frequency domain signals
# 2. Local Field Potential (LFP) from the Subthalamic Nucleus (STN) - time domain signals
# 
# ## Dataset Overview
# 
# The dataset focuses on beta oscillations (13-30 Hz) in the subthalamic nucleus (STN), which are typically used as pathological biomarkers for Parkinson's Disease symptoms. The LFP signals are derived from synchronized activity of neuron populations between the cortex, STN, and thalamus.
# 
# ## Required Packages
# 
# To run this notebook, you'll need the following Python packages:
# - dandi
# - h5py
# - numpy
# - matplotlib
# - seaborn
# - requests

# %%
# Import required packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dandi.dandiapi import DandiAPIClient
import requests

# Set up plotting style
sns.set_theme()

# %% [markdown]
# ## Accessing the Dataset
# 
# First, let's use the DANDI API to list all assets in the dataset. This will help us understand what files are available for analysis.

# %%
# Initialize DANDI API client
client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
assets = list(dandiset.get_assets())

# Display information about available files
print(f"Total number of files: {len(assets)}")
print("\nSample of available files:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading and Examining LFP Data
# 
# We'll load data from one of the files and examine its content. The dataset contains both healthy and parkinsonian recordings.
# For this example, we'll look at a healthy simulated beta recording.

# %%
# Get the first file
sample_asset = assets[0]
file_url = sample_asset.get_content_url(follow_redirects=1)

# Download the NWB file
local_path = 'sample.nwb'
print(f"Downloading file: {sample_asset.path}")
response = requests.get(file_url)
with open(local_path, 'wb') as f:
    f.write(response.content)

# %% [markdown]
# ### Exploring the LFP Signal
# 
# Now that we have loaded the data, let's examine the LFP signal characteristics. The recording contains:
# - Multiple electrodes recording from the STN
# - Time series data at 1000 Hz sampling rate
# - Both time-domain and frequency-domain information

# %%
# Open the NWB file and access the LFP data
with h5py.File(local_path, 'r') as f:
    # Get LFP data
    lfp_data = f['processing/ecephys/LFP/LFP/data'][:]
    electrodes = f['processing/ecephys/LFP/LFP/electrodes'][:]
    start_time = f['processing/ecephys/LFP/LFP/starting_time'][()]
    
    # Calculate time points
    sampling_rate = 1000  # Hz
    duration = len(lfp_data) / sampling_rate
    time_points = np.linspace(start_time, start_time + duration, len(lfp_data))
    
    print("=== LFP Data Information ===")
    print(f"Number of timepoints: {len(lfp_data)}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Starting time: {start_time}")
    print(f"Number of electrodes: {len(electrodes)}")

# %% [markdown]
# ### Visualizing the LFP Signal
# 
# Let's create several visualizations to understand the signal characteristics:
# 1. Time domain plot of the raw signal
# 2. Spectrogram showing frequency content over time
# 3. Power spectral density showing the distribution of signal power across frequencies

# %%
# Plot time domain signal (first second of data)
plt.figure(figsize=(12, 6))
plt.plot(time_points[:1000], lfp_data[:1000])
plt.xlabel('Time (s)')
plt.ylabel('LFP Signal (μV)')
plt.title('Sample LFP Signal (First 1 second)')
plt.grid(True)
plt.show()

# %%
# Create spectrogram
plt.figure(figsize=(12, 6))
Pxx, freqs, bins, im = plt.specgram(lfp_data[:10000], Fs=sampling_rate, 
                                  NFFT=256, noverlap=128,
                                  cmap='viridis')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('LFP Spectrogram')
plt.colorbar(im, label='Power (dB)')
plt.ylim(0, 100)  # Focus on frequencies up to 100 Hz
plt.show()

# %%
# Calculate and plot power spectrum
plt.figure(figsize=(12, 6))
f, Pxx = plt.psd(lfp_data, Fs=sampling_rate, NFFT=1024, 
                noverlap=512, scale_by_freq=True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('LFP Power Spectrum')
plt.grid(True)
plt.xlim(0, 100)  # Focus on frequencies up to 100 Hz
plt.show()

# %% [markdown]
# ### Analyzing Beta Band Activity
# 
# The beta band (13-30 Hz) is particularly important in this dataset as it serves as a biomarker for Parkinson's Disease. Let's create a function to analyze the beta band power.

# %%
def analyze_beta_band(data, sampling_rate):
    """Analyze power in the beta frequency band (13-30 Hz)"""
    # Calculate PSD
    f, Pxx = plt.psd(data, Fs=sampling_rate, NFFT=1024, 
                     noverlap=512, scale_by_freq=True, visible=False)
    
    # Find indices for beta band
    beta_mask = (f >= 13) & (f <= 30)
    beta_freqs = f[beta_mask]
    beta_power = Pxx[beta_mask]
    
    return beta_freqs, beta_power

# Analyze beta band
beta_freqs, beta_power = analyze_beta_band(lfp_data, sampling_rate)

# Plot beta band power
plt.figure(figsize=(10, 6))
plt.plot(beta_freqs, 10 * np.log10(beta_power))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB/Hz)')
plt.title('Beta Band (13-30 Hz) Power Spectrum')
plt.grid(True)
plt.show()

# Clean up
import os
os.remove(local_path)

# %% [markdown]
# ## Next Steps
# 
# This notebook demonstrates basic loading and analysis of LFP data from the PESD dataset. Researchers might want to:
# 
# 1. Compare beta band power between healthy and parkinsonian recordings
# 2. Analyze temporal patterns in beta activity
# 3. Investigate relationships between different frequency bands
# 4. Examine correlations between electrodes
# 
# The dataset contains multiple recordings that can be analyzed using similar methods to those shown here.