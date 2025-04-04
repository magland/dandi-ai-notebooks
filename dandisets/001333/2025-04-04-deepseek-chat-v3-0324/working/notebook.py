# %% [markdown]
# # Exploring Dandiset 001333: Parkinson's Electrophysiological Signal Dataset (PESD)
# 
# This notebook provides an introduction to exploring and analyzing the Parkinson's Electrophysiological Signal Dataset (PESD) from DANDI archive.
# 
# **Note:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please exercise caution when interpreting the code or results.

# %% [markdown]
# ## Dataset Overview
# 
# The dataset contains electrophysiological signals from both healthy and parkinsonian subjects. Key features:
# - Beta Average Rectified Voltage (ARV) signals in frequency domain
# - Local Field Potential (LFP) signals from Subthalamic Nucleus (STN) in time domain
# - Parkinsonian signals show high power density at beta frequency (13-30 Hz)
# 
# More details available in the original article: ["Preliminary Results of Neuromorphic Controller Design and a Parkinson's Disease Dataset Building for Closed-Loop Deep Brain Stimulation"](https://arxiv.org/abs/2407.17756)

# %%
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# %% [markdown]
# ## Listing Dandiset Assets
# 
# First, let's list all assets available in this Dandiset:

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
assets = list(dandiset.get_assets())

print(f"Total assets: {len(assets)}")
print("\nSample assets:")
for asset in assets[:5]:
    print(f"- {asset.path} (size: {asset.size/1024:.1f} KB)")

# %% [markdown]
# ## Loading NWB Files
# 
# The dataset contains NWB files with electrophysiology data. Here's how to load them:

# %%
def load_nwb_file(asset_url):
    """Helper function to load an NWB file from DANDI"""
    try:
        # Note: In practice you would use the lindi URL from dandi-notebook-gen-tools
        # This is a placeholder that would need to be updated with proper file loading
        with h5py.File(asset_url, 'r') as f:
            print(f"File contains datasets: {list(f.keys())}")
            return f
    except Exception as e:
        print(f"Error loading NWB file: {e}")
        return None

# Example usage (would need actual accessible URL):
# nwb_file = load_nwb_file("https://api.dandiarchive.org/api/assets/e0fa57b2-02a4-4c20-92df-d7eb64b60170/download/")

# %% [markdown]
# ## Data Exploration
# 
# Once files are properly loaded, you could explore:
# - Time series data (LFP signals)
# - Frequency domain data (Beta ARV)
# - Electrode information
# - Recording metadata

# %%
# Placeholder for data exploration code
# Would include actual analysis once files are properly accessible

# %% [markdown]
# ## Visualization Examples
# 
# When data is loaded, some potential visualizations include:

# %%
# Example plot (placeholder with random data)
plt.figure(figsize=(10, 4))
plt.plot(np.random.randn(1000), alpha=0.7)
plt.title("Example LFP Signal (placeholder)")
plt.xlabel("Time (samples)")
plt.ylabel("Voltage (μV)")
plt.show()

# %% [markdown]
# ## Next Steps
# 
# To continue analysis:
# 1. Ensure you have proper access to the NWB files
# 2. Update the file loading code with correct URLs
# 3. Explore specific signals of interest
# 4. Compare healthy vs parkinsonian subjects