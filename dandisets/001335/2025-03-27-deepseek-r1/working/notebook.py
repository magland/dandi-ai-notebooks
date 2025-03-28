# %% [markdown]
# # DANDI 001335 Analysis Notebook
# **AI-Generated Notebook**  
# *Verify results before scientific use*

# %% [markdown]
# ## Dataset Overview
# - **Dandiset**: [001335](https://dandiarchive.org/dandiset/001335)
# - **Title**: Neuropixels Hippocampal Recordings During Odor Presentation
# - **Subject**: sub-M541 (Male C57BL/6J mouse)
# - **Session**: 14m recording, 6 odors (A-F), 101 trials each
# - **Key Features**:
#   - 238 spike-sorted units (CA1/DG)
#   - 64-channel LFP @2500Hz
#   - Full trial event markers
# - **License**: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

# %% [markdown]
# ## Environment Setup

# %%
# Core neuroscience packages
from dandi.dandiapi import DandiAPIClient
import pynwb
import lindi

# Analysis stack
import numpy as np
import pandas as pd
from scipy import signal, stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14
})

# %% [markdown]
# ## Data Access & Validation

# %%
# Initialize DANDI client
client = DandiAPIClient()
dandiset = client.get_dandiset("001335")
asset = list(dandiset.get_assets())[0]

# Load data
nwb_url = 'https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json'
h5_file = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwb = pynwb.NWBHDF5IO(file=h5_file, mode='r').read()

# Validate critical metadata
assert 2025 == nwb.session_start_time.year, "Unexpected recording year"

# Extract region names from both bytes and string representations
locations = []
for loc in nwb.electrodes.location[:]:
    if isinstance(loc, bytes):
        # Decode bytes and clean quotes/spaces
        loc_str = loc.decode().strip("'\" ")
    elif isinstance(loc, str) and (loc.startswith("b'") or loc.startswith('b"')):
        # Remove byte prefix and quotes
        loc_str = loc[2:-1].strip("'\" ")
    else:
        # Handle normal strings
        loc_str = str(loc).strip("'\" ")
    locations.append(loc_str)

print("Cleaned electrode locations:", np.unique(locations))
assert 'CA1' in locations, f"Missing CA1 electrodes. Found: {np.unique(locations)}"

print("✅ Dataset validation passed")

# %% [markdown]
# ## Neural Response Analysis

# %%
def analyze_unit_response(unit, events, pre_window=(-1,0), resp_window=(0,1)):
    """Analyze unit responses with statistical validation"""
    spikes = unit.spike_times[:]
    
    # Calculate rates
    baseline = len(spikes)/846.5  # Overall baseline
    pre_counts = []
    resp_counts = []
    
    for start, _ in events:
        pre = ((spikes >= start+pre_window[0]) & (spikes < start+pre_window[1])).sum()
        resp = ((spikes >= start+resp_window[0]) & (spikes < start+resp_window[1])).sum()
        pre_counts.append(pre/abs(pre_window[0]))
        resp_counts.append(resp/(resp_window[1]-resp_window[0]))
    
    # Statistical testing
    t_stat, p_val = stats.ttest_rel(resp_counts, pre_counts)
    
    return {
        'unit_id': unit.id,
        'location': unit.location,
        'baseline': baseline,
        'response_ratio': np.mean(resp_counts)/baseline,
        'p_value': p_val
    }

# Analyze first 10 CA1 units
odor_a_events = list(zip(nwb.intervals['Odor A ON'].start_time, 
                        nwb.intervals['Odor A ON'].stop_time))
ca1_units = [u for u in nwb.units if u.location == 'CA1'][:10]

results = [analyze_unit_response(u, odor_a_events) for u in ca1_units]
response_df = pd.DataFrame(results)
display(response_df.head())

# %% [markdown]
# ### Visualization: Significant Responses

# %%
sig_df = response_df[response_df.p_value < 0.05]

plt.figure(figsize=(10,5))
sns.barplot(x='unit_id', y='response_ratio', hue='location',
            data=sig_df, palette='viridis', edgecolor='k')
plt.axhline(1, color='r', linestyle='--', label='Baseline')
plt.title('Significant Odor Responses (p < 0.05)\nCA1 Units')
plt.ylabel('Response Ratio (Post/Pre)')
plt.xlabel('Unit ID')
plt.legend(bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spatial Organization Analysis

# %%
# Create electrode positions DataFrame
electrode_df = pd.DataFrame({
    'x': [e.x for e in nwb.electrodes],
    'y': [e.y for e in nwb.electrodes],
    'location': [e.location for e in nwb.electrodes]
})

# Plot spatial distribution
plt.figure(figsize=(10,6))
sns.scatterplot(data=electrode_df, x='x', y='y', hue='location',
               palette='Set2', s=100, alpha=0.7)
plt.title('Electrode Locations by Brain Region')
plt.xlabel('Medial-Lateral (μm)')
plt.ylabel('Dorsal-Ventral (μm)')
plt.legend(title='Brain Region')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## LFP Spectral Analysis

# %%
def lfp_power_analysis(lfp_data, fs=2500, nperseg=1024):
    """Compute standardized spectral analysis"""
    f, t, Sxx = signal.spectrogram(lfp_data, fs=fs, nperseg=nperseg)
    return {
        'frequencies': f,
        'times': t,
        'power': 10 * np.log10(Sxx),
        'mean_power': np.mean(Sxx, axis=1)
    }

# Analyze CA1 LFP
ca1_mask = electrode_df.location == 'CA1'
lfp_data = nwb.processing['ecephys'].data_interfaces['LFP'].data[:, ca1_mask]
spectral_data = lfp_power_analysis(lfp_data[:30000,0])  # First 12s

# Visualization
plt.figure(figsize=(12,4))
plt.pcolormesh(spectral_data['times'], spectral_data['frequencies'],
               spectral_data['power'], cmap='viridis', shading='gouraud')
plt.colorbar(label='Power (dB)')
plt.ylim(0, 100)
plt.title('CA1 LFP Spectrogram (First 12 Seconds)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Critical Notes & Verification
# 1. **Data Provenance**: Remote streaming via DANDI API
# 2. **Statistical Limits**: Uncorrected p-values (apply FDR for research)
# 3. **Spatial Context**: Electrode positions in probe space, not anatomical
# 4. **Verification Requirements**:
#   - Validate unit isolation quality metrics
#   - Confirm odor timing alignment
#   - Check spectrogram parameters