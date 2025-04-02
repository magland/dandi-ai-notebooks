'''
This script analyzes multiple beta band recordings from both healthy and Parkinson's subjects
to determine if the patterns are consistent across different sessions.
'''

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set the plotting style
import seaborn as sns
sns.set_theme()

# List of asset IDs for healthy beta subjects (using 5 examples)
healthy_beta_assets = [
    "b344c8b7-422f-46bb-b016-b47dc1e87c65",  # ses-162
    "da77917e-655c-4eeb-a0a6-7529a8a35901",  # ses-218
    "aa743a96-e79b-40b1-a0df-55ef3460e856",  # ses-279
    "d0e8beef-ee7a-44a9-bebc-e0865b4c8f42",  # ses-345
    "068afe41-3b8e-402f-91d4-9f297df677a2"   # ses-423
]

# List of asset IDs for Parkinson's beta subjects (using 5 examples)
parkinsons_beta_assets = [
    "6b17c99d-19b9-4846-b1c9-671d9b187149",  # ses-111
    "f78e0730-f53e-4513-8068-4b5e0e1a21c2",  # ses-112
    "ad19aec9-221f-4f8b-8c95-e31345480f54",  # ses-120
    "710ab238-483d-4314-860a-d64663c7cd16",  # ses-163
    "b3ee75b5-9a42-440c-aaf2-1f33e9fc0c49"   # ses-164
]

# Function to load beta band voltage data from a given asset ID
def load_beta_data(asset_id):
    url = f"https://lindi.neurosift.org/dandi/dandisets/001333/assets/{asset_id}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    
    # Extract session ID from the NWB file
    session_id = nwb.session_id if hasattr(nwb, 'session_id') else asset_id[-10:-6]
    
    # Access Beta Band Voltage data
    beta_data = nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"].data[:]
    
    # Calculate statistics
    stats = {
        'mean': np.mean(beta_data),
        'median': np.median(beta_data),
        'std': np.std(beta_data),
        'max': np.max(beta_data)
    }
    
    return session_id, stats, beta_data

# Load data for healthy subjects
print("Loading healthy subject data...")
healthy_stats = []
healthy_data = {}
healthy_session_ids = []

for asset_id in healthy_beta_assets:
    try:
        session_id, stats, data = load_beta_data(asset_id)
        healthy_stats.append(stats)
        healthy_data[session_id] = data
        healthy_session_ids.append(session_id)
        print(f"  Loaded session {session_id}, mean beta: {stats['mean']:.8f}")
    except Exception as e:
        print(f"  Error loading asset {asset_id}: {e}")

# Load data for Parkinson's subjects
print("\nLoading Parkinson's subject data...")
parkinsons_stats = []
parkinsons_data = {}
parkinsons_session_ids = []

for asset_id in parkinsons_beta_assets:
    try:
        session_id, stats, data = load_beta_data(asset_id)
        parkinsons_stats.append(stats)
        parkinsons_data[session_id] = data
        parkinsons_session_ids.append(session_id)
        print(f"  Loaded session {session_id}, mean beta: {stats['mean']:.8f}")
    except Exception as e:
        print(f"  Error loading asset {asset_id}: {e}")

# Create DataFrames for statistics
healthy_df = pd.DataFrame(healthy_stats, index=healthy_session_ids)
healthy_df['condition'] = 'Healthy'

parkinsons_df = pd.DataFrame(parkinsons_stats, index=parkinsons_session_ids)
parkinsons_df['condition'] = 'Parkinson\'s'

# Combine DataFrames
combined_df = pd.concat([healthy_df, parkinsons_df])
print("\nStatistics summary:")
print(combined_df.groupby('condition').mean())

# Plot mean beta values across sessions
plt.figure(figsize=(12, 6))
combined_df.reset_index().plot(x='index', y='mean', kind='bar', color=combined_df.reset_index()['condition'].map({'Healthy': 'blue', 'Parkinson\'s': 'orange'}))
plt.title('Mean Beta Band Voltage by Session')
plt.xlabel('Session ID')
plt.ylabel('Mean Beta Band Voltage (V)')
plt.legend(['Mean Beta Band Voltage'])
plt.grid(True, axis='y')
plt.savefig('mean_beta_by_session.png')
plt.close()

# Plot distribution of beta values for all sessions
plt.figure(figsize=(12, 6))

# Calculate an appropriate number of bins
n_bins = 30

# Plot histogram for each healthy session
for session_id, data in healthy_data.items():
    plt.hist(data, bins=n_bins, alpha=0.3, label=f"Healthy {session_id}", color='blue')

# Plot histogram for each Parkinson's session
for session_id, data in parkinsons_data.items():
    plt.hist(data, bins=n_bins, alpha=0.3, label=f"Parkinson's {session_id}", color='orange')

plt.title('Distribution of Beta Band Voltage Across Sessions')
plt.xlabel('Beta Band Voltage (V)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('beta_distributions.png')
plt.close()

# Box plot comparing healthy vs. Parkinson's across sessions
plt.figure(figsize=(12, 6))

# Prepare data for box plot
healthy_all = np.concatenate([data for data in healthy_data.values()])
parkinsons_all = np.concatenate([data for data in parkinsons_data.values()])

# Create a DataFrame for the box plot
box_df = pd.DataFrame({
    'Healthy': healthy_all,
    'Parkinson\'s': parkinsons_all
})

# Create box plot
box_df.boxplot()
plt.title('Beta Band Voltage: Healthy vs. Parkinson\'s (All Sessions)')
plt.ylabel('Beta Band Voltage (V)')
plt.grid(True)
plt.savefig('beta_boxplot_all_sessions.png')
plt.close()

# Violin plot for more detailed distribution comparison
plt.figure(figsize=(12, 6))
violin_df = pd.DataFrame({
    'Beta Voltage': np.concatenate([healthy_all, parkinsons_all]),
    'Condition': ['Healthy'] * len(healthy_all) + ['Parkinson\'s'] * len(parkinsons_all)
})
sns.violinplot(x='Condition', y='Beta Voltage', data=violin_df)
plt.title('Beta Band Voltage Distribution: Healthy vs. Parkinson\'s')
plt.grid(True)
plt.savefig('beta_violinplot.png')
plt.close()