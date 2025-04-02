'''
This script compares Beta Band Voltage signals between healthy and Parkinson's subjects.
Beta oscillations (13-30 Hz) in the subthalamic nucleus are typically used as 
biomarkers for Parkinson's Disease symptoms.
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

# Load a healthy subject beta file
healthy_f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/b344c8b7-422f-46bb-b016-b47dc1e87c65/nwb.lindi.json")
healthy_nwb = pynwb.NWBHDF5IO(file=healthy_f, mode='r').read()

# Load a Parkinson's subject beta file
parkinsons_f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/6b17c99d-19b9-4846-b1c9-671d9b187149/nwb.lindi.json")
parkinsons_nwb = pynwb.NWBHDF5IO(file=parkinsons_f, mode='r').read()

# Print basic information 
print("Dataset Information:")
print(f"Healthy subject ID: {healthy_nwb.subject.subject_id}")
print(f"Parkinson's subject ID: {parkinsons_nwb.subject.subject_id}")

# Access Beta Band Voltage data
healthy_beta = healthy_nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"]
parkinsons_beta = parkinsons_nwb.processing["ecephys"]["LFP"]["Beta_Band_Voltage"]

# Get data and timestamps
healthy_beta_data = healthy_beta.data[:]
healthy_beta_timestamps = healthy_beta.timestamps[:]

parkinsons_beta_data = parkinsons_beta.data[:]
parkinsons_beta_timestamps = parkinsons_beta.timestamps[:]

print(f"Healthy beta data shape: {healthy_beta_data.shape}")
print(f"Parkinson's beta data shape: {parkinsons_beta_data.shape}")

# Calculate sampling rate from timestamps
healthy_sampling_rate = 1 / np.mean(np.diff(healthy_beta_timestamps))
print(f"Estimated sampling rate from timestamps: {healthy_sampling_rate:.2f} Hz")

# Plot Beta Band Voltage signals for comparison
plt.figure(figsize=(10, 6))
plt.plot(healthy_beta_timestamps, healthy_beta_data, label="Healthy Subject", alpha=0.8)
plt.plot(parkinsons_beta_timestamps, parkinsons_beta_data, label="Parkinson's Subject", alpha=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("Beta Band Voltage (V)")
plt.title("Comparison of Beta Band Voltage: Healthy vs. Parkinson's")
plt.legend()
plt.grid(True)
plt.savefig("beta_comparison.png")
plt.close()

# Compute and plot statistics on the beta band voltage
def plot_beta_stats(healthy_data, parkinsons_data, filename):
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame for easier analysis and plotting
    df = pd.DataFrame({
        'Healthy': healthy_data,
        'Parkinsons': parkinsons_data
    })
    
    # Plot histograms
    plt.subplot(2, 2, 1)
    plt.hist(healthy_data, bins=30, alpha=0.7, label="Healthy")
    plt.hist(parkinsons_data, bins=30, alpha=0.7, label="Parkinson's")
    plt.xlabel("Beta Band Voltage (V)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Beta Band Voltage")
    plt.legend()
    plt.grid(True)
    
    # Plot box plots
    plt.subplot(2, 2, 2)
    df.boxplot()
    plt.ylabel("Beta Band Voltage (V)")
    plt.title("Box Plot of Beta Band Voltage")
    plt.grid(True)
    
    # Plot statistical measures
    plt.subplot(2, 2, 3)
    stats = {
        'Mean': [np.mean(healthy_data), np.mean(parkinsons_data)],
        'Median': [np.median(healthy_data), np.median(parkinsons_data)],
        'Std Dev': [np.std(healthy_data), np.std(parkinsons_data)],
        'Max': [np.max(healthy_data), np.max(parkinsons_data)],
        'Min': [np.min(healthy_data), np.min(parkinsons_data)]
    }
    
    stats_df = pd.DataFrame(stats, index=['Healthy', 'Parkinsons'])
    plt.axis('off')
    plt.table(cellText=stats_df.values, rowLabels=stats_df.index, 
              colLabels=stats_df.columns, loc='center', cellLoc='center')
    plt.title("Statistical Measures of Beta Band Voltage")
    
    # Plot a segment of the signals for detailed comparison
    plt.subplot(2, 2, 4)
    # Take a 5-second segment or whatever is available
    segment_length = min(100, len(healthy_data), len(parkinsons_data))
    plt.plot(healthy_beta_timestamps[:segment_length], 
             healthy_data[:segment_length], label="Healthy")
    plt.plot(parkinsons_beta_timestamps[:segment_length], 
             parkinsons_data[:segment_length], label="Parkinson's")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Beta Band Voltage (V)")
    plt.title("First 100 Samples of Beta Band Voltage")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot statistical comparisons
plot_beta_stats(healthy_beta_data, parkinsons_beta_data, "beta_statistics.png")