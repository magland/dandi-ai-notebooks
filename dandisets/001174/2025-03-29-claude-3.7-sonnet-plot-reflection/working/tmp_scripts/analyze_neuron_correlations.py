"""
This script analyzes correlations between neurons in the calcium imaging data.
It calculates a correlation matrix between neurons and visualizes potential
functional relationships between them.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# Set the output directory for plots
output_dir = "tmp_scripts"

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/9c3678d5-22c3-402c-8cd4-6bc38c4d61e3/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get fluorescence data
fluorescence = nwb.processing['ophys']['Fluorescence']['RoiResponseSeries']

# Extract fluorescence time series data for all neurons
# Using a subset of the data to make computation faster (first 3000 frames)
num_frames = min(3000, fluorescence.data.shape[0])
fluor_data = fluorescence.data[:num_frames, :]

# Compute pairwise correlations between neurons
corr_matrix = np.corrcoef(fluor_data.T)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(label='Correlation coefficient')
plt.title('Correlation Matrix Between Neurons')
plt.xlabel('Neuron ID')
plt.ylabel('Neuron ID')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'neuron_correlation_matrix.png'), dpi=150)
plt.close()

# Identify pairs of highly correlated neurons (|correlation| > 0.5)
high_corr_pairs = []
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[0]):  # Only look at upper triangle
        if abs(corr_matrix[i, j]) > 0.5:
            high_corr_pairs.append((i, j, corr_matrix[i, j]))

print(f"Number of highly correlated neuron pairs (|r| > 0.5): {len(high_corr_pairs)}")
if high_corr_pairs:
    for i, j, corr in high_corr_pairs[:5]:  # Show at most 5 examples
        print(f"Neurons {i} and {j}: r = {corr:.3f}")

# Plot activity traces for a few highly correlated pairs
if high_corr_pairs:
    # Find the most correlated pair
    most_corr_pair = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[0]
    i, j, corr = most_corr_pair
    
    # Plot time series for both neurons
    plt.figure(figsize=(12, 6))
    
    time_seconds = np.arange(num_frames) / fluorescence.rate
    
    plt.subplot(2, 1, 1)
    plt.plot(time_seconds, fluor_data[:, i], label=f'Neuron {i}')
    plt.plot(time_seconds, fluor_data[:, j], label=f'Neuron {j}', alpha=0.7)
    plt.title(f'Activity traces for most correlated pair: Neurons {i} & {j} (r = {corr:.3f})')
    plt.legend()
    plt.ylabel('Fluorescence (a.u.)')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot to show correlation
    plt.subplot(2, 1, 2)
    plt.scatter(fluor_data[:, i], fluor_data[:, j], alpha=0.5)
    plt.xlabel(f'Neuron {i} Fluorescence (a.u.)')
    plt.ylabel(f'Neuron {j} Fluorescence (a.u.)')
    plt.title(f'Correlation between Neurons {i} & {j}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlated_neurons_example.png'), dpi=150)
    plt.close()

# Calculate mean correlation for each neuron
mean_corr = np.zeros(corr_matrix.shape[0])
for i in range(corr_matrix.shape[0]):
    # Exclude self-correlation (which is always 1)
    mean_corr[i] = np.mean(np.delete(corr_matrix[i, :], i))

# Plot mean correlation by neuron
plt.figure(figsize=(10, 6))
plt.bar(range(len(mean_corr)), mean_corr)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Neuron ID')
plt.ylabel('Mean Correlation with Other Neurons')
plt.title('Average Correlation of Each Neuron with All Other Neurons')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_neuron_correlation.png'), dpi=150)
plt.close()

print("Correlation analysis completed successfully.")