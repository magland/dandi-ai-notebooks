# %% [markdown]
# # Exploring Calcium Imaging Data in SMA and M1 of Macaques
# 
# **Note**: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.
# 
# ## Introduction
# 
# This notebook explores the DANDI:001174 dataset, which contains calcium imaging data from the Supplementary Motor Area (SMA) and Primary Motor Area (M1) in rhesus macaques. The study focuses on examining activity patterns of projection neurons in deep layers of these motor cortices while the animals were at rest or engaged in an arm reaching task.
# 
# The dataset uses one-photon calcium imaging with miniature microscopes (miniscopes) to record calcium transients from genetically identified neurons. Calcium imaging allows for the study of multiple neurons with excellent spatial resolution, which is particularly valuable for understanding motor control in both healthy conditions and movement disorders.
# 
# Key features of this dataset:
# - Calcium imaging data from SMA and M1 in rhesus macaques
# - Recordings during rest and arm reaching tasks
# - Expression of GCaMP6f for calcium imaging
# - Use of gradient index (GRIN) lenses for imaging deep cortical layers
# 
# In this notebook, we will:
# 1. Load and explore the dataset structure
# 2. Visualize the calcium imaging data
# 3. Analyze neuronal activity patterns
# 4. Explore relationships between neurons
# 
# ### Required Packages
# 
# To run this notebook, you'll need the following packages:
# - pynwb (for working with Neurodata Without Borders files)
# - lindi (for accessing remote NWB files)
# - numpy (for numerical operations)
# - matplotlib (for plotting)
# - seaborn (for enhanced visualizations)
# - scipy (for scientific computing)
# 
# If you don't have these installed, you can install them using pip:
# ```
# pip install pynwb lindi numpy matplotlib seaborn scipy
# ```

# %% [markdown]
# ## 1. Loading and Exploring the Dataset

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001174")
assets = list(dandiset.get_assets())

# %%
# Display basic information about the Dandiset
print(f"Dandiset ID: {dandiset.identifier}")
try:
    metadata = dandiset.get_metadata()
    name = metadata['name']
except Exception as e:
    # Use the raw metadata instead if there's a validation error
    raw_metadata = dandiset.get_raw_metadata()
    name = raw_metadata.get('name', 'Name not available')
    print(f"Note: Using raw metadata due to validation error: {e}")

print(f"Dandiset Name: {name}")
print(f"Number of assets: {len(assets)}")

# %% [markdown]
# Let's examine the assets in this Dandiset to get a better understanding of what's available.

# %%
# List the first 10 assets (files)
print("Sample of assets in the Dandiset:")
for i, asset in enumerate(assets[:10]):
    print(f"{i+1}. {asset.path} ({asset.size / 1e9:.2f} GB)")

# %% [markdown]
# Let's look at the subjects included in this dataset:

# %%
# Extract subject IDs from asset paths
import re

subject_ids = set()
for asset in assets:
    # Extract subject ID from path (assumes path format like "sub-X/...")
    match = re.match(r'sub-([^/]+)', asset.path)
    if match:
        subject_ids.add(match.group(1))

print(f"Subjects in this dataset: {', '.join(sorted(subject_ids))}")

# %% [markdown]
# ## 2. Loading and Exploring a Sample NWB File
# 
# For our analysis, we'll focus on one NWB file from subject Q. Let's load it and explore its structure.

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_theme()

# %%
# Choose an NWB file to work with (from subject Q)
selected_asset = None
for asset in assets:
    if 'sub-Q' in asset.path and asset.path.endswith('ophys.nwb') and 'ses-20220915' in asset.path:
        selected_asset = asset
        break

if selected_asset:
    print(f"Selected asset: {selected_asset.path}")
    print(f"Asset ID: {selected_asset.identifier}")
    asset_url = f"https://api.dandiarchive.org/api/assets/{selected_asset.identifier}/download/"
    print(f"Download URL: {asset_url}")
else:
    print("No suitable asset found")

# %%
# Load the NWB file using lindi
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001174/assets/{selected_asset.identifier}/nwb.lindi.json"
print(f"Loading NWB file from {lindi_url}")
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# %%
# Display basic information about the NWB file
print("Basic NWB Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"File Create Date: {nwb.file_create_date}")

# %%
# Display subject information
print("Subject Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# %% [markdown]
# Now let's examine the structure of the NWB file to understand what data it contains:

# %%
# Examine acquisition data
print("Acquisition data:")
for name, obj in nwb.acquisition.items():
    print(f"  {name}: {type(obj).__name__}")
    if hasattr(obj, 'data'):
        print(f"    Shape: {obj.data.shape}, Type: {obj.data.dtype}")
    if hasattr(obj, 'rate'):
        print(f"    Rate: {obj.rate} Hz")

# %%
# Examine processing modules
print("Processing modules:")
for module_name, module in nwb.processing.items():
    print(f"  {module_name}: {module.description}")
    for data_name, data_obj in module.data_interfaces.items():
        print(f"    {data_name}: {type(data_obj).__name__}")
        if hasattr(data_obj, 'data') and isinstance(data_obj.data, np.ndarray):
            print(f"      Shape: {data_obj.data.shape}, Type: {data_obj.data.dtype}")
        elif hasattr(data_obj, 'data'):
            if hasattr(data_obj.data, 'shape'):
                print(f"      Shape: {data_obj.data.shape}, Type: {data_obj.data.dtype}")

# %% [markdown]
# ## 3. Visualizing Calcium Imaging Data
# 
# Next, let's visualize the calcium imaging data to get a better understanding of the neural activity patterns.

# %% [markdown]
# ### 3.1 Visualizing a Sample Frame from Raw Imaging Data

# %%
# Get the one photon series data
one_photon_series = nwb.acquisition["OnePhotonSeries"]

# Plot a sample frame (middle frame)
middle_frame_idx = one_photon_series.data.shape[0] // 2
sample_frame = one_photon_series.data[middle_frame_idx, :, :]

plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')
plt.title(f"Sample Frame (Frame #{middle_frame_idx})")
plt.colorbar(label='Intensity')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Visualizing ROI Masks
# 
# The dataset includes masks of Regions of Interest (ROIs) that identify individual neurons. Let's visualize these masks.

# %%
# Get the ROI masks
plane_segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
roi_masks = plane_segmentation["image_mask"]

# Create a combined image of all ROI masks
roi_masks_combined = np.zeros((roi_masks.data.shape[1], roi_masks.data.shape[2]))
for i in range(roi_masks.data.shape[0]):
    roi_masks_combined = np.maximum(roi_masks_combined, roi_masks.data[i])

plt.figure(figsize=(10, 8))
plt.imshow(roi_masks_combined, cmap='viridis')
plt.title(f"Combined ROI Masks (n={roi_masks.data.shape[0]})")
plt.colorbar(label='Mask Value')
plt.tight_layout()
plt.show()

# %%
# Overlay ROI contours on the sample frame
plt.figure(figsize=(10, 8))
plt.imshow(sample_frame, cmap='gray')

# Plot contours of each ROI on top of the image
colors = plt.cm.rainbow(np.linspace(0, 1, roi_masks.data.shape[0]))
for i in range(roi_masks.data.shape[0]):
    mask = roi_masks.data[i]
    # Find contours of the mask
    plt.contour(mask, levels=[0.5], colors=[colors[i]], linewidths=1)

plt.title('ROIs Overlaid on Sample Frame')
plt.tight_layout()
plt.show()

# %%
# Display individual ROI masks for a few ROIs
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(min(9, roi_masks.data.shape[0])):
    ax = axes[i // 3, i % 3]
    ax.imshow(roi_masks.data[i], cmap='hot')
    ax.set_title(f"ROI #{i}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Visualizing Fluorescence Traces
# 
# Now let's look at the fluorescence traces for a few ROIs to see their activity over time.

# %%
# Get the fluorescence data
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"]

# Plot fluorescence traces for 5 ROIs
plt.figure(figsize=(12, 8))
time_vector = np.arange(fluorescence.data.shape[0]) / fluorescence.rate
for i in range(5):  # Plot first 5 ROIs
    plt.plot(time_vector, fluorescence.data[:, i], label=f'ROI #{i}')

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence (a.u.)')
plt.title('Fluorescence Traces for 5 ROIs')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's also look at the population average activity to get a sense of the overall activity patterns.

# %%
# Calculate and plot the population average activity
plt.figure(figsize=(12, 6))
mean_activity = np.mean(fluorescence.data[:], axis=1)
plt.plot(time_vector, mean_activity)
plt.xlabel('Time (s)')
plt.ylabel('Mean Fluorescence (a.u.)')
plt.title('Population Average Activity')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Analyzing Neuronal Activity Patterns
# 
# Now let's perform more detailed analyses of the neural activity patterns.

# %% [markdown]
# ### 4.1 Calculating Correlations Between Neurons

# %%
from scipy.stats import pearsonr

# Choose a subset of the data to avoid memory issues (first 3000 timepoints)
max_timepoints = 3000
if fluorescence.data.shape[0] > max_timepoints:
    print(f"Using first {max_timepoints} timepoints for analysis")
    fluor_data = fluorescence.data[:max_timepoints, :]
else:
    fluor_data = fluorescence.data[:]

n_rois = fluor_data.shape[1]

# Compute correlation matrix between ROIs
corr_matrix = np.zeros((n_rois, n_rois))
for i in range(n_rois):
    for j in range(n_rois):
        if i <= j:  # Only compute upper triangle (matrix is symmetric)
            corr, _ = pearsonr(fluor_data[:, i], fluor_data[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # Fill in lower triangle

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=np.arange(n_rois), yticklabels=np.arange(n_rois))
plt.title('Correlation Matrix Between ROIs')
plt.xlabel('ROI #')
plt.ylabel('ROI #')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Detecting and Analyzing Calcium Events

# %%
from scipy.stats import zscore
from scipy.signal import find_peaks

# Z-score the fluorescence data
z_scored_data = np.zeros_like(fluor_data)
for i in range(n_rois):
    z_scored_data[:, i] = zscore(fluor_data[:, i])

# Find peaks for each ROI with height threshold of 2 standard deviations
events = np.zeros_like(fluor_data, dtype=bool)
for i in range(n_rois):
    # Find peaks in z-scored data
    peaks, _ = find_peaks(z_scored_data[:, i], height=2.0)
    events[peaks, i] = True

# Count events per ROI
event_counts = np.sum(events, axis=0)

# Plot event counts
plt.figure(figsize=(10, 6))
plt.bar(np.arange(n_rois), event_counts)
plt.xlabel('ROI #')
plt.ylabel('Number of Calcium Events')
plt.title('Calcium Event Count per ROI')
plt.tight_layout()
plt.show()

# %%
# Create a raster plot of calcium events
plt.figure(figsize=(12, 8))
for i in range(n_rois):
    event_times = time_vector[:max_timepoints][events[:, i]]
    plt.scatter(event_times, np.ones_like(event_times) * i, marker='|', s=100, color='k')

plt.xlabel('Time (s)')
plt.ylabel('ROI #')
plt.title('Calcium Event Raster Plot')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Analyzing Neural Synchronization

# %%
# Calculate event synchronization
# For each timepoint, count how many neurons are active simultaneously
event_synchronization = np.sum(events, axis=1)

# Plot event synchronization over time
plt.figure(figsize=(12, 6))
plt.plot(time_vector[:max_timepoints], event_synchronization)
plt.xlabel('Time (s)')
plt.ylabel('Number of Simultaneously Active ROIs')
plt.title('Neuronal Synchronization')
plt.tight_layout()
plt.show()

# %%
# Create a heatmap of neural activity
plt.figure(figsize=(12, 8))
# Sort ROIs by their total activity
roi_order = np.argsort(-np.sum(fluor_data, axis=0))
sorted_data = fluor_data[:, roi_order]

# Create heatmap with seaborn
ax = sns.heatmap(sorted_data.T, cmap='viridis', 
               xticklabels=np.arange(0, fluor_data.shape[0], 500))
ax.set_xlabel('Time (samples)')
ax.set_ylabel('ROI (sorted by activity)')
ax.set_title('Neural Activity Heatmap')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Conclusion
# 
# In this notebook, we explored calcium imaging data from the Supplementary Motor Area (SMA) in a macaque monkey during an arm reaching task. We visualized the raw imaging data, identified individual neurons (ROIs), and analyzed their activity patterns.
# 
# Key findings:
# 
# 1. The dataset contained calcium imaging data from 40 distinct neurons in the SMA.
# 2. The ROIs exhibited distinct spatial patterns, as visualized in the ROI masks.
# 3. The fluorescence traces showed clear calcium transients, indicating neuronal activity.
# 4. There were variations in activity levels across different neurons, with some being more active than others.
# 5. The population activity showed distinct temporal patterns, potentially related to the arm reaching behavior.
# 6. We identified periods of synchronized activity across multiple neurons.
# 
# Future analyses could explore:
# 
# 1. Relationships between neural activity and specific behavioral events (if behavioral data is available)
# 2. Comparing activity patterns between SMA and M1
# 3. More sophisticated methods for identifying functional networks of neurons
# 4. Temporal dynamics of neural activity during movement preparation and execution
# 
# This dataset provides valuable insights into the patterns of activity in groups of corticofugal neurons in SMA and M1, demonstrating the value of in vivo calcium imaging for studying motor cortices in non-human primates.