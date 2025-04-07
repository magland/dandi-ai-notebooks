# %% [markdown]
# # Exploration of Dandiset 001176
#
# **Dandiset name:** Cortical acetylcholine dynamics are predicted by cholinergic axon activity and behavior state
#
# **Description:**  
# This dataset contains simultaneous in vivo imaging of acetylcholine sensors (GACh3.0) and GCaMP-labeled axons in mouse cortex alongside eye tracking, pupil size, and behavioral monitoring (running). It enables studies of neuromodulation and brain states.
#
# **Keywords:** acetylcholine, brain states, two-photon imaging, neuromodulator, axon imaging
#
# **Citation:**  
# Reimer, Jacob; Neyhart, Erin (2025) Cortical acetylcholine dynamics... [Data set]. DANDI Archive. https://dandiarchive.org/dandiset/001176/draft
#
# ---
#
# **This notebook was auto-generated via dandi-notebook-gen. It has not been fully verified. Use caution interpreting code, data access, or analyses.**
#
# ---

# %% [markdown]
# ## Accessing Dandiset assets via DANDI API

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001176")
assets = list(dandiset.get_assets())

# %% [markdown]
# ## Example: loading a sample NWB file
# We are loading the session:
# ```
# sub-22713/sub-22713_ses-22713-2-1-Ach-V1_behavior+ophys.nwb
# ```
# (~1.7MB test file with behavior & imaging)

# %%
import lindi
import pynwb

nwb_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/be84b6ff-7016-4ed8-af63-aa0e07c02530/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_url)
nwbfile = pynwb.NWBHDF5IO(file=f, mode='r').read()

# %% [markdown]
# ## Exploring file metadata and content structure

# %%
print("Session description:", nwbfile.session_description)
print("Identifier:", nwbfile.identifier)
print("Start time:", nwbfile.session_start_time)
print("Experimenter:", nwbfile.experimenter)
print("Subject ID:", nwbfile.subject.subject_id)
print("Species:", nwbfile.subject.species)
print("Sex:", nwbfile.subject.sex)
print("Imaging plane descriptions:", list(nwbfile.processing['ophys'].data_interfaces.keys()))

# %% [markdown]
# ## Behavioral data: Eye tracking and pupil size

# %%
from IPython.display import Image, display
display(Image("tmp_scripts/pupil_eye_tracking.png"))

# %% [markdown]
# *This plot shows the tracked pupil radius and eye positions over time.*  
# The pupil trace reveals gradual changes with oscillations, while eye positions show stable tracking interspersed with positional adjustments. This confirms that the behavioral monitoring contains interpretable signals.

# %% [markdown]
# ## Calcium fluorescence trace

# %%
display(Image("tmp_scripts/fluorescence_trace.png"))

# %% [markdown]
# *This trace shows temporal calcium dynamics within a segmented ROI.*  
# An obvious transient increase is visible, typical of neural activation signals. This illustrates how to visualize example calcium traces, though further analysis is needed for scientific interpretation.

# %% [markdown]
# ## Summary images from two-photon imaging

# %%
display(Image("tmp_scripts/summary_images.png"))

# %% [markdown]
# *Left: average fluorescence, Right: pixelwise correlation*  
# Aggregate images reveal cellular/cortical structure with discernible cell bodies as bright spots, suitable for illustration of imaging data.

# %% [markdown]
# ## Segmentation mask example

# %%
display(Image("tmp_scripts/segmentation_masks.png"))

# %% [markdown]
# The segmentation mask shows an isolated, sharply defined ROI identified via image analysis, suitable for illustrating segmentation concepts.

# %% [markdown]
# ## Notes:
# - These examples used a single session for illustration.
# - Loading larger files or full Dandisets may require more compute or time.
# - Advanced analyses, e.g., statistics, event detection, neural modeling, can be performed but are beyond this initial exploration.
#
# ---
#
# **Please verify code/data handling before relying on downstream analyses.**