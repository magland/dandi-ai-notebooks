# %% [markdown]
# # DANDI Archive - Dandiset 001366 Exploration
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.
#
# In this notebook, we will explore Dandiset 001366, titled "Comparison of Approaches for Surface Vessel Diameter and Pulsatility Quantification". This dataset contains movies of a pial vessel of mice used in experiments.
#
# The notebook will guide you through:
#
# 1.  An introduction to the Dandiset, including its name, description, and key metadata.
# 2.  Code to load and explore the dataset's structure.
# 3.  Demonstration of how to access and visualize sample data from NWB files.
# 4.  Examples of common analyses that might be relevant to the dataset's content.
#
# Before you begin, make sure you have the following packages installed:
# ```bash
# pip install dandi pynwb lindi matplotlib seaborn
# ```

# %%
# Use the DANDI API to list all of the assets in the Dandiset
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("001366")
assets = list(dandiset.get_assets())
assets

# %% [markdown]
# The code block above uses the DANDI API to list all assets in the Dandiset.

# %% [markdown]
# ## Loading and Exploring the Dataset Structure
#
# We will now load an NWB file from the Dandiset and explore its structure. We will use the `lindi` and `pynwb` libraries to load the NWB file.
#
# The NWB file we will be exploring is: `sub-031224-M4/sub-031224-M4_ses-03122024-m4-baseline_image.nwb`
# The asset ID is: `2f12bce3-f841-46ca-b928-044269122a59`

# %%
import pynwb
import lindi

# Load https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/2f12bce3-f841-46ca-b928-044269122a59/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

nwb.session_description # (str) the mouse skull was thinned at the area of the middle cerebral artery (MCA) and fitted with a head plate for fixation.
nwb.identifier # (str) Qinwen_6th_March_03122024-m4-baseline
nwb.session_start_time # (datetime) 2024-03-12T01:00:00.000000-04:00
nwb.file_create_date # (datetime) 2025-03-21T10:58:09.704462-04:00
nwb.timestamps_reference_time # (datetime) 2024-03-12T01:00:00.000000-04:00
nwb.experimenter # (List[str]) ["Huang, Qinwen"]
nwb.experiment_description # (str) Vessel diameter and pulsatility measurement.
nwb.institution # (str) University of Rochester
nwb.keywords # (List[str]) ["Vessel diameter, Radon transform, full width at half maximum, vessel pulsation, image analysis"]
nwb.protocol # (str)
nwb.lab # (str)
nwb.subject # (Subject)
nwb.subject.age # (str) P2M
nwb.subject.age__reference # (str) birth
nwb.subject.description # (str) Before the imaging, the mouse was anesthetized with ketamine and xylazine cocktail (80 mg/kg, 10 mg/kg), then retro-orbital injected with fluorescent tracer (0.1 ml, 1%, albumin from Bovine serum 647, Thermo Fisher Scientific catalog: A34785)
nwb.subject.genotype # (str)
nwb.subject.sex # (str) M
nwb.subject.species # (str) Mus musculus
nwb.subject.subject_id # (str) 031224_M4
nwb.subject.weight # (str)
nwb.subject.date_of_birth # (datetime)

Movies = nwb.acquisition["Movies"] # (ImageSeries) 16-bit grayscale movie of a pial vessel
Movies.starting_time # 0 sec
Movies.rate # 30 Hz

# %% [markdown]
# The code block above loads the NWB file and prints some of its metadata. We can see that the NWB file contains a movie of a pial vessel. The movie is stored in the `Movies` object. We can also see the subject's metadata, such as age, sex, and species.

# %% [markdown]
# ## Accessing and Visualizing Sample Data
#
# Now, let's access and visualize some sample data from the NWB file. We will plot the first 5 frames of the movie.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

Movies = nwb.acquisition["Movies"]
data = Movies.data
num_frames = data.shape[0]
print(f"Number of frames: {num_frames}")

# Plot the first 5 frames
num_plots = min(5, num_frames)
fig, axes = plt.subplots(1, num_plots, figsize=(15, 3))
for i in range(num_plots):
    axes[i].imshow(data[i, :, :], cmap='gray')
    axes[i].set_title(f"Frame {i}")
    axes[i].axis('off')

plt.show()

# %% [markdown]
# The code block above plots the first 5 frames of the movie. We can see the vascular structure in these frames.

# %% [markdown]
# ![movie_frames](tmp_scripts/movie_frames.png)
#
# The image above shows a series of five grayscale frames. These frames appear to represent sequential images, possibly from a time-lapse or video sequence, capturing a scene that resembles a biological structure, such as blood vessels.
#
# Analysis:
#
# - These frames likely capture an anatomical or physiological feature, possibly for medical, biological, or research purposes.
# - The use of grayscale and focus on vascularity suggest applications in angiography or other vascular studies.
# - The consistency across frames suggests a stable imaging environment and robust data capturing method, important for accurate longitudinal studies or detailed observation of structures.
#
# Potential Applications:
#
# - Such imaging sequences can be used to monitor blood flow, structural changes, or to assess the impact of treatments in medical or biological research.
#
# Next, let's calculate the mean intensity of each frame and plot it.

# %%
import numpy as np

data = Movies.data
num_frames = min(100, data.shape[0])
print(f"Number of frames: {num_frames}")

# Calculate the mean intensity of each frame for the first 100 frames
mean_intensities = np.mean(data[:num_frames, :, :], axis=(1, 2))

# Plot the mean intensities
plt.figure(figsize=(15, 5))
plt.plot(mean_intensities)
plt.xlabel("Frame")
plt.ylabel("Mean Intensity")
plt.title("Mean Intensity of Each Frame")
plt.show()

# %% [markdown]
# The code block above calculates the mean intensity of each frame and plots it. We can observe the changes in intensity of the movie over time.

# %% [markdown]
# ![mean_intensity](tmp_scripts/mean_intensity.png)
#
# The plot displays the mean intensity of each frame over a sequence of 100 frames.
#
# Data Analysis:
#
# 1. Oscillating Pattern:
#    - The plot exhibits a clear oscillating pattern, indicating a periodic fluctuation in mean intensity across frames.
#    - Peaks and troughs appear consistently throughout the plot, suggesting a repeating cycle.
#
# 2. Amplitude and Frequency:
#    - The amplitude of oscillations is moderate.
#    - The period of oscillation appears consistent, with about 6-7 oscillations within the 100 frames, implying a regular frequency.
#
# Conclusion:
#
# This plot likely represents a scenario where intensity measurements fluctuate periodically, possibly due to regular changes in an external factor affecting intensity (such as lighting conditions, cyclical events in a monitored process, etc.). The consistency in both amplitude and frequency suggests a stable oscillatory process. This could be due to the pulsatility of the blood vessels.