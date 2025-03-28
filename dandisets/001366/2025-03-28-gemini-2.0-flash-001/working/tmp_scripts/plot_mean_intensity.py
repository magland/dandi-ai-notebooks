import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/2f12bce3-f841-46ca-b928-044269122a59/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

Movies = nwb.acquisition["Movies"]
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
plt.savefig("tmp_scripts/mean_intensity.png")
print("Saved plot to tmp_scripts/mean_intensity.png")
# plt.show()