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
num_frames = data.shape[0]
print(f"Number of frames: {num_frames}")

# Plot the first 5 frames
num_plots = min(5, num_frames)
fig, axes = plt.subplots(1, num_plots, figsize=(15, 3))
for i in range(num_plots):
    axes[i].imshow(data[i, :, :], cmap='gray')
    axes[i].set_title(f"Frame {i}")
    axes[i].axis('off')

plt.savefig("tmp_scripts/movie_frames.png")
print("Saved plot to tmp_scripts/movie_frames.png")
# plt.show()