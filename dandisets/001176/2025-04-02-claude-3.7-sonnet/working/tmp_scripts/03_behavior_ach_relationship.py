# Script to examine the relationship between behavioral states and acetylcholine activity
# This script explores how pupil size and locomotion correlate with acetylcholine sensor signals

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/babeee4c-bb8f-4d0b-b898-3edf99244f25/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Load pupil radius data
print("Loading pupil data...")
pupil_tracking = nwb.acquisition["PupilTracking"]
pupil_radius = pupil_tracking["pupil_raw_radius"]
pupil_timestamps = pupil_radius.timestamps[:]
pupil_data = pupil_radius.data[:]

# Load treadmill velocity data
print("Loading treadmill data...")
treadmill = nwb.acquisition["treadmill_velocity"]
treadmill_timestamps = treadmill.timestamps[:]
treadmill_data = treadmill.data[:]

# Load fluorescence data (acetylcholine sensor)
print("Loading fluorescence data...")
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"]
fluor_timestamps = fluorescence.timestamps[:]
fluor_data = fluorescence.data[:, 0]  # Just the first ROI

# Find the common time window for all three datasets
print("Aligning datasets...")
start_time = max(pupil_timestamps[0], 
                 np.nanmin(treadmill_timestamps), 
                 fluor_timestamps[0])
end_time = min(pupil_timestamps[-1], 
               np.nanmax(treadmill_timestamps), 
               fluor_timestamps[-1])

# Filter data to common time window
pupil_mask = (pupil_timestamps >= start_time) & (pupil_timestamps <= end_time)
treadmill_mask = (treadmill_timestamps >= start_time) & (treadmill_timestamps <= end_time)
fluor_mask = (fluor_timestamps >= start_time) & (fluor_timestamps <= end_time)

# Get filtered data
pupil_times_filtered = pupil_timestamps[pupil_mask]
pupil_data_filtered = pupil_data[pupil_mask]

# For treadmill, we also need to filter out NaN values
valid_treadmill_mask = ~np.isnan(treadmill_timestamps) & ~np.isnan(treadmill_data)
combined_treadmill_mask = treadmill_mask & valid_treadmill_mask
treadmill_times_filtered = treadmill_timestamps[combined_treadmill_mask]
treadmill_data_filtered = treadmill_data[combined_treadmill_mask]

fluor_times_filtered = fluor_timestamps[fluor_mask]
fluor_data_filtered = fluor_data[fluor_mask]

# Create a common time base for resampling (using fluorescence timestamps as reference)
common_times = fluor_times_filtered

# Resample pupil and treadmill data to match fluorescence timestamps
print("Resampling data to common timebase...")
# Pupil resampling
pupil_interp = interp1d(pupil_times_filtered, pupil_data_filtered, 
                        bounds_error=False, fill_value="extrapolate")
pupil_resampled = pupil_interp(common_times)

# Treadmill resampling (only if we have valid data)
if len(treadmill_times_filtered) > 0:
    treadmill_interp = interp1d(treadmill_times_filtered, treadmill_data_filtered, 
                                bounds_error=False, fill_value="extrapolate")
    treadmill_resampled = treadmill_interp(common_times)
else:
    treadmill_resampled = np.zeros_like(common_times)

# Plot time series of all three variables
print("Plotting time series...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Subsample for plotting to avoid overwhelming the figure
# Take every Nth point
N = 100
plot_indices = np.arange(0, len(common_times), N)

# Plot pupil radius
ax1.plot(common_times[plot_indices], pupil_resampled[plot_indices])
ax1.set_ylabel("Pupil Radius (pixels)")
ax1.set_title("Pupil Size Over Time")

# Plot treadmill velocity
ax2.plot(common_times[plot_indices], treadmill_resampled[plot_indices])
ax2.set_ylabel("Velocity (units/s)")
ax2.set_title("Locomotion (Treadmill Velocity) Over Time")

# Plot fluorescence
ax3.plot(common_times[plot_indices], fluor_data_filtered[plot_indices])
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Fluorescence (a.u.)")
ax3.set_title("Acetylcholine Sensor Activity Over Time")

plt.tight_layout()
plt.savefig("tmp_scripts/aligned_time_series.png", dpi=300)
print("Time series figure saved as tmp_scripts/aligned_time_series.png")

# Plot correlations between variables
print("Analyzing correlations...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Correlation between pupil size and ACh
ax1.scatter(pupil_resampled[plot_indices], fluor_data_filtered[plot_indices], alpha=0.5)
ax1.set_xlabel("Pupil Radius (pixels)")
ax1.set_ylabel("ACh Fluorescence (a.u.)")
ax1.set_title("Pupil Size vs. ACh Activity")

# Add correlation line and coefficient
slope, intercept, r_value, p_value, std_err = stats.linregress(
    pupil_resampled[plot_indices], fluor_data_filtered[plot_indices])
x_vals = np.array([min(pupil_resampled[plot_indices]), max(pupil_resampled[plot_indices])])
y_vals = intercept + slope * x_vals
ax1.plot(x_vals, y_vals, 'r-')
ax1.annotate(f"r = {r_value:.2f}, p = {p_value:.4f}", 
             xy=(0.05, 0.95), xycoords='axes fraction',
             verticalalignment='top')

# Correlation between locomotion and ACh
ax2.scatter(treadmill_resampled[plot_indices], fluor_data_filtered[plot_indices], alpha=0.5)
ax2.set_xlabel("Treadmill Velocity (units/s)")
ax2.set_ylabel("ACh Fluorescence (a.u.)")
ax2.set_title("Locomotion vs. ACh Activity")

# Add correlation line and coefficient
slope, intercept, r_value, p_value, std_err = stats.linregress(
    treadmill_resampled[plot_indices], fluor_data_filtered[plot_indices])
x_vals = np.array([min(treadmill_resampled[plot_indices]), max(treadmill_resampled[plot_indices])])
y_vals = intercept + slope * x_vals
ax2.plot(x_vals, y_vals, 'r-')
ax2.annotate(f"r = {r_value:.2f}, p = {p_value:.4f}", 
             xy=(0.05, 0.95), xycoords='axes fraction',
             verticalalignment='top')

plt.tight_layout()
plt.savefig("tmp_scripts/correlation_analysis.png", dpi=300)
print("Correlation figure saved as tmp_scripts/correlation_analysis.png")

# Create a categorical analysis by binning locomotion
print("Creating categorical analysis...")
# Define locomotion state (moving vs stationary)
velocity_threshold = 1.0  # units/s
is_moving = treadmill_resampled > velocity_threshold

# Calculate mean ACh fluorescence for moving vs stationary
mean_ach_moving = np.mean(fluor_data_filtered[is_moving])
mean_ach_stationary = np.mean(fluor_data_filtered[~is_moving])
sem_ach_moving = stats.sem(fluor_data_filtered[is_moving])
sem_ach_stationary = stats.sem(fluor_data_filtered[~is_moving])

# Calculate statistical significance
ttest_result = stats.ttest_ind(
    fluor_data_filtered[is_moving], 
    fluor_data_filtered[~is_moving],
    equal_var=False  # Welch's t-test, not assuming equal variance
)

# Plot bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(['Stationary', 'Moving'], 
               [mean_ach_stationary, mean_ach_moving],
               yerr=[sem_ach_stationary, sem_ach_moving])
plt.ylabel('Mean ACh Fluorescence (a.u.)')
plt.title(f'ACh Activity by Locomotion State\n(p = {ttest_result.pvalue:.4f})')
# Add sample size information
plt.annotate(f"n = {np.sum(~is_moving)}", xy=(0, mean_ach_stationary), 
             xytext=(0, mean_ach_stationary - 5*sem_ach_stationary),
             ha='center')
plt.annotate(f"n = {np.sum(is_moving)}", xy=(1, mean_ach_moving), 
             xytext=(1, mean_ach_moving - 5*sem_ach_moving),
             ha='center')

plt.tight_layout()
plt.savefig("tmp_scripts/behavioral_state_analysis.png", dpi=300)
print("Behavioral state analysis saved as tmp_scripts/behavioral_state_analysis.png")