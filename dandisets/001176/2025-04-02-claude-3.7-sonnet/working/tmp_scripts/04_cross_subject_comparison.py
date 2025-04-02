# Script to compare data across different subjects
# This script loads data from a different subject and performs similar analyses
# to compare findings across subjects

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

print("Loading NWB file for subject 23893...")
# Load a different subject's NWB file (choosing subject 23893 from V1)
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/38cc792e-0f9b-4255-a57f-78fd6c1315a8/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic metadata about this subject
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Session ID: {nwb.identifier}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Experiment description: {nwb.experiment_description}")

# Load pupil radius data
print("\nLoading pupil data...")
if "PupilTracking" in nwb.acquisition:
    pupil_tracking = nwb.acquisition["PupilTracking"]
    pupil_radius = pupil_tracking["pupil_raw_radius"]
    pupil_timestamps = pupil_radius.timestamps[:]
    pupil_data = pupil_radius.data[:]
    print(f"Pupil data shape: {pupil_data.shape}")
    print(f"Pupil timestamps range: {pupil_timestamps[0]} to {pupil_timestamps[-1]}")
else:
    print("No pupil tracking data available for this subject")
    pupil_data = None
    pupil_timestamps = None

# Load treadmill velocity data
print("\nLoading treadmill data...")
if "treadmill_velocity" in nwb.acquisition:
    treadmill = nwb.acquisition["treadmill_velocity"]
    treadmill_timestamps = treadmill.timestamps[:]
    treadmill_data = treadmill.data[:]
    print(f"Treadmill data shape: {treadmill_data.shape}")
    print(f"Treadmill timestamps range: {np.nanmin(treadmill_timestamps)} to {np.nanmax(treadmill_timestamps)}")
else:
    print("No treadmill velocity data available for this subject")
    treadmill_data = None
    treadmill_timestamps = None

# Load fluorescence data (acetylcholine sensor)
print("\nLoading fluorescence data...")
if "ophys" in nwb.processing and "Fluorescence" in nwb.processing["ophys"].data_interfaces:
    fluorescence = nwb.processing["ophys"]["Fluorescence"]
    # Get the first RoiResponseSeries
    roi_series_name = list(fluorescence.roi_response_series.keys())[0]
    fluor_series = fluorescence.roi_response_series[roi_series_name]
    fluor_timestamps = fluor_series.timestamps[:]
    fluor_data = fluor_series.data[:, 0]  # Just the first ROI
    print(f"Fluorescence data shape: {fluor_series.data.shape}")
    print(f"Fluorescence timestamps range: {fluor_timestamps[0]} to {fluor_timestamps[-1]}")
else:
    print("No fluorescence data available for this subject")
    fluor_data = None
    fluor_timestamps = None

# If we have all three data types, create plots similar to the previous subject
if pupil_data is not None and treadmill_data is not None and fluor_data is not None:
    # Find the common time window for all three datasets
    print("\nAligning datasets...")
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
    ax1.set_title("Pupil Size Over Time - Subject 23893")

    # Plot treadmill velocity
    ax2.plot(common_times[plot_indices], treadmill_resampled[plot_indices])
    ax2.set_ylabel("Velocity (units/s)")
    ax2.set_title("Locomotion (Treadmill Velocity) Over Time - Subject 23893")

    # Plot fluorescence
    ax3.plot(common_times[plot_indices], fluor_data_filtered[plot_indices])
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Fluorescence (a.u.)")
    ax3.set_title("Acetylcholine Sensor Activity Over Time - Subject 23893")

    plt.tight_layout()
    plt.savefig("tmp_scripts/subject23893_time_series.png", dpi=300)
    print("Time series figure saved as tmp_scripts/subject23893_time_series.png")

    # Create a categorical analysis by binning locomotion
    print("Creating behavioral state analysis...")
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
    plt.title(f'ACh Activity by Locomotion State - Subject 23893\n(p = {ttest_result.pvalue:.4f})')
    
    # Add sample size information
    plt.annotate(f"n = {np.sum(~is_moving)}", xy=(0, mean_ach_stationary), 
                 xytext=(0, mean_ach_stationary - 5*sem_ach_stationary),
                 ha='center')
    plt.annotate(f"n = {np.sum(is_moving)}", xy=(1, mean_ach_moving), 
                 xytext=(1, mean_ach_moving - 5*sem_ach_moving),
                 ha='center')

    plt.tight_layout()
    plt.savefig("tmp_scripts/subject23893_behavioral_analysis.png", dpi=300)
    print("Behavioral state analysis saved as tmp_scripts/subject23893_behavioral_analysis.png")
else:
    print("\nCannot create comparative plots because one or more data types are missing.")

# Examine the summary images for this subject
print("\nExamining summary images...")
if "ophys" in nwb.processing and "SummaryImages_chan1" in nwb.processing["ophys"].data_interfaces:
    summary_images = nwb.processing["ophys"]["SummaryImages_chan1"]
    # Get average and correlation images
    avg_img = summary_images["average"].data[:]
    
    # Normalize for better visualization
    avg_img_norm = (avg_img - avg_img.min()) / (avg_img.max() - avg_img.min())
    
    # Plot the average image
    plt.figure(figsize=(8, 8))
    plt.imshow(avg_img_norm, cmap='gray')
    plt.title('Average Image - Subject 23893')
    plt.axis('off')
    plt.savefig("tmp_scripts/subject23893_average_image.png", dpi=300)
    print("Average image saved as tmp_scripts/subject23893_average_image.png")
else:
    print("No summary images available for this subject.")