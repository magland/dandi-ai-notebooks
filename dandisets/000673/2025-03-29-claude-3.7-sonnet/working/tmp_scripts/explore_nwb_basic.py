"""
This script explores the basic structure of an NWB file from Dandiset 000673.
It displays general information about the file, including subject information,
available data types, and key metadata.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import os

# Make sure output directory exists for plots
os.makedirs('tmp_scripts', exist_ok=True)

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000673/assets/9fdbe18f-158f-47c5-ba67-4c56118d6cf5/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information about the dandiset
print("Basic NWB File Information:")
print("=" * 30)
print(f"Session description: {nwb.session_description}")
print(f"NWB identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"File create date: {nwb.file_create_date}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"Keywords: {nwb.keywords}")
print("\n")

# Subject information
print("Subject Information:")
print("=" * 30)
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Species: {nwb.subject.species}")
print("\n")

# Print available groups in the NWB file
print("Available Groups in the NWB File:")
print("=" * 30)
print("Acquisition:")
for name in nwb.acquisition:
    print(f"  - {name}")

print("\nStimulus:")
for name in nwb.stimulus:
    print(f"  - {name}")

print("\nIntervals:")
for name in nwb.intervals:
    print(f"  - {name}")

print("\nUnits Information:")
print(f"  - Number of units: {len(nwb.units.id)}")
print(f"  - Available columns: {nwb.units.colnames}")

print("\nElectrodes Information:")
print(f"  - Number of electrodes: {len(nwb.electrodes.id)}")
print(f"  - Available columns: {nwb.electrodes.colnames}")
print("\n")

# Print a sample of trial information
print("Sample Trial Information:")
print("=" * 30)
num_trials = len(nwb.intervals["trials"]["id"])
print(f"Number of trials: {num_trials}")
print("\nFirst 5 trials:")
for i in range(min(5, num_trials)):
    print(f"\nTrial {i+1}:")
    print(f"  Load: {nwb.intervals['trials']['loads'][i]}")
    print(f"  Probe In/Out: {nwb.intervals['trials']['probe_in_out'][i]}")
    print(f"  Response Accuracy: {nwb.intervals['trials']['response_accuracy'][i]}")
    print(f"  Start Time: {nwb.intervals['trials']['start_time'][i]:.2f}s")
    print(f"  Stop Time: {nwb.intervals['trials']['stop_time'][i]:.2f}s")

# Display spike times for the first unit
print("\nSample Unit Information:")
print("=" * 30)
unit_index = 0
spike_times = nwb.units["spike_times"][unit_index]
print(f"Unit {unit_index} (on electrode {nwb.units['electrodes'].data[unit_index]}):")
print(f"  Number of spikes: {len(spike_times)}")
print(f"  First 5 spike times: {spike_times[:5]}")
print(f"  Mean firing rate: {len(spike_times) / (nwb.intervals['trials']['stop_time'][-1] - nwb.intervals['trials']['start_time'][0]):.2f} Hz")
print(f"  SNR: {nwb.units['waveforms_mean_snr'].data[unit_index]:.2f}")

# Plot the spike count histogram for the first unit
plt.figure(figsize=(10, 5))
plt.hist(spike_times, bins=100)
plt.title(f"Spike Times Histogram for Unit {unit_index}")
plt.xlabel("Time (s)")
plt.ylabel("Spike Count")
plt.tight_layout()
plt.savefig("tmp_scripts/unit_spike_histogram.png")
plt.close()

# Check the stimulus templates
print("\nStimulus Information:")
print("=" * 30)
stimulus_presentation = nwb.stimulus["StimulusPresentation"]
num_stimuli = len(stimulus_presentation.timestamps)
print(f"Number of stimulus presentations: {num_stimuli}")

# Print information about the available images
print("\nImage Templates:")
num_images = 0
image_sizes = []
for k in stimulus_presentation.indexed_images.images.keys():
    image = stimulus_presentation.indexed_images.images[k]
    image_sizes.append(image.data.shape)
    num_images += 1
    if num_images <= 5:  # Only print details for first 5 images
        print(f"  Image {k}: {image.data.shape}")
print(f"Total number of unique images: {num_images}")

# Plot the trial distribution by load condition
loads = nwb.intervals["trials"]["loads"][:]
unique_loads, load_counts = np.unique(loads, return_counts=True)
plt.figure(figsize=(8, 5))
plt.bar(unique_loads, load_counts)
plt.title("Distribution of Trials by Memory Load")
plt.xlabel("Memory Load")
plt.ylabel("Number of Trials")
plt.xticks(unique_loads)
plt.tight_layout()
plt.savefig("tmp_scripts/trial_load_distribution.png")
plt.close()

# Plot response accuracy by memory load
response_accuracy = nwb.intervals["trials"]["response_accuracy"][:]
accuracy_by_load = {}
for load in unique_loads:
    load_indices = np.where(loads == load)[0]
    accuracy = np.mean(response_accuracy[load_indices]) * 100
    accuracy_by_load[load] = accuracy

plt.figure(figsize=(8, 5))
plt.bar(list(accuracy_by_load.keys()), list(accuracy_by_load.values()))
plt.title("Response Accuracy by Memory Load")
plt.xlabel("Memory Load")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(list(accuracy_by_load.keys()))
plt.tight_layout()
plt.savefig("tmp_scripts/accuracy_by_load.png")
plt.close()

# Display electrode locations
electrode_locations = nwb.electrodes["location"].data[:]
unique_locations, location_counts = np.unique(electrode_locations, return_counts=True)
plt.figure(figsize=(10, 5))
plt.bar(range(len(unique_locations)), location_counts)
plt.title("Distribution of Electrodes by Location")
plt.xlabel("Location")
plt.ylabel("Number of Electrodes")
plt.xticks(range(len(unique_locations)), unique_locations, rotation=45, ha='right')
plt.tight_layout()
plt.savefig("tmp_scripts/electrode_locations.png")
plt.close()

print("\nPlots saved to tmp_scripts directory.")
print("Script execution completed.")