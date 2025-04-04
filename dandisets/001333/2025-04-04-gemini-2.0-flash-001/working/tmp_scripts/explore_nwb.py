import tempfile
from dandi.dandiapi import DandiAPIClient
import h5py
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import os

# Script to explore the contents of an NWB file in Dandiset 001333
# using the DANDI API and h5py.

client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
assets = list(dandiset.get_assets())

nwb_file_path = "sub-healthy-simulated-beta/sub-healthy-simulated-beta_ses-218_ecephys.nwb"
asset = next(asset for asset in assets if asset.path == nwb_file_path)

# Download the NWB file to a temporary location
download_url = asset.download_url

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nwb") as tmpfile:
        urllib.request.urlretrieve(download_url, tmpfile.name)
        temp_nwb_path = tmpfile.name

    with h5py.File(temp_nwb_path, 'r') as nwbfile:
        # Get the data and timestamps
        data = nwbfile['processing']['ecephys']['LFP']['Beta_Band_Voltage']['data']
        timestamps = nwbfile['processing']['ecephys']['LFP']['Beta_Band_Voltage']['timestamps']

        # Extract a small subset of the data (e.g., first 1000 samples)
        subset_size = 1000
        data_subset = data[:subset_size]
        timestamps_subset = timestamps[:subset_size]

        # Plot the data subset
        plt.figure(figsize=(10, 5))  # Adjust figure size for better visualization
        plt.plot(timestamps_subset, data_subset)
        plt.xlabel("Time (s)")
        plt.ylabel("Beta Band Voltage (ARV)")
        plt.title("Sample Beta Band Voltage Data")
        plt.grid(True)

        # Save the plot to a file
        plot_filename = "beta_band_voltage.png"
        plt.savefig(os.path.join("tmp_scripts", plot_filename))
        print(f"Plot saved to tmp_scripts/{plot_filename}")

except Exception as e:
    print(f"Error loading NWB file: {e}")
finally:
    # Clean up the temporary file
    if 'temp_nwb_path' in locals():
        os.remove(temp_nwb_path)