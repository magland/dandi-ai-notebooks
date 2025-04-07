"""
Exploratory script: Connect to a remote PESD NWB file, list top-level groups, acquisitions, and processing modules.
Does NOT load all data arrays to avoid timeouts.
This will inform detailed visualization and analysis steps.
"""

import h5py

nwb_url = "tmp_scripts/sample_file.nwb"

def list_group(group, indent=0):
    prefix = '  ' * indent
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{prefix}[Group] {key}")
            list_group(item, indent +1)
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            print(f"{prefix}[Dataset] {key} | shape: {shape}, dtype: {dtype}")

try:
    with h5py.File(nwb_url, 'r') as f:
        print("Opened remote NWB file successfully.")
        print("\n--- Top-level keys ---")
        for key in f.keys():
            print(f"{key}")

        # Explore acquisition
        if 'acquisition' in f:
            print("\n--- Acquisition ---")
            list_group(f['acquisition'])
        else:
            print("\nNo 'acquisition' group found.")

        # Explore processing
        if 'processing' in f:
            print("\n--- Processing ---")
            list_group(f['processing'])
        else:
            print("\nNo 'processing' group found.")

except Exception as e:
    print(f"Error opening or exploring NWB file: {e}")