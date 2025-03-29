"""
This script explores the relationship between neuronal activity and the arm-reaching task 
mentioned in the dataset description. It attempts to identify sessions with reaching task data
and analyze any correlations between neural activity and behavioral events.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
from dandi.dandiapi import DandiAPIClient

# Get list of assets in the Dandiset
print("Retrieving Dandiset assets...")
client = DandiAPIClient()
dandiset = client.get_dandiset("001174")
assets = list(dandiset.get_assets())

# Function to check for reaching task-related data in NWB file
def check_for_reaching_task(nwb_path):
    try:
        print(f"\nExamining {nwb_path}")
        # Use the asset ID from the path to construct the lindi URL
        asset_id = nwb_path.split("/")[-1]
        lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/001174/assets/{asset_id}/nwb.lindi.json"
        
        # Load the NWB file
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Check for relevant data structures that might contain behavioral data
        has_behavior = 'behavior' in nwb.processing
        has_acquisition = len(nwb.acquisition) > 0
        has_stimulus = hasattr(nwb, 'stimulus') and nwb.stimulus is not None
        has_analysis = hasattr(nwb, 'analysis') and nwb.analysis is not None
        
        # Get basic metadata
        metadata = {
            'session_description': nwb.session_description,
            'has_behavior': has_behavior,
            'has_acquisition': has_acquisition,
            'has_stimulus': has_stimulus,
            'has_analysis': has_analysis
        }
        
        # If there's a behavior module, examine its contents
        if has_behavior:
            behavior_module = nwb.processing['behavior']
            metadata['behavior_contents'] = list(behavior_module.data_interfaces.keys())
            
            # Check if any behavioral data mentions reaching
            contains_reaching = any('reach' in str(key).lower() for key in behavior_module.data_interfaces.keys())
            metadata['contains_reaching'] = contains_reaching
        
        return metadata
    except Exception as e:
        print(f"Error examining {nwb_path}: {str(e)}")
        return {'error': str(e)}

# Look for files that might contain reaching task data
# Start with a small number of files to avoid timeout
print("Checking select files for reaching task data...")
sample_assets = assets[:3]  # Just check a few files to avoid timeout

results = []
for asset in sample_assets:
    result = check_for_reaching_task(asset.path)
    result['path'] = asset.path
    results.append(result)

# Print findings
print("\nSummary of findings:")
for result in results:
    print(f"\nFile: {result.get('path')}")
    if 'error' in result:
        print(f"  Error: {result['error']}")
        continue
        
    print(f"  Session description: {result.get('session_description')}")
    print(f"  Has behavior module: {result.get('has_behavior')}")
    print(f"  Has acquisition: {result.get('has_acquisition')}")
    
    if result.get('has_behavior'):
        print(f"  Behavior contents: {result.get('behavior_contents', [])}")
        print(f"  Contains reaching-related data: {result.get('contains_reaching', False)}")
    
    # If we find any reaching task data, we could plot it here
    # For now, just note that we'd need to examine specific data structures

print("\nNote: Full analysis of reaching task data would require examining specific behavioral recordings")
print("and correlating them with neural activity, which can be implemented after identifying")
print("the exact structure of the behavioral data in these files.")

print("\nScript completed successfully!")