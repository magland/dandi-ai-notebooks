"""
This script retrieves information about multiple NWB files in the Dandiset
to identify files with different Pulse Repetition Frequencies (PRFs) for comparison.
"""
from dandi.dandiapi import DandiAPIClient
import json

# Initialize DANDI API client
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")

# Get assets
print("Retrieving assets...")
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} assets")

# Extract relevant information
file_info = []
for asset in assets:
    path = asset.path
    size = asset.size
    asset_id = asset.identifier
    
    # Check for subject and PRF information in file path
    subject = None
    if 'sub-' in path:
        subject = path.split('/')[0].replace('sub-', '')
    
    # Add to our list
    file_info.append({
        'path': path,
        'subject': subject,
        'size': size,
        'asset_id': asset_id,
        'url': f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    })

# Sort by subject
file_info.sort(key=lambda x: (x['subject'], x['path']))

# Print information
print("\nAvailable Files by Subject:")
current_subject = None
for info in file_info:
    if info['subject'] != current_subject:
        current_subject = info['subject']
        print(f"\nSubject: {current_subject}")
    print(f"  {info['path']} ({info['size']/1024:.1f} KB)")

# Save to JSON file for later use
with open('tmp_scripts/file_info.json', 'w') as f:
    json.dump(file_info, f, indent=2)

print("\nFile information saved to tmp_scripts/file_info.json")