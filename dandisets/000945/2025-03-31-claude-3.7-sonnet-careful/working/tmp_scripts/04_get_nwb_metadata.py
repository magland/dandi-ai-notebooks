"""
This script examines metadata from a few NWB files to identify 
Pulse Repetition Frequency (PRF) values and other important parameters.
"""
import pynwb
import lindi
import json
import numpy as np

# Load file info
with open('tmp_scripts/file_info.json', 'r') as f:
    file_info = json.load(f)

# Select a few files from different subjects to examine
subjects = ['BH497', 'BH498', 'BH506', 'BH512']
subject_files = {}
for subject in subjects:
    subject_files[subject] = [info for info in file_info if info['subject'] == subject]

# Select one file from each subject for examination
sample_files = []
for subject, files in subject_files.items():
    if files:
        sample_files.append(files[0])

print(f"Examining {len(sample_files)} sample files for metadata")

# Function to extract metadata from NWB file
def get_nwb_metadata(lindi_url):
    try:
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Extract basic metadata
        metadata = {
            'session_description': nwb.session_description,
            'identifier': nwb.identifier,
            'session_start_time': str(nwb.session_start_time),
            'institution': nwb.institution,
            'subject_id': nwb.subject.subject_id,
        }
        
        # Extract info about trials
        num_trials = len(nwb.intervals["trials"]['id'])
        
        # Sample some trial durations
        trial_durations = nwb.intervals["trials"]['stop_time'][:5] - nwb.intervals["trials"]['start_time'][:5]
        mean_duration = np.mean(trial_durations)
        
        metadata['num_trials'] = num_trials
        metadata['mean_trial_duration'] = float(mean_duration)
        
        # Get information about units
        num_units = len(nwb.units['id'].data)
        metadata['num_units'] = num_units
        
        return metadata
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Process sample files
all_metadata = []
for file_data in sample_files:
    print(f"Processing {file_data['path']}...")
    asset_id = file_data['asset_id']
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000945/assets/{asset_id}/nwb.lindi.json"
    
    metadata = get_nwb_metadata(lindi_url)
    if metadata:
        metadata['path'] = file_data['path']
        metadata['asset_id'] = asset_id
        all_metadata.append(metadata)
        print(f"  Session Description: {metadata.get('session_description', 'N/A')}")
        print(f"  Identifier: {metadata.get('identifier', 'N/A')}")
        print()

# Save metadata to file
with open('tmp_scripts/nwb_metadata.json', 'w') as f:
    json.dump(all_metadata, f, indent=2)

print(f"Metadata saved to tmp_scripts/nwb_metadata.json")

# Look for PRF information in identifiers
print("\nAnalyzing identifiers for PRF information:")
for metadata in all_metadata:
    identifier = metadata.get('identifier', '')
    print(f"  {identifier}")