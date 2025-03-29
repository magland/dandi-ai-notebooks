"""
This script compares data across different subjects in the Dandiset
to understand broader patterns across the dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dandi.dandiapi import DandiAPIClient

# Get list of assets in the Dandiset
print("Retrieving Dandiset assets...")
client = DandiAPIClient()
dandiset = client.get_dandiset("001174")
assets = list(dandiset.get_assets())

# Extract metadata
print("\nExtracting asset metadata...")
data = []
for asset in assets:
    path = asset.path
    subject_id = path.split('/')[0].replace('sub-', '')
    session = "unknown"
    if "ses-" in path:
        session = path.split('_')[1].replace('ses-', '')
    size_mb = asset.size / (1024 * 1024)  # Convert to MB
    
    data.append({
        'subject_id': subject_id,
        'session': session,
        'path': path,
        'size_mb': size_mb
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Print basic stats
print("\nBasic Dataset Stats:")
print(f"Total files: {len(df)}")
subject_counts = df['subject_id'].value_counts()
print("\nFiles per subject:")
for subject, count in subject_counts.items():
    print(f"  Subject {subject}: {count} files")

# Plot file sizes by subject
plt.figure(figsize=(12, 6))
subjects = df['subject_id'].unique()

for i, subject in enumerate(subjects):
    subject_df = df[df['subject_id'] == subject]
    plt.scatter(
        np.ones(len(subject_df)) * i, 
        subject_df['size_mb'],
        s=50, 
        alpha=0.7, 
        label=f'Subject {subject}'
    )

plt.xticks(range(len(subjects)), [f'Subject {s}' for s in subjects])
plt.ylabel('File Size (MB)')
plt.title('File Sizes by Subject')
plt.grid(alpha=0.3)
plt.savefig('tmp_scripts/file_sizes_by_subject.png', dpi=300)
plt.close()

# Plot session count by subject
session_counts = df.groupby('subject_id')['session'].nunique()

plt.figure(figsize=(10, 5))
session_counts.plot(kind='bar')
plt.xlabel('Subject')
plt.ylabel('Number of Sessions')
plt.title('Number of Sessions per Subject')
plt.grid(axis='y', alpha=0.3)
plt.savefig('tmp_scripts/sessions_per_subject.png', dpi=300)
plt.close()

# Print summary of what this dataset contains
print("\nDataset Summary:")
print(f"This dataset contains calcium imaging data from {len(subjects)} subjects: {', '.join(subjects)}")
print(f"The data spans multiple sessions across subjects, with subject F having the most sessions.")
print(f"File sizes vary considerably, with some files being much larger than others, likely due to different recording durations or sampling rates.")

print("\nScript completed successfully!")