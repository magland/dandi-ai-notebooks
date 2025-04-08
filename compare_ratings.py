import json
import matplotlib.pyplot as plt
import numpy as np

# Load the two rating files
try:
    print("Loading ratings.json...")
    with open('ratings.json', 'r') as f:
        ratings1 = json.load(f)['ratings']
    print(f"Loaded {len(ratings1)} entries from ratings.json")

    print("Loading ratings_claude-3.5-sonnet.json...")
    with open('ratings_claude-3.5-sonnet.json', 'r') as f:
        ratings2 = json.load(f)['ratings']
    print(f"Loaded {len(ratings2)} entries from ratings_claude-3.5-sonnet.json")
except Exception as e:
    print(f"Error loading files: {str(e)}")
    exit(1)

# Get all possible score fields
score_fields = [
    'overall_score',
    'describe-dandiset',
    'load-dandiset',
    'load-dandiset-assets',
    'load-nwb',
    'load-nwb-data',
    'visualize-nwb-data',
    'plot-quality',
    'plot-quantity'
]

# Create dict to store matched scores
matched_scores = {field: {'x': [], 'y': []} for field in score_fields}

# Match scores by notebook and subfolder
print("\nMatching scores...")
match_count = 0
for r1 in ratings1:
    for r2 in ratings2:
        if r1['notebook'] == r2['notebook'] and r1['subfolder'] == r2['subfolder']:
            match_count += 1
            # Found a matching entry
            for field in score_fields:
                if field in r1 and field in r2:
                    matched_scores[field]['x'].append(r1[field])
                    matched_scores[field]['y'].append(r2[field])
print(f"Found {match_count} matching entries")

# Print number of points per field
print("\nPoints per category:")
for field in score_fields:
    print(f"{field}: {len(matched_scores[field]['x'])} points")

print("\nCreating scatter plot...")
plt.figure(figsize=(12, 12))

# Define colors for better visibility
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#f781bf', '#999999', '#dede00']

# Add jitter to help visualize overlapping points
jitter_amount = 0.15
for i, field in enumerate(score_fields):
    x = np.array(matched_scores[field]['x'])
    y = np.array(matched_scores[field]['y'])

    # Add random jitter
    x_jitter = x + np.random.uniform(-jitter_amount, jitter_amount, len(x))
    y_jitter = y + np.random.uniform(-jitter_amount, jitter_amount, len(y))

    plt.scatter(x_jitter, y_jitter, alpha=0.6, label=field, c=[colors[i]], s=100)

# Add diagonal line
plt.plot([0, 10], [0, 10], 'k--', alpha=0.5)

plt.xlabel('Original Ratings', fontsize=12)
plt.ylabel('Claude-3.5-sonnet Ratings', fontsize=12)
plt.title('Comparison of Rating Scores: Original vs Claude-3.5-sonnet', fontsize=14, pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

# Make square aspect ratio
plt.gca().set_aspect('equal')

# Set axis limits with some padding
plt.xlim(0, 10.5)
plt.ylim(0, 10.5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
print("\nSaving plot to rating_comparison.png...")
plt.savefig('rating_comparison.png', bbox_inches='tight', dpi=300)
print("Done!")
