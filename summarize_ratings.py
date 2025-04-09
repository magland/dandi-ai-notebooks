#!/usr/bin/env python3

import os
import json
import yaml
from typing import Dict, List, Tuple

model = None
# model = "anthropic/claude-3.5-sonnet"

def load_questions(questions_yaml: str) -> List[Dict]:
    """Load questions from questions.yml."""
    with open(questions_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data['questions']

def find_rating_files(base_dir: str) -> List[Tuple[str, str, str]]:
    """Find all rating json files in the dandisets directory.
    Returns list of tuples (dandiset_id, subfolder, rating_file_path)
    """
    rating_files = []

    # Walk through dandisets directory
    for dandiset_id in os.listdir(base_dir):
        dandiset_path = os.path.join(base_dir, dandiset_id)
        if not os.path.isdir(dandiset_path):
            continue

        # Check each subfolder
        for subfolder in os.listdir(dandiset_path):
            subfolder_path = os.path.join(dandiset_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Look for ratings.json file
            if model is None:
                ratings_file = f"{dandiset_id}_ratings.json"
            else:
                ratings_file = f'{dandiset_id}_ratings_{model.split("/")[-1]}.json'
            ratings_path = os.path.join(subfolder_path, ratings_file)
            if os.path.isfile(ratings_path):
                rating_files.append((dandiset_id, subfolder, ratings_path))

    return rating_files

def load_rating(rating_path: str) -> Dict:
    """Load a ratings.json file."""
    with open(rating_path, 'r') as f:
        return json.load(f)

def create_markdown_table(questions: List[Dict], ratings: List[Tuple[str, str, Dict]]) -> str:
    """Create markdown table from questions and ratings data with grouping by dandiset."""
    # Create header
    headers = ['Notebook', 'Subfolder', 'Overall']
    headers.extend([q['name'] for q in questions])

    md = '# Notebook Ratings\n\n'

    # Add table headers
    md += '| ' + ' | '.join(headers) + ' |\n'
    md += '| ' + ' | '.join(['---' for _ in headers]) + ' |\n'

    # Sort ratings by dandiset_id and subfolder
    sorted_ratings = sorted(ratings, key=lambda x: (x[0], x[1]))

    # Track current dandiset for grouping
    current_dandiset = None

    # Add rows
    for dandiset_id, subfolder, rating_data in sorted_ratings:
        # Add separator between dandisets
        if current_dandiset != dandiset_id:
            if current_dandiset is not None:  # Not the first group
                md += '| ' + ':---:|' * len(headers) + '\n'
            current_dandiset = dandiset_id

        notebook_link = "https://github.com/dandi-ai-notebooks/" + dandiset_id + "/blob/main/" + subfolder + "/" + dandiset_id + ".ipynb"

        # Create score lookup by question name
        scores = {s['name']: s['score'] for s in rating_data['scores']}

        # Calculate overall average from all available scores
        available_scores = [s['score'] for s in rating_data['scores']]
        overall_score = f"{sum(available_scores) / len(available_scores):.2f}" if available_scores else ""

        # Add notebook link, subfolder and overall score
        row = [f'[{dandiset_id}.ipynb]({notebook_link})', subfolder, overall_score]

        # Add score for each question (or blank if not found)
        for question in questions:
            name = question['name']
            score = scores.get(name, None)
            if score is not None:
                score = f'{score:.1f}'
            else:
                score = ''
            row.append(score)

        md += '| ' + ' | '.join(row) + ' |\n'

    return md

def create_json_data(questions: List[Dict], ratings: List[Tuple[str, str, Dict]]) -> Dict:
    """Create JSON data structure from questions and ratings data."""
    json_data = []

    for dandiset_id, subfolder, rating_data in ratings:
        # Create score lookup by question name
        scores = {s['name']: s['score'] for s in rating_data['scores']}

        # Calculate overall average
        available_scores = [s['score'] for s in rating_data['scores']]
        overall_score = sum(available_scores) / len(available_scores) if available_scores else None

        entry = {
            'notebook': f'{dandiset_id}.ipynb',
            'dandiset_id': dandiset_id,
            'subfolder': subfolder,
            'overall_score': overall_score
        }

        # Add individual question scores
        for question in questions:
            name = question['name']
            entry[name] = scores.get(name)

        json_data.append(entry)

    return {'ratings': json_data}

def main():
    # Load questions
    questions = load_questions('questions.yml')

    # Find rating files
    rating_files = find_rating_files('dandisets')
    print(f"Found {len(rating_files)} rating files")

    # Load ratings
    ratings_data = []
    for dandiset_id, subfolder, rating_path in rating_files:
        print(f'Processing {dandiset_id}/{subfolder}')
        rating = load_rating(rating_path)
        ratings_data.append((dandiset_id, subfolder, rating))

    # Create markdown table
    md_content = create_markdown_table(questions, ratings_data)

    # Create JSON data
    json_data = create_json_data(questions, ratings_data)

    # Write markdown file
    if model is None:
        ratings_md_file = 'ratings.md'
    else:
        ratings_md_file = f'ratings_{model.split("/")[-1]}.md'
    with open(ratings_md_file, 'w') as f:
        f.write(md_content)

    # Write JSON file
    if model is None:
        ratings_json_file = 'ratings.json'
    else:
        ratings_json_file = f'ratings_{model.split("/")[-1]}.json'
    with open(ratings_json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Created {ratings_md_file} and {ratings_json_file}")

if __name__ == "__main__":
    main()
