#!/usr/bin/env python3

import os
import json
import yaml
from typing import Dict, List, Tuple

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
            ratings_file = f"{dandiset_id}_ratings.json"
            ratings_path = os.path.join(subfolder_path, ratings_file)
            if os.path.isfile(ratings_path):
                rating_files.append((dandiset_id, subfolder, ratings_path))

    return rating_files

def load_rating(rating_path: str) -> Dict:
    """Load a ratings.json file."""
    with open(rating_path, 'r') as f:
        return json.load(f)

def create_markdown_table(questions: List[Dict], ratings: List[Tuple[str, str, Dict]]) -> str:
    """Create markdown table from questions and ratings data."""
    # Create header
    headers = ['Notebook', 'Subfolder', 'Overall']
    headers.extend([q['name'] for q in questions])

    md = '# Notebook Ratings\n\n'

    # Add table headers
    md += '| ' + ' | '.join(headers) + ' |\n'
    md += '| ' + ' | '.join(['---' for _ in headers]) + ' |\n'

    # Add rows
    for dandiset_id, subfolder, rating_data in ratings:
        notebook_path = f'dandisets/{dandiset_id}/{subfolder}/{dandiset_id}.ipynb'

        # Create score lookup by question name
        scores = {s['name']: s['score'] for s in rating_data['scores']}

        # Calculate overall average from all available scores
        available_scores = [s['score'] for s in rating_data['scores']]
        overall_score = f"{sum(available_scores) / len(available_scores):.2f}" if available_scores else ""

        # Add notebook link, subfolder and overall score
        row = [f'[{dandiset_id}.ipynb]({notebook_path})', subfolder, overall_score]

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

def main():
    # Load questions
    questions = load_questions('questions.yml')

    # Find rating files
    rating_files = find_rating_files('dandisets')
    print(f"Found {len(rating_files)} rating files")

    # Load ratings
    ratings_data = []
    for dandiset_id, subfolder, rating_path in rating_files:
        rating = load_rating(rating_path)
        ratings_data.append((dandiset_id, subfolder, rating))

    # Create markdown table
    md_content = create_markdown_table(questions, ratings_data)

    # Write markdown file
    with open('ratings.md', 'w') as f:
        f.write(md_content)

    print("Created ratings.md")

if __name__ == "__main__":
    main()
