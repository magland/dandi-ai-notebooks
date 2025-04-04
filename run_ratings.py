#!/usr/bin/env python3

import os
from notebook_grader import rate_notebook
from typing import List, Tuple

def find_notebooks(base_dir: str) -> List[Tuple[str, str]]:
    """Find notebooks matching the pattern dandisets/<DANDISET_ID>/subfolder/<DANDISET_ID>.ipynb."""
    notebook_paths = []

    # List dandiset directories
    for dandiset_id in os.listdir(base_dir):
        dandiset_path = os.path.join(base_dir, dandiset_id)
        if not os.path.isdir(dandiset_path):
            continue

        # List subdirectories within dandiset
        for subfolder in os.listdir(dandiset_path):
            subfolder_path = os.path.join(dandiset_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Check for matching notebook
            notebook_name = f"{dandiset_id}.ipynb"
            notebook_path = os.path.join(subfolder_path, notebook_name)
            if os.path.isfile(notebook_path):
                notebook_paths.append((dandiset_id, notebook_path))

    return notebook_paths

def main():
    # Find all matching notebooks
    notebooks = find_notebooks('dandisets')
    print(f"Found {len(notebooks)} notebooks to process")

    # Process each notebook
    for i, (dandiset_id, notebook_path) in enumerate(notebooks, 1):
        print(f"\nProcessing notebook {i}/{len(notebooks)}")
        print(f"Dandiset: {dandiset_id}")
        print(f"Path: {notebook_path}")

        # Construct output path by replacing .ipynb with _ratings.json
        output_path = notebook_path.replace('.ipynb', '_ratings.json')

        try:
            rate_notebook(
                notebook_path_or_url=notebook_path,
                model=None,
                auto=True,
                questions_yaml='questions.yml',
                output_json=output_path
            )
            print(f"Successfully created: {output_path}")
        except Exception as e:
            print(f"Error processing {notebook_path}: {e}")

        # # Skip the pause after the last notebook
        # if i < len(notebooks):
        #     input("\nPress Enter to continue to next notebook...")
        print('')
        print('')

if __name__ == "__main__":
    main()
