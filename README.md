# DANDI AI Notebooks

This repository automates the generation of Jupyter notebooks for analyzing DANDI datasets (dandisets). It processes multiple dandisets and creates customized analysis notebooks based on configuration files. The automation uses:
- [dandi-notebook-gen](https://github.com/magland/dandi-notebook-gen) - for generating DANDI analysis notebooks
- [minicline](https://github.com/magland/minicline) - for command-line interaction and notebook generation

This package was created as part of the [Pre-COSYNE Brainhack](https://pre-cosyne-brainhack.github.io/hackathon2025/posts/about/), March 2025, Montreal.

For now, during development phase, we are using the google/gemini-2.0-flash-001 model because it is fast and inexpensive. Later on, we will switch to something like anthropic/claude-3.5-sonnet.

## Repository Structure

```
dandisets/
  ├── [dandiset_id]/
      └── [date]/
          ├── config.yaml       # Configuration for notebook generation
          ├── [dandiset_id].ipynb   # Generated analysis notebook
          ├── metadata.json     # Generation metadata and system info
          └── working/         # Working directory with analysis scripts
              ├── notebook.py
              ├── notebook.ipynb
              └── tmp_scripts/  # Generated analysis scripts and plots
```

## How It Works

1. The script scans through directories in `dandisets/`, each representing a DANDI dataset
2. For each dandiset, it processes dated subdirectories containing:
   - A `config.yaml` file specifying the model to use
   - Generated analysis notebooks and scripts
3. For each configuration:
   - Creates a working directory for analysis files
   - Generates a Jupyter notebook using the specified model
   - Produces analysis scripts and visualizations
   - Records metadata including version info and system details

## Configuration

Each dandiset directory requires a `config.yaml` file specifying:
- `model`: The model to use for notebook generation

## Generated Files

For each processed dandiset:
- Main notebook file (`[dandiset_id].ipynb`)
- Metadata file with generation details
- Working directory containing:
  - Analysis scripts
  - Generated plots
  - Execution logs
  - Temporary processing files

## Requirements

- Python packages:
  - dandi_notebook_gen
  - pyyaml
  - jupyter

## Notes

For the gh workflow to be able to push updates back to the repo you need to enable "Read and write permissions" in Settings -> Actions for the repo.