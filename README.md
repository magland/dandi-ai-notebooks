# DANDI AI Notebooks

This repository automates the generation of Jupyter notebooks for analyzing DANDI datasets (dandisets). It processes multiple dandisets and creates customized analysis notebooks based on configuration files. The automation uses [dandi-notebook-gen](https://github.com/magland/dandi-notebook-gen) and [minicline](https://github.com/magland/minicline).

This packages were created as part of the [Pre-COSYNE Brainhack](https://pre-cosyne-brainhack.github.io/hackathon2025/posts/about/), March 2025, Montreal.

[View results](./results.md) (updates regularly)

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

## Usage

### Creating New Configurations

Use `add_config.py` to create new dandiset configurations:

```bash
python add_config.py <dandiset_id> <model>
```

This creates a new dated directory with a config.yaml file. For models with paths (e.g., 'deepseek/deepseek-r1'), only the final part is used in the directory name.

## Grading notebooks

To grade a notebook using the notebook-grader package, first install the latest version of the package:

```bash
# Clone https://github.com/magland/notebook-grader
cd notebook-grader
git pull
pip install -e .
```

Then run the grading script:

```bash
cd dandisets/<DANDISET_ID>/<SUBDIR>
notebook-grader grade-notebook <DANDISET_ID>.ipynb --auto --output-notebook <DANDISET_ID>_graded.ipynb --model google/gemini-2.0-flash-001
```

The output notebook will contain new markdown cells that identify potential problems, if any are found.

## Requirements

- Python packages:
  - dandi_notebook_gen
  - pyyaml
  - jupyter

## Notes

For the gh workflow to be able to push updates back to the repo you need to enable "Read and write permissions" in Settings -> Actions for the repo.
