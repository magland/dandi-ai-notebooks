import os
import json
import yaml
import shutil
import platform
from dandi_notebook_gen.generator import generate_notebook
import dandi_notebook_gen

def process_dandiset_directory(dandiset_path):
    dandiset_id = os.path.basename(dandiset_path)
    print(f"\nProcessing dandiset: {dandiset_id}")

    # Find all subdirectories within this dandiset
    print(f"Scanning subdirectories in {dandiset_path}")
    for subdir in os.listdir(dandiset_path):
        notebook_dir = os.path.join(dandiset_path, subdir)
        if not os.path.isdir(notebook_dir):
            print(f"Skipping {subdir} - not a directory")
            continue

        print(f"\nProcessing subdirectory: {subdir}")
        # Check if notebook already exists
        notebook_file = os.path.join(notebook_dir, f"{dandiset_id}.ipynb")
        if os.path.exists(notebook_file):
            print(f"Skipping {notebook_file} - notebook already exists")
            continue

        # Read config.yaml
        config_path = os.path.join(notebook_dir, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Warning: No config.yaml found in {notebook_dir}")
            continue

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model = config.get('model')
        if not model:
            print(f"Warning: No model specified in config.yaml for {notebook_dir}")
            continue

        # Clear working directory if it exists
        working_dir = os.path.join(notebook_dir, "working")
        if os.path.exists(working_dir):
            print(f"Clearing existing working directory: {working_dir}")
            shutil.rmtree(working_dir)

        # Clear metadata.json if it exists
        print("Checking for existing metadata.json")
        metadata_file = os.path.join(notebook_dir, "metadata.json")
        if os.path.exists(metadata_file):
            print("Removing existing metadata.json")
            os.remove(metadata_file)

        # Generate the notebook
        print(f"\nGenerating notebook for dandiset {dandiset_id}")
        print(f"Model: {model}")
        try:
            generate_notebook(
                dandiset_id=dandiset_id,
                output_path=notebook_file,
                model=model,
                auto=True,
                approve_all_commands=True,
                working_dir=working_dir
            )
        except Exception as e:
            print(f"Error generating notebook: {e}")
            with open(f'{notebook_dir}/working/metadata.json', 'w') as f:
                json.dump({"error": str(e)}, f, indent=2)
            continue

        # Create metadata.json
        metadata = {
            "dandi_notebook_gen_version": dandi_notebook_gen.__version__,
            "system_info": {
                "platform": platform.platform(),
                "hostname": platform.node(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }
        }
        with open(f'{notebook_dir}/working/metadata.json', 'r') as f:
            metadata.update(json.load(f))

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSuccessfully generated:")
        print(f"- Notebook: {notebook_file}")
        print(f"- Metadata: {metadata_file}")

def main():
    dandisets_dir = os.path.join(os.path.dirname(__file__), "dandisets")
    print(f"Starting notebook generation process")
    print(f"Scanning dandisets directory: {dandisets_dir}")

    # Process each dandiset directory
    for item in os.listdir(dandisets_dir):
        dandiset_path = os.path.join(dandisets_dir, item)
        if os.path.isdir(dandiset_path):
            process_dandiset_directory(dandiset_path)

if __name__ == "__main__":
    main()
