import os
import json
from matplotlib.pylab import f
import yaml
import time
import shutil
import platform
from typing import Dict, List, Any
from dandi_notebook_gen.generator import generate_notebook
import dandi_notebook_gen

def get_relative_path(path: str) -> str:
    """Convert absolute path to relative path from current directory."""
    return os.path.relpath(path, os.path.dirname(__file__))

def _get_cost_for_model(model):
    if model == 'google/gemini-2.0-flash-001':
        return 0.1, 0.4
    elif model == 'openai/gpt-4o':
        return 2.5, 10
    elif model == 'anthropic/claude-3.5-sonnet':
        return 3, 15
    elif model == 'anthropic/claude-3.7-sonnet':
        return 3, 15
    elif model == 'anthropic/claude-3.7-sonnet:thinking':
        return 3, 15
    elif model == 'deepseek/deepseek-r1':
        return 0.55, 2.19
    else:
        return None, None

def collect_notebook_info(notebook_dir: str, dandiset_id: str, subfolder: str) -> Dict[str, Any]:
    """Collect all information about a notebook."""
    # Read config
    with open(os.path.join(notebook_dir, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    # Read metadata
    with open(os.path.join(notebook_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    # Get image files
    images_dir = os.path.join(notebook_dir, f"{dandiset_id}_files")
    image_files = []
    if os.path.exists(images_dir):
        image_files = [
            get_relative_path(os.path.join(images_dir, f))
            for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
        ]

    return {
        "dandiset_id": dandiset_id,
        "subfolder": subfolder,
        "config": config,
        "metadata": metadata,
        "paths": {
            "notebook": get_relative_path(os.path.join(notebook_dir, f"{dandiset_id}.ipynb")),
            "html": get_relative_path(os.path.join(notebook_dir, f"{dandiset_id}.html")),
            "images": sorted(image_files)
        }
    }

def process_dandiset_directory(dandiset_path, notebooks_data, start_time, timeout_sec):
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
        notebook_path = os.path.join(notebook_dir, f"{dandiset_id}.ipynb")
        metadata_path = os.path.join(notebook_dir, "metadata.json")
        config_path = os.path.join(notebook_dir, "config.yaml")
        html_path = os.path.join(notebook_dir, f"{dandiset_id}.html")
        # html_files_path = os.path.join(notebook_dir, f"{dandiset_id}_files")
        if not os.path.exists(notebook_path):
            # Read config.yaml
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
            if os.path.exists(metadata_path):
                print("Removing existing metadata.json")
                os.remove(metadata_path)

            # Clear html file if it exists
            print("Checking for existing html file")
            if os.path.exists(html_path):
                print("Removing existing html file")
                os.remove(html_path)

            # Generate the notebook
            print(f"\nGenerating notebook for dandiset {dandiset_id}")
            print(f"Model: {model}")
            try:
                generate_notebook(
                    dandiset_id=dandiset_id,
                    output_path=notebook_path,
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
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\nSuccessfully generated:")
            print(f"- Notebook: {notebook_path}")
            print(f"- Metadata: {metadata_path}")

        # Generate HTML file if doesn't exist
        if not os.path.exists(html_path):
            print(f"\nGenerating HTML file for {notebook_path}")
            # image files will go to html_files_path
            os.system(f"jupyter nbconvert --to html {notebook_path} --ExtractOutputPreprocessor.enabled=True")
            print(f"HTML file generated: {html_path}")

        # If both notebook and HTML exist, collect information
        if os.path.exists(notebook_path) and os.path.exists(html_path):
            notebooks_data.append(
                collect_notebook_info(notebook_dir, dandiset_id, subdir)
            )

            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_sec:
                print(f"\nTimeout reached after {elapsed_time:.1f} seconds. Stopping notebook generation.")
                return True  # Signal that we hit the timeout

    return False  # Signal that we haven't hit the timeout

def generate_markdown_report(notebooks_data: List[Dict[str, Any]]) -> str:
    """Generate a markdown report from the notebook data."""
    # Sort notebooks by dandiset ID and then by timestamp
    sorted_notebooks = sorted(notebooks_data, key=lambda x: (x['dandiset_id'], x['metadata'].get('timestamp', '')))

    # Create summary table
    md = "# DANDI Notebook Generation Results\n\n"
    md += "## Summary\n\n"
    md += "| Dandiset ID | Notebook | Model | Generated At | Generation Time (s) | Images | Tokens | Est $ |\n"
    md += "|-------------|----------|-------|--------------|---------------------|--------| ------ | ----- |\n"

    for nb in sorted_notebooks:
        runtime = nb['metadata'].get('elapsed_time_seconds', 0)
        timestamp = nb['metadata'].get('timestamp', 'N/A')
        model = nb['config'].get('model', 'N/A')
        image_count = len(nb['paths'].get('images', []))
        notebook_link = nb['paths']['notebook']

        total_prompt_tokens = nb['metadata'].get('total_prompt_tokens', 0)
        total_completion_tokens = nb['metadata'].get('total_completion_tokens', 0)
        total_prompt_tokens_k = total_prompt_tokens / 1000
        total_completion_tokens_k = total_completion_tokens / 1000

        prompt_cost, completion_cost = _get_cost_for_model(model)
        if prompt_cost is not None and completion_cost is not None:
            est_cost = total_prompt_tokens / 1e6 * prompt_cost + total_completion_tokens / 1e6 * completion_cost
        else:
            est_cost = "unknown"

        md += f"| {nb['dandiset_id']} | [{nb['dandiset_id']}.ipynb]({notebook_link}) | {model} | {timestamp} | {runtime:.2f} | {image_count} | {total_prompt_tokens_k:.1f}k / {total_completion_tokens_k:.1f}k | {est_cost:.2f} |\n"

    # Detailed sections grouped by dandiset
    current_dandiset = None
    for nb in sorted_notebooks:
        if nb['dandiset_id'] != current_dandiset:
            current_dandiset = nb['dandiset_id']
            md += f"\n## Dandiset {current_dandiset}\n\n"

        md += f"### {nb['subfolder']}\n\n"
        md += f"**Model:** {nb['config'].get('model', 'N/A')}  \n"
        md += f"**Generated:** {nb['metadata'].get('timestamp', 'N/A')}  \n"
        md += f"**Generation Time:** {nb['metadata'].get('elapsed_time_seconds', 0):.2f}s  \n"
        md += f"**Prompt Tokens:** {nb['metadata'].get('total_prompt_tokens', 0) / 1000:.1f}k  \n"
        md += f"**Completion Tokens:** {nb['metadata'].get('total_completion_tokens', 0) / 1000:.1f}k  \n"
        prompt_cost, completion_cost = _get_cost_for_model(nb['config'].get('model'))
        if prompt_cost is not None and completion_cost is not None:
            est_cost = nb['metadata'].get('total_prompt_tokens', 0) / 1e6 * prompt_cost + nb['metadata'].get('total_completion_tokens', 0) / 1e6 * completion_cost
            md += f"**Estimated Cost:** ${est_cost:.2f}  \n"
        else:
            md += f"**Estimated Cost:** unknown  \n"
        md += f"**Notebook:** [{nb['dandiset_id']}.ipynb]({nb['paths']['notebook']})  \n"

        # Add images if any
        if nb['paths'].get('images'):
            md += "\n**Generated Images:**\n\n"
            for img_path in nb['paths']['images']:
                md += f"![{os.path.basename(img_path)}]({img_path})  \n"

        md += "\n"

    return md

def main():
    timeout_sec = 60 * 60  # 1 hour timeout
    start_time = time.time()
    dandisets_dir = os.path.join(os.path.dirname(__file__), "dandisets")
    print(f"Starting notebook generation process")
    print(f"Scanning dandisets directory: {dandisets_dir}")

    # Initialize empty list to store all notebook data
    notebooks_data = []

    # Process each dandiset directory
    for item in os.listdir(dandisets_dir):
        dandiset_path = os.path.join(dandisets_dir, item)
        if os.path.isdir(dandiset_path):
            timeout_reached = process_dandiset_directory(dandiset_path, notebooks_data, start_time, timeout_sec)
            if timeout_reached:
                break

    # Write all notebook data to results.json
    with open('results.json', 'w') as f:
        results = {
            "notebooks": notebooks_data
        }
        json.dump(results, f, indent=2)

    # Generate and write markdown report
    markdown_content = generate_markdown_report(notebooks_data)
    with open('results.md', 'w') as f:
        f.write(markdown_content)

    print(f"\nResults written to:")
    print("- results.json")
    print("- results.md")

if __name__ == "__main__":
    main()
