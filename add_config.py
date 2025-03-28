import sys
import os
from datetime import datetime
import yaml

def add_config(dandiset_id: str, model: str):
    dandiset_id = dandiset_id.zfill(6)
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Use only part after last / in model name if it exists
    model_suffix = model.split('/')[-1]
    dir_path = f'dandisets/{dandiset_id}/{date_str}-{model_suffix}'

    if os.path.exists(dir_path):
        raise Exception(f'Directory already exists: {dir_path}')

    os.makedirs(dir_path)

    config = {
        'model': model
    }

    config_path = os.path.join(dir_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python add_config.py <DANDISET_ID> <model>')
        sys.exit(1)

    dandiset_id = sys.argv[1]
    model = sys.argv[2]

    add_config(dandiset_id, model)
