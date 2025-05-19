import os
import json


def update_config_paths(config_path):
    base_dir = os.path.dirname(__file__)  

    # Load existing config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update only path
    config['CheckpointDIR'] = os.path.join(base_dir, 'MetaSAM')
    config['DataDIR'] = os.path.join(base_dir, 'data')

    # Save the updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("Config updated with base:", base_dir)

# Example usage
update_config_paths(os.path.join('code','config.json'))