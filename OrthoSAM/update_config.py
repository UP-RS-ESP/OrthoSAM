import os
import json


def update_config_paths(config_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))

    with open(config_path, 'r') as f:
        config = json.load(f)

    config['CheckpointDIR'] = os.path.join(base_dir, 'OrthoSAM', 'MetaSAM')
    config['DataDIR'] = os.path.join(base_dir, 'data')
    config['MainOutDIR'] = os.path.join(base_dir, 'output')
    config['BaseDIR'] = base_dir

    if not os.path.exists(config['DataDIR']):
        os.makedirs(config['DataDIR'])
    if not os.path.exists(config['MainOutDIR']):
        os.makedirs(config['MainOutDIR'])

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("Config updated with base:", base_dir)

update_config_paths(os.path.join('OrthoSAM','config.json'))