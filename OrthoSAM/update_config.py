import json
from pathlib import Path
import os

def update_config_paths(config_path):
    # Resolve base directory (two levels up from this file)
    base_dir = Path(__file__).resolve().parent.parent

    config_path = Path(config_path)

    # If config file doesn't exist, create it with default values
    if not config_path.exists():
        default_config = {
            "MODEL_TYPE": "vit_h",
            "CheckpointDIR": str(base_dir / 'OrthoSAM' / 'MetaSAM'),
            "DataDIR": str(base_dir / 'data'),
            "MainOutDIR": str(base_dir / 'output'),
            "BaseDIR": str(base_dir)
        }
        config = default_config
        print(f"Config file does not exist. Creating a new one at {config_path}")
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update paths in existing config
        config['CheckpointDIR'] = str(base_dir / 'OrthoSAM' / 'MetaSAM')
        config['DataDIR'] = str(base_dir / 'data')
        config['MainOutDIR'] = str(base_dir / 'output')
        config['BaseDIR'] = str(base_dir)

    # Make sure required directories exist
    Path(config['DataDIR']).mkdir(parents=True, exist_ok=True)
    Path(config['MainOutDIR']).mkdir(parents=True, exist_ok=True)
    Path(config['CheckpointDIR']).mkdir(parents=True, exist_ok=True)

    # Save config back to file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print("Config updated with base:", base_dir)

update_config_paths(os.path.join('OrthoSAM','config.json'))