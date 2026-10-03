from pathlib import Path

import yaml


def load_config(config_path):
    """Load an experiment YAML config as a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
