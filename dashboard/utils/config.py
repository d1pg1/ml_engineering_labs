"""Load dashboard configuration from config.yaml."""
from pathlib import Path
from typing import Any, Dict

import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def repo_root() -> Path:
    return _REPO_ROOT
