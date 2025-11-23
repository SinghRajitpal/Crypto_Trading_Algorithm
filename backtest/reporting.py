import os
import json
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
