import json
import os
from typing import Any, Dict


class MonitoringStore:
    """Append-only JSONL store for monitoring events."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
