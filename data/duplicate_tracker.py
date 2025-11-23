from typing import Dict, Set


class DuplicateTracker:
    """Tracks cross-symbol duplicate timestamps per timeframe."""

    def __init__(self) -> None:
        self._seen: Dict[str, Set[int]] = {}

    def seen_before(self, timeframe: str, ts: int) -> bool:
        key = timeframe
        if key not in self._seen:
            self._seen[key] = set()
        if ts in self._seen[key]:
            return True
        self._seen[key].add(ts)
        return False
