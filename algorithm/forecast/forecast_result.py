from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ForecastResult:
    """Container for per-bar expected returns and diagnostics."""

    timestamp: int
    universe: List[str]
    expected_returns: Dict[str, float]
    betas: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
