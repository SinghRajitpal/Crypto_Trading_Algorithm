from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)


class BarValidator:
    """Validates bars before they are ingested by downstream models."""

    def __init__(self, max_abs_return: float, min_volume: float) -> None:
        self.max_abs_return = max_abs_return
        self.min_volume = min_volume

    def is_valid(
        self,
        symbol: str,
        bar: Dict[str, float],
        prev_close: Optional[float],
    ) -> bool:
        close = bar.get("close")
        volume = bar.get("volume")

        if close is None or close <= 0:
            logger.debug(f"[BarValidator] Rejecting {symbol}: invalid close {close}")
            return False

        if volume is None or volume < self.min_volume:
            logger.debug(f"[BarValidator] Rejecting {symbol}: volume {volume}")
            return False

        if prev_close and prev_close > 0:
            simple_return = abs((close - prev_close) / prev_close)
            if simple_return > self.max_abs_return:
                logger.warning(
                    f"[BarValidator] Rejecting {symbol}: return {simple_return:.2%} exceeds threshold"
                )
                return False

        return True


class UniverseSelector:
    """Maintains the tradable universe using market-cap ranks and volume filters."""

    def __init__(
        self,
        max_rank: int,
        min_dollar_volume: float,
        lookback_days: int,
        default_universe: List[str],
    ) -> None:
        self.max_rank = max_rank
        self.min_dollar_volume = min_dollar_volume
        self.lookback_days = max(1, lookback_days)
        self.default_universe = list(default_universe)
        self.current_universe: List[str] = list(default_universe)
        self.last_refresh_date: Optional[date] = None

        # Rank data (lower rank == higher market cap)
        self.market_cap_ranks: Dict[str, int] = {}
        self._bootstrap_default_ranks()

        # Daily dollar volume history
        self.daily_dollar_volume: Dict[str, Dict[date, float]] = defaultdict(dict)

    def _bootstrap_default_ranks(self) -> None:
        for idx, symbol in enumerate(self.default_universe, start=1):
            self.market_cap_ranks[symbol] = idx

    def update_market_cap_snapshot(self, market_caps: Dict[str, float]) -> None:
        """Update ranks based on the latest market-cap snapshot."""
        sorted_symbols = sorted(
            market_caps.items(), key=lambda item: item[1], reverse=True
        )
        for rank, (symbol, _) in enumerate(sorted_symbols, start=1):
            self.market_cap_ranks[symbol] = rank

    def record_bar_metrics(
        self, symbol: str, timestamp_ms: int, close: float, volume: float
    ) -> None:
        """Accumulate daily dollar volume for each symbol."""
        if close <= 0 or volume is None:
            return

        bar_date = datetime.utcfromtimestamp(timestamp_ms / 1000).date()
        self.daily_dollar_volume[symbol][bar_date] = (
            self.daily_dollar_volume[symbol].get(bar_date, 0.0) + float(close) * float(volume)
        )
        self._prune_old_entries(symbol, bar_date)

    def _prune_old_entries(self, symbol: str, latest_date: date) -> None:
        """Remove history beyond the lookback window to bound memory usage."""
        cutoff = latest_date - timedelta(days=self.lookback_days)
        entries = self.daily_dollar_volume[symbol]
        for day in list(entries.keys()):
            if day < cutoff:
                entries.pop(day, None)

    def refresh_if_needed(self, timestamp_ms: int) -> bool:
        """Refresh the active universe once per UTC day."""
        current_date = datetime.utcfromtimestamp(timestamp_ms / 1000).date()
        if self.last_refresh_date == current_date:
            return False

        updated = self._recompute_universe(current_date)
        self.last_refresh_date = current_date
        return updated

    def _recompute_universe(self, current_date: date) -> bool:
        ranked = sorted(self.market_cap_ranks.items(), key=lambda item: item[1])
        next_universe: List[str] = []

        for symbol, rank in ranked:
            if rank > self.max_rank:
                continue

            median_volume = self._median_dollar_volume(symbol, current_date)
            if median_volume < self.min_dollar_volume:
                continue

            next_universe.append(symbol)
            if len(next_universe) >= self.max_rank:
                break

        if not next_universe:
            logger.warning(
                "[UniverseSelector] No assets passed the filters; falling back to defaults"
            )
            next_universe = list(self.default_universe)

        if next_universe != self.current_universe:
            logger.info(
                f"[UniverseSelector] Updated universe ({len(next_universe)} assets): {next_universe}"
            )
            self.current_universe = next_universe
            return True

        return False

    def _median_dollar_volume(self, symbol: str, current_date: date) -> float:
        history = self.daily_dollar_volume.get(symbol, {})
        if not history:
            return 0.0

        cutoff = current_date - timedelta(days=self.lookback_days)
        filtered = [value for day, value in history.items() if day >= cutoff]
        if not filtered:
            return 0.0

        return float(np.median(filtered))

    def get_active_universe(self) -> List[str]:
        return list(self.current_universe)
