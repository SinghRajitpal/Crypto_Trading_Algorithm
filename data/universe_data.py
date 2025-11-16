import asyncio
from typing import Dict, List

import requests

from utils.logging_config import get_logger

logger = get_logger(__name__)


class UniverseDataFetcher:
    """Fetch daily market-cap snapshots for universe ranking."""

    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    BINANCE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

    def __init__(self, max_rank: int) -> None:
        self.max_rank = max_rank
        self._binance_symbols = None

    async def fetch_market_caps(self) -> Dict[str, float]:
        """Return symbol -> market cap for tradable Binance futures pairs."""
        return await asyncio.to_thread(self._fetch_snapshot)

    def _fetch_snapshot(self) -> Dict[str, float]:
        if self._binance_symbols is None:
            self._binance_symbols = self._fetch_binance_perpetual_symbols()

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
        }
        response = requests.get(self.COINGECKO_URL, params=params, timeout=10)
        response.raise_for_status()
        market_data = response.json()

        snapshot: Dict[str, float] = {}
        for coin in market_data:
            symbol = f"{coin['symbol'].upper()}USDT"
            if symbol in self._binance_symbols:
                snapshot[symbol] = float(coin.get("market_cap") or 0.0)
            if len(snapshot) >= self.max_rank:
                break

        logger.info(
            "Fetched market-cap snapshot | eligible=%d | max_rank=%d",
            len(snapshot),
            self.max_rank,
        )
        return snapshot

    def _fetch_binance_perpetual_symbols(self) -> List[str]:
        response = requests.get(self.BINANCE_INFO_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = [
            entry["symbol"]
            for entry in data.get("symbols", [])
            if entry.get("contractType") == "PERPETUAL" and entry.get("quoteAsset") == "USDT"
        ]
        return symbols
