from typing import Dict, Any

from binance import AsyncClient
from binance import BinanceSocketManager
from binance.enums import *

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class BinanceClient:
    """Async client using python-binance for futures (demo supported)."""

    def __init__(self, demo: bool = True) -> None:
        self.demo = demo
        self.client: AsyncClient | None = None
        self.socket_manager: BinanceSocketManager | None = None
        self._exchange_info: Dict[str, Any] | None = None

    async def setup_account_config(self) -> None:
        api_key = config.binance_futures_demo.get("demo_api_key") if self.demo else config.binance_futures.get("api_key")
        api_secret = config.binance_futures_demo.get("demo_api_secret") if self.demo else config.binance_futures.get("api_secret")
        # Use demo mode (Binance futures demo endpoint)
        self.client = await AsyncClient.create(api_key, api_secret, testnet=False, demo=self.demo)
        self.socket_manager = BinanceSocketManager(self.client, user_timeout=60_000)
        await self._set_one_way_mode()
        await self._ensure_exchange_info()

    async def _set_one_way_mode(self) -> None:
        if not self.client:
            return
        try:
            await self.client.futures_change_position_mode(dualSidePosition="false")
            logger.info("Position mode set to One-Way")
        except Exception as exc:
            logger.warning(f"Position mode setting failed: {exc}")

    async def _ensure_exchange_info(self) -> None:
        if self._exchange_info:
            return
        if not self.client:
            raise RuntimeError("Client not initialized")
        self._exchange_info = await self.client.futures_exchange_info()

    async def get_account_metrics(self) -> Dict[str, float]:
        if not self.client:
            raise RuntimeError("Client not initialized")
        balances = await self._futures_balance_v2()
        positions = await self._positions_v2()
        total_wallet = 0.0
        for bal in balances:
            if bal.get("asset") == "USDT":
                total_wallet = float(bal.get("balance", 0.0))
                break
        total_unrealized = sum(float(pos.get("unRealizedProfit", 0.0)) for pos in positions)
        total_margin = sum(float(pos.get("initialMargin", 0.0)) for pos in positions)
        exposure_pct = (total_margin / total_wallet * 100.0) if total_wallet else 0.0
        return {
            "total_wallet_balance": total_wallet,
            "total_unrealized_pnl": total_unrealized,
            "total_margin_used": total_margin,
            "available_margin": total_wallet - total_margin,
            "exposure_percentage": exposure_pct,
            "position_count": sum(1 for pos in positions if abs(float(pos.get("positionAmt", 0.0))) > 0),
        }

    async def _futures_balance_v2(self):
        """Call balance with v2 endpoint, fallback to default on failure."""
        try:
            # Some versions accept version=2
            return await self.client.futures_account_balance(version=2)
        except Exception:
            try:
                return await self.client._request_futures_api("get", "balance", True, version=2)
            except Exception:
                return await self.client.futures_account_balance()

    async def _positions_v2(self):
        """Call positions with v2 endpoint, fallback to default."""
        try:
            return await self.client.futures_position_information(version=2)
        except Exception:
            return await self.client.futures_position_information()

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price=None, params=None):
        if not self.client:
            raise RuntimeError("Client not initialized")
        params = params or {}
        order = await self.client.futures_create_order(
            symbol=self._format_symbol(symbol),
            type=order_type.upper(),
            side=side.upper(),
            quantity=amount,
            price=price,
            **params,
        )
        return order

    def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        if not self._exchange_info:
            raise RuntimeError("Exchange info not loaded; call setup_account_config() first")
        sym = self._format_symbol(symbol)
        info = next((s for s in self._exchange_info.get("symbols", []) if s.get("symbol") == sym), None)
        if not info:
            return {"min_qty": 0.0, "min_notional": 0.0, "step_size": 0.0}
        filters = {f["filterType"]: f for f in info.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        notional = filters.get("MIN_NOTIONAL", {})
        step_size = float(lot.get("stepSize", 0.0) or 0.0)
        min_qty = float(lot.get("minQty", 0.0) or 0.0)
        min_notional = float(notional.get("notional", 0.0) or 0.0)
        return {"min_qty": min_qty, "min_notional": min_notional, "step_size": step_size}

    async def close(self) -> None:
        if self.client:
            await self.client.close_connection()
            self.client = None
        
    def _format_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "")

    def get_kline_socket(self, symbol: str, timeframe: str):
        if not self.socket_manager:
            raise RuntimeError("Socket manager not initialized")
        return self.socket_manager.kline_futures_socket(symbol=self._format_symbol(symbol), interval=timeframe)

    @staticmethod
    def interval_to_milliseconds(interval: str) -> int:
        """Convert Binance interval string to milliseconds."""
        ms_per_unit = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
        }
        unit = interval[-1]
        value = int(interval[:-1])
        if unit not in ms_per_unit:
            raise ValueError(f"Unsupported interval: {interval}")
        return value * ms_per_unit[unit]
