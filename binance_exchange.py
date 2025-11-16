import ccxt.pro as ccxt
from typing import Dict

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


class BinanceClient:
    """Minimal ccxt-based client for account metrics and market orders."""

    def __init__(self, testnet: bool = True) -> None:
        options = {
            "enableRateLimit": True,
        }
        if testnet:
            options["testnet"] = True

        api_key = config.binance_futures_testnet.get("testnet_api_key") if testnet else config.binance_futures.get("api_key")
        api_secret = config.binance_futures_testnet.get("testnet_api_secret") if testnet else config.binance_futures.get("api_secret")
        if api_key and api_secret:
            options["apiKey"] = api_key
            options["secret"] = api_secret

        self.exchange = ccxt.binanceusdm(options)
        self.exchange.options["defaultType"] = "future"
        if testnet:
            self.exchange.set_sandbox_mode(True)

        self._markets_loaded = False

    async def setup_account_config(self) -> None:
        """Set one-way mode and load markets."""
        try:
            await self.exchange.fapiPrivatePostPositionSideDual({"dualSidePosition": "false"})
            logger.info("Position mode set to One-Way")
        except Exception as exc:
            logger.warning(f"Position mode setting failed: {exc}")
        await self.ensure_markets_loaded()

    async def ensure_markets_loaded(self) -> None:
        if self._markets_loaded:
            return
        await self.exchange.load_markets()
        self._markets_loaded = True

    async def get_account_metrics(self) -> Dict[str, float]:
        await self.ensure_markets_loaded()
        balance = await self.exchange.fetch_balance()
        positions = await self.exchange.fetch_positions()

        total_wallet = float(balance["total"].get("USDT", 0.0))
        total_unrealized = sum(float(pos.get("unrealizedPnl", 0.0)) for pos in positions)
        total_margin = sum(float(pos.get("initialMargin", 0.0)) for pos in positions)
        exposure_pct = (total_margin / total_wallet * 100.0) if total_wallet else 0.0

        return {
            "total_wallet_balance": total_wallet,
            "total_unrealized_pnl": total_unrealized,
            "total_margin_used": total_margin,
            "available_margin": total_wallet - total_margin,
            "exposure_percentage": exposure_pct,
            "position_count": sum(1 for pos in positions if abs(float(pos.get("contracts", 0.0))) > 0),
        }

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price=None, params=None):
        await self.ensure_markets_loaded()
        formatted_symbol = self._format_symbol(symbol)
        return await self.exchange.create_order(formatted_symbol, order_type, side, amount, price, params or {})

    def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        if not self._markets_loaded:
            raise RuntimeError("Markets not loaded; call setup_account_config() first")
        formatted_symbol = self._format_symbol(symbol)
        market = self.exchange.market(formatted_symbol)
        min_qty = float(market["limits"]["amount"]["min"] or 0.0)
        min_notional = float(market["limits"]["cost"]["min"] or 0.0)
        precision = market["precision"].get("amount", 0)
        step_size = 10 ** (-precision) if precision else 0.0
        return {
            "min_qty": min_qty,
            "min_notional": min_notional,
            "step_size": step_size,
        }

    async def close(self) -> None:
        await self.exchange.close()

    def _format_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol
