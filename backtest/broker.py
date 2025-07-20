"""backtest/broker.py

Simulated Binance Futures client for back-testing.

Only the subset of methods used by ExecutionEngine / OrderExecutor is
implemented.  This allows us to swap the real BinanceClient with SimBroker
without touching the rest of the codebase.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

import pandas as pd

# Constants - More realistic Binance Futures fees and costs

TAKER_FEE = 0.0004  # 0.04% per side (standard Binance futures taker fee)
MAKER_FEE = 0.0002  # 0.02% per side (maker fee, not currently used)
FUNDING_MULTIPLIER = 1.0  # Applied to funding rates (can increase for more conservative backtesting)

# Helper aliases

Position = Dict[str, Any]
Order = Dict[str, Any]


class SimBroker:
    """Minimal async client that mimics ``BinanceClient`` for back-tests."""

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self._cash: float = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trade_log: List[Dict[str, Any]] = []

        # Track the latest observed prices so equity() can be synchronous if needed
        self._last_prices: Dict[str, float] = {}

        # Price lookup callback set by BacktestingEngine each bar
        #   signature:  async def price(symbol: str) -> float
        self._price_callback = None

        # Current bar timestamp injected by BacktestingEngine so that trade
        # log times align with the simulated candle rather than wall-clock.
        self._bar_ts: Optional[str] = None

    # Plumbing helpers

    def _now_iso(self) -> str:
        """Return ISO timestamp for log rows.

        During back-tests the engine injects the *bar* timestamp via
        ``set_bar_timestamp`` so that order and funding records align exactly
        with candle times.  When not set (e.g. during live trading) we fall
        back to the actual current time.
        """
        return self._bar_ts or datetime.now(UTC).isoformat(timespec="seconds")

    async def _price(self, symbol: str) -> float:
        if self._price_callback is None:
            raise RuntimeError("SimBroker price callback not set")
        price = await self._price_callback(symbol)
        # Cache last price for equity calculation
        if price is not None:
            self._last_prices[symbol] = price
        return price

    # Balance / positions

    async def get_balance(self):
        return {
            "free": {"USDT": self._cash},
            "total": {"USDT": self._cash},
        }

    async def get_open_positions(self, symbol: Optional[str] = None):
        if symbol:
            p = self._positions.get(symbol)
            return [p] if p else []
        return list(self._positions.values())

    # BinanceClient wrapper expects these even if empty
    async def get_open_orders(self, symbol: Optional[str] = None):
        return []

    async def cancel_all_orders(self, symbol: Optional[str] = None):
        return []

    # Execution – simple market fills

    async def open_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[int] = None,
        margin_type: Optional[str] = None,
        slippage_bp: float = 0.0,  # slippage in basis points (1 bp = 0.01%)
    ):
        if amount <= 0:
            return {"status": "error", "error": "Amount must be positive"}

        # Use provided price or fetch latest
        px = price if price is not None else await self._price(symbol)
        if px is None:
            return {"status": "error", "error": "Price unavailable"}

        # Apply slippage: positive for worse fills (default 0)
        if slippage_bp:
            px *= 1 + (slippage_bp / 10000.0) * (1 if side == "buy" else -1)

        notional = amount * px
        fee = notional * TAKER_FEE
        # ------------------------------------------------------------------
        # Margin handling ----------------------------------------------------
        # Futures trading requires posting margin equal to *notional / leverage*.
        # We therefore:
        #   1. Deduct *margin* from available cash when the position is opened.
        #   2. Return the same *margin* to cash when the position is closed.
        # This ensures that equity (cash + unrealised PnL) matches the
        # economic reality and provides accurate performance metrics.
        # ------------------------------------------------------------------

        leverage_eff = leverage or 1
        margin_required = notional / leverage_eff

        # Deduct fee **and** margin from cash balance
        self._cash -= fee + margin_required

        contracts = amount if side == "buy" else -amount
        self._positions[symbol] = {
            "symbol": symbol,
            "contracts": contracts,
            "entryPrice": px,
            "leverage": leverage_eff,
            "marginType": margin_type or "isolated",
            "positionSide": "LONG" if contracts > 0 else "SHORT",
            "unrealizedPnl": 0.0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "margin": margin_required,
        }

        self._trade_log.append(
            {
                "timestamp": self._now_iso(),
                "symbol": symbol,
                "type": "open",
                "side": side,
                "contracts": amount,
                "price": px,
                "notional": notional,
                "leverage": leverage_eff,
                "margin": margin_required,
                "fee": fee,
            }
        )

        return {"status": "success", "order": {"id": len(self._trade_log)}, "position": self._positions[symbol]}

    async def close_position(self, symbol: str, side: Optional[str] = None, slippage_bp: float = 3.0):
        """Close position with slippage support."""
        pos = self._positions.get(symbol)
        if not pos:
            return {"status": "no_position"}

        px = await self._price(symbol)
        if px is None:
            return {"status": "error", "error": "Price unavailable"}

        contracts = pos["contracts"]
        
        # Apply slippage: unfavorable for closing positions
        # Long positions close with sells (worse = lower price)
        # Short positions close with buys (worse = higher price)
        if slippage_bp > 0:
            is_long = contracts > 0
            slippage_factor = slippage_bp / 10000.0
            px *= 1 - slippage_factor if is_long else 1 + slippage_factor
        
        notional = abs(contracts) * px
        fee = notional * TAKER_FEE
        pnl = (px - pos["entryPrice"]) * contracts  # long +ve, short -ve

        # ------------------------------------------------------------------
        # Release previously locked margin and realise PnL minus fee
        # ------------------------------------------------------------------

        margin_released = pos.get("margin", notional / max(pos.get("leverage", 1), 1))

        self._cash += margin_released + pnl - fee

        self._trade_log.append(
            {
                "timestamp": self._now_iso(),
                "symbol": symbol,
                "type": "close",
                "side": "sell" if contracts > 0 else "buy",
                "contracts": abs(contracts),
                "price": px,
                "notional": notional,
                "leverage": pos.get("leverage", 1),
                "margin": margin_released,
                "fee": fee,
                "pnl": pnl,
            }
        )

        del self._positions[symbol]
        return {"status": "closed", "pnl": pnl}

    async def close_all_positions(self):
        """Close every open position at current market price (end-of-backtest cleanup)."""
        results = {}
        for sym in list(self._positions.keys()):
            results[sym] = await self.close_position(sym)
        return results

    # ------------------------------------------------------------------
    # Funding payments
    # ------------------------------------------------------------------

    async def apply_funding(self, symbol: str, rate: float) -> float:
        """Apply funding payment for *symbol* at the given *rate*.

        Returns the cash delta (positive means we received, negative we paid).
        Formula used (matching Binance):  funding = notional * rate * funding_multiplier
        Longs pay positive rate, shorts receive; reverse if rate negative.
        """
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0  # no position

        price = await self._price(symbol)
        if price is None:
            return 0.0

        contracts = pos["contracts"]
        notional = abs(contracts) * price
        
        # Apply funding multiplier for more conservative/realistic costs
        effective_rate = rate * FUNDING_MULTIPLIER
        payment = notional * effective_rate * (1 if contracts > 0 else -1)

        self._cash -= payment  # if payment positive and long we pay (cash decrease)

        self._trade_log.append(
            {
                "timestamp": self._now_iso(),
                "symbol": symbol,
                "type": "funding",
                "rate": effective_rate,
                "notional": notional,
                "payment": payment,
            }
        )

        return payment

    # Analysis helpers

    def set_price_callback(self, coro):
        """BacktestingEngine passes in its async price lookup each bar."""
        self._price_callback = coro

    def set_bar_timestamp(self, ts):
        if isinstance(ts, str):
            self._bar_ts = ts
        elif isinstance(ts, datetime):
            self._bar_ts = ts.isoformat(timespec="seconds")
        else:
            # None or unsupported type resets to real-time clock
            self._bar_ts = None

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log)

    async def equity(self) -> float:
        """Return current equity = cash + unrealised PnL for all open positions."""
        equity = self._cash

        for symbol, pos in self._positions.items():
            # Try cached price first to avoid awaiting in tight loops
            price = self._last_prices.get(symbol)
            if price is None:
                try:
                    price = await self._price(symbol)
                except Exception:
                    price = pos["entryPrice"]  # fallback

            contracts = pos["contracts"]
            pnl = (price - pos["entryPrice"]) * contracts

            # Add back margin currently locked in this position so that equity
            # represents *cash + margin + PnL* (matching exchange definition).
            margin_locked = pos.get("margin", 0.0)

            equity += margin_locked + pnl

        return equity

    def reset(self):
        self._cash = self.initial_capital
        self._positions.clear()
        self._trade_log.clear()
        self._last_prices.clear()
        self._bar_ts = None 