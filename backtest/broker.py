"""backtest/broker.py

Simulated Binance Futures client for back-testing.

Only the subset of methods used by ExecutionEngine / OrderExecutor is
implemented.  This allows us to swap the real BinanceClient with SimBroker
without touching the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any

import pandas as pd

import config
from execution.optimizer import MeanVarianceOptimizer

# Constants - More realistic Binance Futures fees and costs
TAKER_FEE = getattr(config, "FEE_RATE", 0.0004)  # per-side taker fee
MAKER_FEE = 0.0002  # 0.02% per side (maker fee, not currently used)
FUNDING_MULTIPLIER = 1.0  # Applied to funding rates (can increase for more conservative backtesting)

# Helper aliases

Position = Dict[str, Any]
Order = Dict[str, Any]

# Approximate Binance futures symbol filters (min qty/notional/step). Fallback uses config.MIN_ORDER_NOTIONAL.
SYMBOL_FILTERS = {
    "BTCUSDT": {"min_qty": 0.001, "step_size": 0.0001, "min_notional": 5.0},
    "ETHUSDT": {"min_qty": 0.01, "step_size": 0.0001, "min_notional": 5.0},
    "BNBUSDT": {"min_qty": 0.1, "step_size": 0.01, "min_notional": 5.0},
    "XRPUSDT": {"min_qty": 1.0, "step_size": 1.0, "min_notional": 5.0},
    "SOLUSDT": {"min_qty": 0.1, "step_size": 0.01, "min_notional": 5.0},
}


class SimBroker:
    """Minimal async client that mimics ``BinanceClient`` for back-tests."""

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self._cash: float = initial_capital
        self._positions: Dict[str, Position] = {}
        self._trade_log: List[Dict[str, Any]] = []
        # Exposure caps mirrored from config
        self.max_gross = config.MAX_GROSS_EXPOSURE
        self.max_net = config.MAX_NET_EXPOSURE

        # Track the latest observed prices so equity() can be synchronous if needed
        self._last_prices: Dict[str, float] = {}

        # Price lookup callback set by BacktestingEngine each bar
        #   signature:  async def price(symbol: str) -> float
        self._price_callback = None

        # Current bar timestamp injected by BacktestingEngine so that trade
        # log times align with the simulated candle rather than wall-clock.
        self._bar_ts: Optional[str] = None

        # Add exchange attribute for compatibility with order manager
        self.exchange = self

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
        slippage_bp: Optional[float] = None,  # slippage in basis points (1 bp = 0.01%)
    ):
        """Net positions: adjust existing position, realize PnL on reductions, update margin."""
        if amount <= 0:
            return {"status": "error", "error": "Amount must be positive"}

        filters = self.get_symbol_filters(symbol)
        step = filters.get("step_size", 0.0) or 0.0
        min_qty = filters.get("min_qty", 0.0) or 0.0
        if step > 0:
            amount = math.floor((amount + 1e-12) / step) * step
        if amount < min_qty:
            return {"status": "error", "error": "Order quantity below minimum"}

        # Use provided price or fetch latest
        px = price if price is not None else await self._price(symbol)
        if px is None:
            return {"status": "error", "error": "Price unavailable"}

        # Enforce minimum notional like exchange filters
        min_notional_rule = max(config.MIN_ORDER_NOTIONAL, filters.get("min_notional", 0.0) or 0.0)
        notional_raw = abs(amount * px)
        if notional_raw < min_notional_rule:
            return {"status": "error", "error": "Order notional below minimum"}

        # Apply slippage: positive for worse fills (default 0)
        effective_slip = slippage_bp
        if effective_slip is None:
            effective_slip = config.SLIPPAGE_BPS_OVERRIDES.get(symbol, config.SLIPPAGE_BPS_DEFAULT)
        if effective_slip:
            px *= 1 + (effective_slip / 10000.0) * (1 if side == "buy" else -1)

        trade_sign = 1 if side.lower() == "buy" else -1
        trade_contracts = trade_sign * amount
        notional = abs(trade_contracts) * px
        fee = notional * TAKER_FEE
        lev_eff = leverage or (self._positions.get(symbol, {}).get("leverage") if self._positions.get(symbol) else 1)

        prev = self._positions.get(symbol)
        prev_contracts = prev["contracts"] if prev else 0.0
        prev_entry = float(prev["entryPrice"]) if prev else 0.0
        prev_margin = prev.get("margin", 0.0) if prev else 0.0

        new_contracts = prev_contracts + trade_contracts

        realized_pnl = 0.0
        if prev_contracts != 0 and (prev_contracts > 0) != (new_contracts > 0) and new_contracts != 0:
            # Flip: realize full prev position PnL
            realized_pnl += (px - prev_entry) * prev_contracts
        elif prev_contracts != 0 and (prev_contracts > 0) == (trade_contracts < 0):
            # Reduction in same symbol
            close_qty = min(abs(trade_contracts), abs(prev_contracts))
            realized_pnl += (px - prev_entry) * (close_qty * (1 if prev_contracts > 0 else -1))

        # Update or close position
        # Exposure checks against config caps (gross/net)
        equity_now = await self.equity()
        gross_other = 0.0
        net_other = 0.0
        for sym, pos in self._positions.items():
            if sym == symbol:
                continue
            px_sym = self._last_prices.get(sym, pos["entryPrice"])
            gross_other += abs(pos["contracts"] * px_sym)
            net_other += pos["contracts"] * px_sym
        new_notional = abs(new_contracts) * px
        new_net = net_other + new_contracts * px
        new_gross = gross_other + new_notional
        if equity_now > 0:
            gross_ratio = new_gross / equity_now
            net_ratio = abs(new_net) / equity_now
            if gross_ratio > self.max_gross or net_ratio > self.max_net:
                return {"status": "error", "error": "Exposure limits exceeded"}

        if new_contracts == 0:
            margin_release = prev_margin
            self._cash += margin_release + realized_pnl - fee
            self._positions.pop(symbol, None)
        else:
            # Weighted average entry when adding to same direction
            if prev_contracts != 0 and prev_contracts * new_contracts > 0:
                w_prev = abs(prev_contracts)
                w_new = abs(trade_contracts)
                new_entry = (prev_entry * w_prev + px * w_new) / (w_prev + w_new)
            else:
                new_entry = px
            margin_new = abs(new_contracts) * px / lev_eff
            # Enforce max gross exposure (simple check): if margin_new exceeds available + locked, reject
            equity_now = await self.equity()
            if margin_new > equity_now * 2:  # crude leverage cap of 2x equity
                return {"status": "error", "error": "Insufficient equity for margin"}
            margin_release = prev_margin
            self._cash += -fee + realized_pnl + margin_release - margin_new

            self._positions[symbol] = {
                "symbol": symbol,
                "contracts": new_contracts,
                "entryPrice": new_entry,
                "leverage": lev_eff,
                "marginType": margin_type or "isolated",
                "positionSide": "LONG" if new_contracts > 0 else "SHORT",
                "unrealizedPnl": 0.0,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "margin": margin_new,
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
                "leverage": lev_eff,
                "margin": self._positions[symbol]["margin"] if symbol in self._positions else 0.0,
                "fee": fee,
                "slippage_bp": effective_slip,
            }
        )

        return {
            "status": "success",
            "order": {"id": len(self._trade_log)},
            "position": self._positions.get(symbol),
            "filled_qty": abs(trade_contracts),
            "fill_price": px,
        }

    async def close_position(self, symbol: str, side: Optional[str] = None, slippage_bp: Optional[float] = None):
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
        effective_slip = slippage_bp
        if effective_slip is None:
            effective_slip = config.SLIPPAGE_BPS_OVERRIDES.get(symbol, config.SLIPPAGE_BPS_DEFAULT)
        if effective_slip and effective_slip > 0:
            is_long = contracts > 0
            slippage_factor = effective_slip / 10000.0
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
                "slippage_bp": effective_slip,
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
        if price is None or price <= 0:
            return 0.0  # price unavailable or invalid

        # Validate rate to prevent NaN propagation
        if rate is None or not isinstance(rate, (int, float)) or rate != rate:  # rate != rate checks for NaN
            return 0.0

        contracts = pos["contracts"]
        notional = abs(contracts) * price
        
        # Additional safety check for notional
        if notional <= 0 or notional != notional:  # Check for NaN
            return 0.0
        
        # Apply funding multiplier for more conservative/realistic costs
        effective_rate = rate * FUNDING_MULTIPLIER
        payment = notional * effective_rate * (1 if contracts > 0 else -1)

        # Final safety check for payment
        if payment != payment:  # Check for NaN
            return 0.0

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

    def update_last_prices(self, price_map: Dict[str, float]) -> None:
        """Push latest bar closes into the broker cache for mark-to-market PnL."""
        for sym, px in price_map.items():
            if px is not None:
                self._last_prices[sym] = px

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

    async def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for a symbol (no-op in backtest, used for compatibility)."""
        return {"status": "success", "symbol": symbol, "leverage": leverage}

    # Exchange-style methods for order manager compatibility
    async def create_order(
        self,
        symbol: str,
        type: Optional[str] = None,
        side: str = "buy",
        amount: float = 0.0,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
        order_type: Optional[str] = None,
        slippage_bp: float = 0.0,
    ):
        """Exchange-style order creation for compatibility with order manager."""
        params = params or {}
        ord_type = (order_type or type or params.get("type") or "market").lower()
        if ord_type == "market":
            result = await self.open_position(symbol, side, amount, price, slippage_bp=slippage_bp)
            if result["status"] == "success":
                fill_price = result.get("fill_price") or price or (result.get("position") or {}).get("entryPrice")
                fill_qty = result.get("filled_qty", amount)
                return {"id": f"sim_{len(self._trade_log)}", "status": "filled", "fills": [{"price": fill_price, "qty": fill_qty}]}
            raise Exception(f"Order failed: {result.get('error', 'Unknown error')}")
        # For non-market orders, mock a filled order
        return {"id": f"sim_{len(self._trade_log)}", "status": "filled"}

    def get_symbol_filters(self, symbol: str) -> Dict[str, float]:
        """Return realistic lot sizes and notionals for popular pairs."""
        filt = dict(SYMBOL_FILTERS.get(symbol.upper(), {}))
        min_notional = max(config.MIN_ORDER_NOTIONAL, filt.get("min_notional", config.MIN_ORDER_NOTIONAL))
        return {
            "min_qty": filt.get("min_qty", 0.0),
            "min_notional": min_notional,
            "step_size": filt.get("step_size", 0.0),
        }

    async def fetch_order(self, order_id: str, symbol: str):
        """Fetch order status (mock implementation)."""
        return {"id": order_id, "status": "filled", "filled": 1.0}

    async def cancel_order(self, order_id: str, symbol: str):
        """Cancel order (mock implementation)."""
        return {"id": order_id, "status": "canceled"}

    def reset(self):
        self._cash = self.initial_capital
        self._positions.clear()
        self._trade_log.clear()
        self._last_prices.clear()
        self._bar_ts = None 
