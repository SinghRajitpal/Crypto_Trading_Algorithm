from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import math

import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OrderInstruction:
    symbol: str
    side: str
    quantity: float
    notional: float
    target_weight: float
    current_weight: float


class TradeGenerator:
    """Converts target portfolio weights into executable order instructions."""

    def __init__(
        self,
        contract_multiplier: float = config.CONTRACT_MULTIPLIER,
        min_notional: float = config.MIN_ORDER_NOTIONAL,
    ) -> None:
        self.contract_multiplier = contract_multiplier
        self.min_notional = min_notional

    def generate_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        nav: float,
        prices: Dict[str, float],
        precision_provider: Optional[Callable[[str], Optional[Dict[str, float]]]] = None,
    ) -> List[OrderInstruction]:
        orders: List[OrderInstruction] = []
        symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in symbols:
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            delta_weight = target - current
            if abs(delta_weight) < 1e-4:
                continue

            price = prices.get(symbol)
            if price is None or price <= 0:
                logger.warning("Skipping %s due to missing or invalid price", symbol)
                continue

            notional_change = delta_weight * nav
            precision = precision_provider(symbol) if precision_provider else None
            min_notional = max(
                self.min_notional,
                precision.get("min_notional", 0.0) if precision else 0.0,
            )
            if abs(notional_change) < min_notional:
                continue

            raw_quantity = notional_change / (price * self.contract_multiplier)
            side = "buy" if raw_quantity > 0 else "sell"
            quantity = abs(raw_quantity)

            if precision:
                quantity = self._apply_step(quantity, precision.get("step_size", 0.0))
                if quantity < precision.get("min_qty", 0.0):
                    continue

            if quantity <= 0:
                continue

            final_notional = quantity * price * self.contract_multiplier
            if final_notional < min_notional:
                continue

            orders.append(
                OrderInstruction(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    notional=final_notional,
                    target_weight=target,
                    current_weight=current,
                )
            )

        return orders

    @staticmethod
    def _apply_step(quantity: float, step: float) -> float:
        if step <= 0:
            return quantity
        return math.floor(quantity / step) * step
