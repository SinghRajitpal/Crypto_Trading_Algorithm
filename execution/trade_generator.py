from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
            if abs(notional_change) < self.min_notional:
                continue

            quantity = notional_change / (price * self.contract_multiplier)
            side = "buy" if quantity > 0 else "sell"

            orders.append(
                OrderInstruction(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    notional=abs(notional_change),
                    target_weight=target,
                    current_weight=current,
                )
            )

        return orders
