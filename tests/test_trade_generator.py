import math

from execution.trade_generator import TradeGenerator, OrderInstruction


def test_skip_small_and_missing_prices():
    tg = TradeGenerator(contract_multiplier=1.0, min_notional=50)
    current = {"A": 0.0}
    target = {"A": 0.001, "B": 0.1}  # B price missing
    prices = {"A": 100.0}
    orders = tg.generate_orders(current, target, nav=1000, prices=prices)
    # delta_weight * nav = 1 -> notional below min_notional, so skipped; B skipped for missing price
    assert orders == []


def test_precision_and_min_qty_filtering():
    tg = TradeGenerator(contract_multiplier=1.0, min_notional=10)
    current = {"A": 0.0}
    target = {"A": 0.05}
    prices = {"A": 100.0}

    def precision_provider(sym):
        return {"step_size": 0.01, "min_qty": 0.05, "min_notional": 5}

    orders = tg.generate_orders(current, target, nav=1000, prices=prices, precision_provider=precision_provider)
    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, OrderInstruction)
    # quantity should respect step and min_qty
    step = 0.01
    assert math.isclose(order.quantity / step, round(order.quantity / step), abs_tol=1e-9)
    assert order.quantity >= 0.05
    assert order.price == prices["A"]


def test_contract_multiplier_per_asset_applied():
    tg = TradeGenerator(contract_multiplier=2.0, min_notional=10)
    current = {"A": 0.0}
    target = {"A": 0.1}
    prices = {"A": 50.0}
    orders = tg.generate_orders(current, target, nav=1000, prices=prices, precision_provider=None)
    assert len(orders) == 1
    order = orders[0]
    # notional should incorporate contract multiplier
    expected_notional = abs((0.1 * 1000) / (50.0 * 2.0)) * 50.0 * 2.0
    assert math.isclose(order.notional, expected_notional, rel_tol=1e-6)


def test_sell_and_buy_sides_set_correctly():
    tg = TradeGenerator(contract_multiplier=1.0, min_notional=10)
    current = {"A": 0.1, "B": -0.1}
    target = {"A": 0.0, "B": 0.0}
    prices = {"A": 100.0, "B": 50.0}
    orders = tg.generate_orders(current, target, nav=1000, prices=prices)
    sides = {o.symbol: o.side for o in orders}
    assert sides["A"] == "sell"
    assert sides["B"] == "buy"
