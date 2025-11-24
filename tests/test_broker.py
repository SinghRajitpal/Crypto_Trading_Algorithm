import asyncio
import math

import config
from backtest.broker import SimBroker, TAKER_FEE


def run(coro):
    return asyncio.run(coro)


def test_open_close_position_updates_cash_and_margin():
    broker = SimBroker(initial_capital=10000.0)
    price_map = {"A": 100.0, "A_close": 110.0}

    async def price(sym):
        return price_map.get(sym, 0.0)

    broker.set_price_callback(price)
    broker.set_bar_timestamp("2024-01-01T00:00:00")

    res_open = run(broker.open_position("A", side="buy", amount=1.0, price=price_map["A"], leverage=10))
    assert res_open["status"] == "success"
    bal_after_open = run(broker.get_balance())
    fee_open = price_map["A"] * TAKER_FEE
    impact_cost = config.IMPACT_KAPPA_DEFAULT * (price_map["A"] ** config.IMPACT_DELTA)
    expected_cash = 10000.0 - (price_map["A"] / 10) - fee_open - impact_cost
    assert math.isclose(bal_after_open["total"]["USDT"], expected_cash, rel_tol=1e-6)

    # Close at higher price; cash should increase by pnl minus fee and margin released
    price_map["A"] = price_map["A_close"]
    res_close = run(broker.close_position("A", slippage_bp=0.0))
    assert res_close["status"] == "closed"
    bal_final = run(broker.get_balance())
    pnl = (price_map["A_close"] - 100.0) * 1.0
    fee_close = price_map["A_close"] * TAKER_FEE
    impact_cost_close = config.IMPACT_KAPPA_DEFAULT * (price_map["A_close"] ** config.IMPACT_DELTA)
    expected_final_cash = expected_cash + pnl - fee_close - impact_cost_close + (100.0 / 10)
    assert math.isclose(bal_final["total"]["USDT"], expected_final_cash, rel_tol=1e-6)
    assert not run(broker.get_open_positions("A"))


def test_equity_includes_margin_and_unrealized_pnl():
    broker = SimBroker(initial_capital=5000.0)
    price_map = {"A": 50.0}

    async def price(sym):
        return price_map.get(sym, 0.0)

    broker.set_price_callback(price)
    run(broker.open_position("A", side="buy", amount=2.0, price=50.0, leverage=5))
    # Move price up; equity should reflect margin + unrealized PnL
    price_map["A"] = 60.0
    equity = run(broker.equity())
    margin = (2.0 * 50.0) / 5
    unrealized = (60.0 - 50.0) * 2.0
    cash = run(broker.get_balance())["total"]["USDT"]
    assert math.isclose(equity, cash + margin + unrealized, rel_tol=1e-6)


def test_apply_funding_adjusts_cash_and_logs():
    broker = SimBroker(initial_capital=2000.0)
    price_map = {"A": 100.0}

    async def price(sym):
        return price_map.get(sym, 0.0)

    broker.set_price_callback(price)
    run(broker.open_position("A", side="buy", amount=1.0, price=100.0, leverage=1))
    cash_before = run(broker.get_balance())["total"]["USDT"]
    payment = run(broker.apply_funding("A", rate=0.001))
    cash_after = run(broker.get_balance())["total"]["USDT"]
    # Long paying positive rate → cash decreases
    assert payment > 0
    assert cash_after < cash_before
    # Funding log row exists
    log = broker.trade_log()
    assert (log["type"] == "funding").any()


def test_symbol_filters_reasonable_defaults():
    broker = SimBroker()
    filt = broker.get_symbol_filters("BTCUSDT")
    assert filt["min_notional"] >= config.MIN_ORDER_NOTIONAL
    assert filt["min_qty"] >= 0.0
    assert filt["step_size"] >= 0.0


def test_exposure_limits_and_insufficient_equity():
    broker = SimBroker(initial_capital=1000.0)
    price_map = {"A": 100.0, "B": 50.0}

    async def price(sym):
        return price_map.get(sym, 0.0)

    broker.set_price_callback(price)
    # First position within gross/net limits
    res1 = run(broker.open_position("A", side="buy", amount=1.0, price=100.0, leverage=1))
    assert res1["status"] == "success"
    # Second position that would exceed gross/net cap (max_gross=1.2 default)
    res2 = run(broker.open_position("B", side="buy", amount=10.0, price=50.0, leverage=1))
    assert res2["status"] == "error"
    assert "Exposure limits exceeded" in res2["error"]


def test_funding_accumulates_over_time():
    broker = SimBroker(initial_capital=5000.0)
    price_map = {"A": 100.0}

    async def price(sym):
        return price_map.get(sym, 0.0)

    broker.set_price_callback(price)
    run(broker.open_position("A", side="buy", amount=1.0, price=100.0, leverage=1))
    cash_before = run(broker.get_balance())["total"]["USDT"]
    # Apply funding three times
    total_payment = 0.0
    for _ in range(3):
        total_payment += run(broker.apply_funding("A", rate=0.001))
    cash_after = run(broker.get_balance())["total"]["USDT"]
    # Long with positive rate should pay (decrease cash)
    assert cash_after < cash_before
    # Funding log rows count
    log = broker.trade_log()
    assert (log["type"] == "funding").sum() == 3
