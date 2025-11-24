import asyncio

from execution.executor import OrderExecutor
from execution.trade_generator import OrderInstruction


class StubClient:
    def __init__(self, should_fail=False):
        self.calls = []
        self.should_fail = should_fail

    async def create_order(self, symbol, order_type=None, type=None, side=None, amount=None, slippage_bp=0.0):
        if self.should_fail:
            raise RuntimeError("boom")
        self.calls.append({"symbol": symbol, "order_type": order_type or type, "side": side, "amount": amount, "slippage_bp": slippage_bp})
        return {"id": "ok"}


def make_instr(symbol="A", side="buy", qty=1.0, notional=100.0):
    return OrderInstruction(
        symbol=symbol,
        side=side,
        quantity=qty,
        notional=notional,
        price=notional / max(qty, 1e-9),
        target_weight=0.1,
        current_weight=0.0,
    )


def run(coro):
    return asyncio.run(coro)


def test_executor_sends_orders_and_records_success():
    client = StubClient()
    ex = OrderExecutor(client)
    orders = [make_instr("AAA", "buy", 2.0), make_instr("BBB", "sell", 1.5)]
    result = run(ex.execute_orders(orders))
    assert len(result) == 2
    assert all(r["status"] == "success" for r in result)
    assert len(client.calls) == 2
    assert client.calls[0]["order_type"] == "market"
    assert client.calls[0]["side"] == "buy"
    assert client.calls[0]["amount"] == 2.0


def test_executor_skips_zero_quantity():
    client = StubClient()
    ex = OrderExecutor(client)
    orders = [make_instr("AAA", "buy", 0.0), make_instr("BBB", "sell", 1.0)]
    result = run(ex.execute_orders(orders))
    # Zero-qty skipped, only one order sent
    assert len(result) == 1
    assert result[0]["symbol"] == "BBB"
    assert len(client.calls) == 1


def test_executor_handles_client_failure():
    client = StubClient(should_fail=True)
    ex = OrderExecutor(client)
    orders = [make_instr("AAA", "buy", 1.0)]
    result = run(ex.execute_orders(orders))
    assert len(result) == 1
    assert result[0]["status"] == "error"
    assert "reason" in result[0]


def test_executor_fallbacks_to_type_param():
    class TypeClient:
        def __init__(self):
            self.calls = []

        async def create_order(self, symbol, type=None, side=None, amount=None, slippage_bp=0.0, order_type=None):
            self.calls.append({"symbol": symbol, "type": type or order_type, "side": side, "amount": amount, "slippage_bp": slippage_bp})
            return {"id": "ok"}

    client = TypeClient()
    ex = OrderExecutor(client)
    orders = [make_instr("AAA", "buy", 1.0)]
    result = run(ex.execute_orders(orders))
    assert result[0]["status"] == "success"
    assert client.calls[0]["type"] == "market"
