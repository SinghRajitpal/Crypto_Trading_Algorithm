import asyncio
import time
from typing import Dict, List, Optional

from binance_exchange import BinanceClient
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.mean_variance import MeanVarianceForecastStrategy
from execution.execution_engine import ProductionExecutionEngine
from utils.logging_config import get_logger, console_log
import config
import numpy as np

logger = get_logger(__name__)


class TradingAlgorithm:
    """Coordinates data, forecasting, and execution for the mean–variance strategy."""

    def __init__(self, testnet: bool = True) -> None:
        self.binance_client = BinanceClient(testnet=testnet)
        self.data_engine = DataEngine(
            binance_client=self.binance_client,
            max_candles=config.RISK_WINDOW + config.REGRESSION_WINDOW + 20,
        )
        self.strategy = MeanVarianceForecastStrategy(self.data_engine)
        self.algo_engine = AlgoEngine(self.data_engine)
        self.execution_engine = ProductionExecutionEngine(
            binance_client=self.binance_client
        )
        self.data_task: Optional[asyncio.Task] = None
        self.running = False
        self._nav_cache = 0.0
        self._last_nav_refresh = 0.0

    async def start(self) -> None:
        if self.running:
            logger.warning("Trading system already running")
            return

        self.running = True
        await self._bootstrap()

        console_log("\n" + "=" * 80)
        console_log(f"{'MEAN-VARIANCE TRADING SYSTEM':^80}")
        console_log("=" * 80)
        logger.info(
            "System ready | timeframe=%s | universe=%d | initial_nav=%.2f",
            config.PRIMARY_TIMEFRAME,
            len(config.DEFAULT_UNIVERSE),
            self._nav_cache,
        )

        try:
            async for forecast in self.algo_engine.run(self.strategy):
                await self._process_forecast(forecast)
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as exc:
            logger.exception("Fatal error in trading loop: %s", exc)
        finally:
            await self.stop()

    async def _bootstrap(self) -> None:
        logger.info("Fetching initial account metrics")
        metrics = await self._refresh_account_metrics(force=True)
        nav = metrics.get("total_wallet_balance", 0.0)
        self.execution_engine.update_total_capital(nav)

        logger.info("Starting data collection tasks")
        self.data_task = asyncio.create_task(self.data_engine.run())
        await asyncio.sleep(5)  # allow initial history to load

    async def _process_forecast(self, forecast) -> None:
        returns_matrix, symbols = self.data_engine.get_return_matrix(
            forecast.universe, config.RISK_WINDOW
        )
        if returns_matrix.size == 0 or not symbols:
            logger.warning(
                "Skipping forecast | reason=insufficient_return_history | assets=%d",
                len(forecast.universe),
            )
            return

        expected_returns = forecast.expected_returns
        if not expected_returns:
            logger.warning("Skipping forecast | reason=no_expected_returns")
            return

        median_mu = float(np.median(list(expected_returns.values())))
        logger.info(
            "Forecast summary | timestamp=%s | assets=%d | median_mu=%.6f",
            forecast.timestamp,
            len(expected_returns),
            median_mu,
        )

        self.execution_engine.refresh_risk_model(symbols, returns_matrix)

        nav_metrics = await self._refresh_account_metrics()
        nav = nav_metrics.get("total_wallet_balance", self._nav_cache)
        self.execution_engine.update_total_capital(nav)

        prices = self._collect_prices(forecast.universe)
        result = await self.execution_engine.process_forecast(
            forecast=forecast,
            nav=nav,
            prices=prices,
        )

        status = result.get("status")
        if status == "completed":
            return
        if status == "skipped":
            logger.debug("Forecast skipped | reason=%s", result.get("reason"))
        else:
            logger.warning("Forecast processing result: %s", status)

    def _collect_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        for symbol in symbols:
            price = self.data_engine.get_latest_price(symbol, config.PRIMARY_TIMEFRAME)
            if price:
                prices[symbol] = price
        return prices

    async def _refresh_account_metrics(self, force: bool = False) -> Dict[str, float]:
        now = time.time()
        if not force and now - self._last_nav_refresh < 10:
            return {"total_wallet_balance": self._nav_cache}

        metrics = await self.binance_client.get_account_metrics()
        self._last_nav_refresh = now
        old_nav = self._nav_cache
        self._nav_cache = metrics.get("total_wallet_balance", self._nav_cache)
        if self._nav_cache > 0:
            change = abs(self._nav_cache - old_nav) / self._nav_cache if old_nav else 1.0
            if change > 0.01:
                logger.info("NAV update | value=%.2f", self._nav_cache)
        return metrics

    async def stop(self) -> None:
        if not self.running:
            return
        logger.info("Stopping trading system")
        self.running = False

        if self.data_task:
            self.data_task.cancel()
            try:
                await self.data_task
            except asyncio.CancelledError:
                pass

        await self.binance_client.close()


async def main():
    system = TradingAlgorithm(testnet=True)
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
