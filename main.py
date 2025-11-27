import asyncio
import time
import os
from typing import Dict, List, Optional

from binance_exchange import BinanceClient
from data.data_engine import DataEngine
from algorithm.algo_engine import AlgoEngine
from algorithm.strategies.gru_forecast import GRUForecastStrategy
from execution.execution_engine import ProductionExecutionEngine
from execution.alerts import check_thresholds
from utils.logging_config import get_logger, console_log
from utils.monitoring_store import MonitoringStore
import config
import numpy as np

logger = get_logger(__name__)


class TradingAlgorithm:
    """Coordinates data, forecasting, and execution for the mean–variance strategy."""

    def __init__(self, testnet: bool = True, gru_model_dir: Optional[str] = None) -> None:
        self.binance_client = BinanceClient(testnet=testnet)
        self.data_engine = DataEngine(
            binance_client=self.binance_client,
            max_candles=config.RISK_WINDOW + config.REGRESSION_WINDOW + 20,
        )
        # Use GRU forecaster by default; fall back to provided directory if set
        model_dir = gru_model_dir or config.GRU_MODEL_DIR
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(f"GRU model directory not found: {model_dir}")
        self.data_engine.primary_timeframe = config.GRU_TIMEFRAME
        self.data_engine.data_fetcher.symbol_timeframes = [(sym, config.GRU_TIMEFRAME) for sym in config.DEFAULT_UNIVERSE]
        self.strategy = GRUForecastStrategy(
            self.data_engine,
            model_dir=model_dir,
            symbols=config.DEFAULT_UNIVERSE,
            timeframe=config.GRU_TIMEFRAME,
        )
        self.algo_engine = AlgoEngine(self.data_engine)
        self.execution_engine = ProductionExecutionEngine(
            binance_client=self.binance_client
        )
        self.data_task: Optional[asyncio.Task] = None
        self.universe_task: Optional[asyncio.Task] = None
        self.running = False
        self._nav_cache = 0.0
        self._last_nav_refresh = 0.0
        self.monitor_store = MonitoringStore(config.MONITOR_LOG_PATH)

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
        await self.binance_client.setup_account_config()
        logger.info("Fetching initial account metrics")
        metrics = await self._refresh_account_metrics(force=True)
        nav = metrics.get("total_wallet_balance", 0.0)
        self.execution_engine.update_total_capital(nav)

        logger.info("Starting data collection tasks")
        self.data_task = asyncio.create_task(self.data_engine.run())
        await self.data_engine.refresh_universe_snapshot()
        self.universe_task = asyncio.create_task(self._universe_refresh_loop())
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

        # Forecast diagnostics monitoring
        fm = forecast.diagnostics.get("forecast_monitor", {}) if forecast.diagnostics else {}
        if fm:
            avg_msep = float(
                np.nanmean([val.get("rolling_msep", float("nan")) for val in fm.values()])
            ) if hasattr(np, "nanmean") else 0.0
            logger.info("Forecast monitor | avg_rolling_msep=%.6f", avg_msep)

        median_mu = float(np.median(list(expected_returns.values())))
        logger.info(
            "Forecast summary | timestamp=%s | assets=%d | median_mu=%.6f",
            forecast.timestamp,
            len(expected_returns),
            median_mu,
        )

        self.execution_engine.refresh_risk_model(symbols, returns_matrix)

        # Data quality monitoring
        dq = self.data_engine.data_quality_report(symbols)
        for sym, stats in dq.items():
            if stats["missing_bars"] > 0 or stats["outliers"] > 0:
                logger.warning(
                    "Data quality | symbol=%s | missing_bars=%d | outliers=%d",
                    sym,
                    stats["missing_bars"],
                    stats["outliers"],
                )

        nav_metrics = await self._refresh_account_metrics()
        nav = nav_metrics.get("total_wallet_balance", self._nav_cache)
        self.execution_engine.update_total_capital(nav)

        prices = self._collect_prices(forecast.universe)
        result = await self.execution_engine.process_forecast(
            forecast=forecast,
            nav=nav,
            prices=prices,
            returns_matrix=returns_matrix,
        )

        status = result.get("status")
        if status == "completed":
            risk_diag = result.get("risk_diag", {})
            risk_cov_loss = result.get("risk_cov_loss", {})
            if risk_diag:
                logger.info(
                    "Risk diagnostics | mahalanobis=%.3f | malv_mean=%s | vol_mse=%s | vol_qlike=%s | symbols=%s | cov_turnover_fro=%s",
                    risk_diag.get("last_mahalanobis_d2", float("nan")),
                    risk_diag.get("malv_mean", float("nan")),
                    risk_diag.get("vol_mse_avg", float("nan")),
                    risk_diag.get("vol_qlike_avg", float("nan")),
                    risk_diag.get("symbols"),
                    risk_diag.get("cov_turnover_fro", float("nan")),
                )
            if risk_cov_loss:
                logger.info(
                    "Covariance loss | mse=%s | qlike=%s",
                    risk_cov_loss.get("cov_port_mse"),
                    risk_cov_loss.get("cov_port_qlike"),
                )
                if risk_cov_loss.get("cov_port_mse") and risk_cov_loss.get("cov_port_mse") > config.MONITOR_COV_MSE_WARN:
                    logger.warning("Covariance MSE above threshold: %.4f", risk_cov_loss.get("cov_port_mse"))
            logger.info(
                "Execution summary | turnover=%.4f | impact_est=%.2f | impact_concave=%.2f | kelly_f=%.3f | drawdown=%.3f",
                result.get("turnover", 0.0),
                result.get("impact_cost_est", 0.0),
                result.get("impact_cost_concave", 0.0),
                result.get("kelly_f", 0.0),
                result.get("kelly_drawdown", 0.0),
            )
            if "realized_slippage_bp" in result:
                logger.info(
                    "Slippage vs impact | realized_bp=%.3f | impact_vs_slippage=%.3f",
                    result.get("realized_slippage_bp"),
                    result.get("impact_vs_slippage", float("nan")),
                )
            if result.get("turnover_sigma") is not None:
                logger.info("Turnover attribution | sigma_component=%.4f", result.get("turnover_sigma"))
            if result.get("forecast_port_var") is not None and result.get("realized_port_var") is not None:
                logger.info(
                    "Portfolio variance | forecast=%.6f | realized=%.6f",
                    result.get("forecast_port_var"),
                    result.get("realized_port_var"),
                )
            check_thresholds(result)
            # Persist monitoring record
            record = {
                "timestamp": forecast.timestamp,
                "risk_diag": risk_diag,
                "risk_cov_loss": risk_cov_loss,
                "turnover": result.get("turnover"),
                "turnover_sigma": result.get("turnover_sigma"),
                "impact_est": result.get("impact_cost_est"),
                "impact_concave": result.get("impact_cost_concave"),
                "impact_propagator": result.get("impact_cost_propagator"),
                "kelly_f": result.get("kelly_f"),
                "drawdown": result.get("kelly_drawdown"),
                "slippage_bp": result.get("realized_slippage_bp"),
                "impact_vs_slippage": result.get("impact_vs_slippage"),
                "forecast_port_var": result.get("forecast_port_var"),
                "realized_port_var": result.get("realized_port_var"),
                "data_quality": self.data_engine.data_quality_report(symbols),
            }
            try:
                self.monitor_store.append(record)
            except Exception as exc:
                logger.warning("Failed to persist monitoring record: %s", exc)
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
        if self.universe_task:
            self.universe_task.cancel()
            try:
                await self.universe_task
            except asyncio.CancelledError:
                pass

    async def _universe_refresh_loop(self) -> None:
        interval = max(1, int(config.UNIVERSE_REFRESH_HOURS * 3600))
        while self.running:
            try:
                updated = await self.data_engine.refresh_universe_snapshot()
                if updated:
                    logger.info("Universe snapshot refreshed by scheduler")
            except Exception as exc:
                logger.warning(f"Universe refresh loop error: {exc}")
            await asyncio.sleep(interval)


async def main():
    system = TradingAlgorithm(testnet=True)
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
