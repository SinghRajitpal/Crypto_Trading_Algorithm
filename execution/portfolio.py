from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import numpy as np
from datetime import datetime, timedelta
import config
from utils.logging_config import get_logger, console_log

# Get logger for this module
logger = get_logger(__name__)

@dataclass
class AllocationWeights:
    """Represents allocation weights and metrics for the portfolio."""
    symbol: str
    weight: float
    allocated_capital: float
    volatility: float
    avg_correlation: float
    raw_weight: float

class ProductionPortfolioManager:
    """Production Portfolio Manager implementing the exact allocation system from the document.
    
    Core Features:
    - Inverse volatility weighting with correlation adjustment: w_i = (1/σ_i) × (1 + α × avg_correlation_i)
    - Daily rebalancing with <1% turnover
    - Volatility targeting at 18% with regime detection
    - EMA-based volatility (ATR) and correlation tracking over 60 bars
    """
    
    def __init__(self, total_capital: float, target_volatility: float = None, max_allocation_pct: float = None):
        """Initialize the production portfolio manager.
        
        Args:
            total_capital: Total trading capital in USDT.
            target_volatility: Target portfolio volatility (default from config).
            max_allocation_pct: Maximum percentage of capital to allocate (default from config).
        """
        self.total_capital = total_capital
        self.target_volatility = target_volatility or config.TARGET_VOLATILITY
        self.max_allocation_pct = max_allocation_pct or config.MAX_ALLOCATION_PCT
        
        # Core data structures for the allocation system
        self.volatility_data: Dict[str, List[float]] = {}  # EMA of 1-min ATR(30) over 60 bars
        self.correlation_data: Dict[Tuple[str, str], List[float]] = {}  # EMA of pairwise returns over 60 bars
        self.allocation_weights: Dict[str, AllocationWeights] = {}
        
        # Regime detection
        self.volatility_history: List[float] = []  # For 30-day percentile calculation
        self.last_rebalance_time = datetime.now()
        
        # Fixed parameters from config
        self.alpha = config.ALPHA_CORRELATION  # Correlation adjustment parameter
        self.lookback_bars = config.LOOKBACK_BARS  # EMA lookback for vol and correlation
        self.regime_percentile = config.REGIME_PERCENTILE  # Percentile for high vol regime
        
        # Track reserved allocations for position management
        self.reserved_allocations: Dict[str, float] = {}
        
        logger.info(f"ProductionPortfolioManager initialized with ${total_capital:.2f}, target_vol={self.target_volatility:.1%}")
    
    def update_volatility_data(self, symbol: str, atr_value: float, current_price: Optional[float] = None) -> None:
        """Update volatility data (EMA of 1-min ATR(30)) for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            atr_value: Current ATR value.
            current_price: Current price for normalization (optional).
        """
        if symbol not in self.volatility_data:
            self.volatility_data[symbol] = []
        
        # Normalize ATR to percentage if current price is available
        # This ensures volatility data is in percentage form for comparison with target_volatility
        if current_price and current_price > 0:
            normalized_atr = atr_value / current_price
            logger.debug(f"{symbol}: Raw ATR {atr_value:.6f} → Normalized {normalized_atr:.6f} ({normalized_atr*100:.3f}%)")
        else:
            # Fallback: assume ATR is already normalized if no price provided
            normalized_atr = atr_value if atr_value < 1.0 else atr_value / 100.0
            logger.debug(f"{symbol}: Using ATR {normalized_atr:.6f} (assuming normalized)")
        
        # Keep rolling window of 60 bars for EMA calculation
        if len(self.volatility_data[symbol]) >= self.lookback_bars:
            self.volatility_data[symbol].pop(0)
        
        self.volatility_data[symbol].append(normalized_atr)
    
    def update_correlation_data(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Update correlation data (EMA of pairwise returns) between symbol pairs.
        
        Args:
            symbol1: First trading pair.
            symbol2: Second trading pair.
            correlation: Current correlation value.
        """
        # Ensure consistent ordering
        if symbol1 > symbol2:
            symbol1, symbol2 = symbol2, symbol1
        
        pair = (symbol1, symbol2)
        if pair not in self.correlation_data:
            self.correlation_data[pair] = []
        
        # Keep rolling window of 60 bars for EMA calculation
        if len(self.correlation_data[pair]) >= self.lookback_bars:
            self.correlation_data[pair].pop(0)
        
        self.correlation_data[pair].append(correlation)
    
    def initialize_market_data(self, data_engine, symbols: List[str]) -> bool:
        """Initialize portfolio with real market data from data engine.
        
        Args:
            data_engine: DataEngine instance for fetching market data.
            symbols: List of symbols to initialize.
            
        Returns:
            True if initialization was successful.
        """
        try:
            logger.info("Initializing portfolio with real market data...")
            
            # Get real-time volatilities from data engine
            volatilities = data_engine.initialize_portfolio_volatilities(symbols)
            
            # Clear existing data and build fresh history
            self.volatility_data.clear()
            
            # Build volatility history for each symbol (need 25+ points for proper EMA)
            for symbol, volatility in volatilities.items():
                for _ in range(25):  # Build sufficient history
                    self.update_volatility_data(symbol, volatility)
                logger.debug(f"{symbol}: Initialized with volatility {volatility:.4f}")
            
            # Add default correlation data
            correlation_pairs = [
                ("BTCUSDT", "ETHUSDT", 0.7),
                ("BTCUSDT", "SOLUSDT", 0.6), 
                ("ETHUSDT", "SOLUSDT", 0.75),
                ("BTCUSDT", "BNBUSDT", 0.65),
                ("ETHUSDT", "BNBUSDT", 0.7),
                ("XRPUSDT", "BTCUSDT", 0.5),
                ("XRPUSDT", "ETHUSDT", 0.55),
                ("BNBUSDT", "SOLUSDT", 0.6)
            ]
            
            # Clear existing correlation data
            self.correlation_data.clear()
            
            for sym1, sym2, corr in correlation_pairs:
                if sym1 in symbols and sym2 in symbols:
                    # Build sufficient history for correlation EMA
                    for _ in range(25):
                        self.update_correlation_data(sym1, sym2, corr)
            
            logger.info(f"Market data initialization complete for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing market data: {e}")
            return False
    
    def get_volatility_ema(self, symbol: str) -> float:
        """Get EMA of volatility (σ_i) for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            EMA of volatility or default if insufficient data.
        """
        if symbol not in self.volatility_data or not self.volatility_data[symbol]:
            return 0.02  # Default 2% volatility
        
        data = self.volatility_data[symbol]
        alpha = 2 / (len(data) + 1)
        
        ema = data[0]
        for value in data[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return max(ema, 0.001)  # Floor at 0.1% to prevent division issues
    
    def get_average_correlation(self, symbol: str, active_symbols: List[str]) -> float:
        """Get average correlation for a symbol with all other active symbols.
        
        Args:
            symbol: Target symbol.
            active_symbols: List of all active symbols.
            
        Returns:
            Average correlation with other symbols.
        """
        correlations = []
        
        for other_symbol in active_symbols:
            if symbol == other_symbol:
                continue
                
            # Get pair key (alphabetically ordered)
            pair = tuple(sorted([symbol, other_symbol]))
            
            if pair in self.correlation_data and self.correlation_data[pair]:
                # Calculate EMA of correlations
                data = self.correlation_data[pair]
                alpha = 2 / (len(data) + 1)
                
                ema = data[0]
                for value in data[1:]:
                    ema = alpha * value + (1 - alpha) * ema
                
                correlations.append(ema)
        
        return sum(correlations) / len(correlations) if correlations else 0.0
    
    def compute_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Compute portfolio weights using the document's formula:
        w_i = (1/σ_i) × (1 + α × avg_correlation_i), normalized so Σw_i = 1
        
        Args:
            symbols: List of symbols to allocate.
            
        Returns:
            Dictionary of normalized weights.
        """
        if not symbols:
            return {}
        
        raw_weights = {}
        
        for symbol in symbols:
            sigma_i = self.get_volatility_ema(symbol)
            avg_corr_i = self.get_average_correlation(symbol, symbols)
            
            # Apply volatility floor to prevent extreme weights
            sigma_i = max(sigma_i, 0.001)  # 0.1% minimum volatility
            
            # Core formula: w_i = (1/σ_i) × (1 + α × avg_correlation_i)
            # CRITICAL FIX: Use the exact formula from the document
            correlation_adjustment = 1 + self.alpha * avg_corr_i  # Correct formula: ADD correlation effect
            raw_weight = (1 / sigma_i) * correlation_adjustment
            raw_weights[symbol] = raw_weight
        
        # Normalize so sum = 1
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            # Equal weights fallback
            return {s: 1.0 / len(symbols) for s in symbols}
        
        normalized_weights = {s: w / total_weight for s, w in raw_weights.items()}
        
        logger.debug(f"Computed weights for {len(symbols)} symbols")
        for symbol in symbols:
            vol = self.get_volatility_ema(symbol)
            corr = self.get_average_correlation(symbol, symbols)
            logger.debug(f"  {symbol}: weight={normalized_weights[symbol]:.4f}, vol={vol:.4f}, corr={corr:.3f}")
        
        return normalized_weights
    
    def is_high_volatility_regime(self) -> bool:
        """Check if current market is in high volatility regime.
        σ_hat > 75th percentile over last 30 days.
        
        Returns:
            True if high volatility regime.
        """
        if len(self.volatility_history) < 5:  # Need minimum data points
            return False  # Insufficient data
        
        # Calculate current average volatility across all active symbols
        active_symbols = list(self.volatility_data.keys())
        if not active_symbols:
            return False
        
        current_volatilities = [self.get_volatility_ema(s) for s in active_symbols]
        sigma_hat = sum(current_volatilities) / len(current_volatilities)
        
        # FIXED: Use proper percentile calculation with minimum history requirement
        if len(self.volatility_history) >= 10:
            # Use the document formula: high_vol_regime if σ_hat > percentile over last 30 days
            percentile_threshold = np.percentile(self.volatility_history, self.regime_percentile)
        else:
            # With limited history, use conservative threshold (1.5x average)
            percentile_threshold = np.mean(self.volatility_history) * 1.5
        
        is_high_vol = sigma_hat > percentile_threshold
        
        logger.debug(f"Volatility regime check: σ_hat={sigma_hat:.4f}, {self.regime_percentile}th percentile={percentile_threshold:.4f}, high_vol={is_high_vol}")
        
        return is_high_vol
    
    def calculate_scaling_multiplier(self) -> float:
        """Calculate volatility targeting scaling multiplier:
        m = min(1, target_vol/σ_hat) × (0.5 if high_vol_regime else 1.0)
        
        Returns:
            Scaling multiplier.
        """
        active_symbols = list(self.volatility_data.keys())
        if not active_symbols:
            return 1.0
        
        # Calculate average volatility across portfolio (σ_hat)
        sigma_hat = sum(self.get_volatility_ema(s) for s in active_symbols) / len(active_symbols)
        
        # FIXED: Update volatility history BEFORE checking regime
        self.volatility_history.append(sigma_hat)
        if len(self.volatility_history) > 30:  # Keep 30 days for percentile
            self.volatility_history.pop(0)
        
        # Volatility targeting component: min(1, target_vol/σ_hat)
        vol_scaling = min(1.0, self.target_volatility / max(sigma_hat, 0.001))
        
        # Regime adjustment: 0.5 if high_vol_regime else 1.0
        regime_factor = 0.5 if self.is_high_volatility_regime() else 1.0
        
        # Final scaling multiplier
        multiplier = vol_scaling * regime_factor
        
        logger.debug(f"Scaling multiplier: {multiplier:.3f} (vol_scaling={vol_scaling:.3f}, regime_factor={regime_factor:.3f})")
        logger.debug(f"  σ_hat={sigma_hat:.4f}, target_vol={self.target_volatility:.4f}")
        
        return multiplier
    
    def should_rebalance(self) -> bool:
        """Check if daily rebalance is needed.
        
        Returns:
            True if rebalance needed.
        """
        now = datetime.now()
        hours_since_rebalance = (now - self.last_rebalance_time).total_seconds() / 3600
        return hours_since_rebalance >= config.REBALANCE_HOURS
    
    def rebalance_portfolio(self, active_symbols: List[str]) -> Dict[str, AllocationWeights]:
        """Perform daily portfolio rebalancing using the document's allocation system.
        
        Args:
            active_symbols: List of symbols to rebalance.
            
        Returns:
            Dictionary of allocation weights for each symbol.
        """
        self.last_rebalance_time = datetime.now()
        
        if not active_symbols:
            return {}
        
        # Step 1: Compute raw weights (normalized so they sum to 1)
        weights = self.compute_weights(active_symbols)
        
        # Step 2: Calculate scaling multiplier for volatility targeting
        scaling_multiplier = self.calculate_scaling_multiplier()
        
        # Step 3: Calculate allocations with proper scaling
        # The scaling multiplier should reduce overall exposure during high volatility
        base_allocation = self.total_capital * self.max_allocation_pct
        total_scaled_allocation = base_allocation * scaling_multiplier
        
        new_allocations = {}
        total_check = 0.0
        
        for symbol in active_symbols:
            # Apply normalized weight to scaled total allocation
            weight = weights[symbol]
            allocated_capital = weight * total_scaled_allocation
            
            # Additional safety: cap individual allocation at reasonable percentage
            max_individual_allocation = self.total_capital * 0.50  # Max 50% per asset
            allocated_capital = min(allocated_capital, max_individual_allocation)
            
            allocation = AllocationWeights(
                symbol=symbol,
                weight=weight,
                allocated_capital=allocated_capital,
                volatility=self.get_volatility_ema(symbol),
                avg_correlation=self.get_average_correlation(symbol, active_symbols),
                raw_weight=weight
            )
            
            new_allocations[symbol] = allocation
            total_check += allocated_capital
        
        self.allocation_weights = new_allocations
        
        # Log rebalancing details
        allocation_pct = (total_check / self.total_capital) * 100
        
        logger.info("Daily rebalance completed:")
        logger.info(f"  Total capital: ${self.total_capital:.2f}")
        logger.info(f"  Scaling multiplier: {scaling_multiplier:.3f}")
        logger.info(f"  Max base allocation: {self.max_allocation_pct:.1%} (${base_allocation:.2f})")
        logger.info(f"  Scaled allocation target: ${total_scaled_allocation:.2f}")
        logger.info(f"  Actual allocated: ${total_check:.2f} ({allocation_pct:.1f}%)")
        logger.info(f"  High vol regime: {self.is_high_volatility_regime()}")
        logger.info("  Asset allocations:")
        
        for symbol, alloc in new_allocations.items():
            asset_pct = (alloc.allocated_capital / self.total_capital) * 100
            logger.info(f"    {symbol}: ${alloc.allocated_capital:.2f} ({asset_pct:.1f}% of total) vol={alloc.volatility:.4f}")
        
        return new_allocations
    
    def get_allocated_capital(self, symbol: str) -> float:
        """Get allocated capital for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            Allocated capital in USDT.
        """
        if symbol in self.allocation_weights:
            return self.allocation_weights[symbol].allocated_capital
        return 0.0
    
    def get_all_allocations(self) -> Dict[str, float]:
        """Get all current allocations.
        
        Returns:
            Dictionary mapping symbol to allocated capital.
        """
        return {symbol: weights.allocated_capital 
                for symbol, weights in self.allocation_weights.items()}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary.
        
        Returns:
            Portfolio summary dictionary.
        """
        total_allocated = sum(a.allocated_capital for a in self.allocation_weights.values())
        
        return {
            "total_capital": self.total_capital,
            "allocated_capital": total_allocated,
            "allocation_percentage": total_allocated / self.total_capital,
            "target_volatility": self.target_volatility,
            "high_volatility_regime": self.is_high_volatility_regime(),
            "active_symbols": len(self.allocation_weights),
            "last_rebalance": self.last_rebalance_time.isoformat(),
            "should_rebalance": self.should_rebalance()
        }
    
    def process_symbols_from_config(self) -> None:
        """Initialize symbols from config file."""
        try:
            import config
            symbols = [symbol for symbol, _ in config.symbols]
            logger.info(f"Loaded {len(symbols)} symbols from config")
        except ImportError:
            logger.warning("Warning: Could not load config.symbols")
    
    def reserve_allocation(self, symbol: str, amount: float) -> bool:
        """Reserve allocation for a position.
        
        Args:
            symbol: Trading pair symbol.
            amount: Amount to reserve (margin required).
            
        Returns:
            True if reservation successful, False if would exceed limits.
        """
        current_reserved = self.reserved_allocations.get(symbol, 0.0)
        allocated_capital = self.get_allocated_capital(symbol)
        
        if current_reserved + amount > allocated_capital:
            return False
        
        self.reserved_allocations[symbol] = current_reserved + amount
        return True
    
    def release_allocation(self, symbol: str, amount: float) -> bool:
        """Release reserved allocation.
        
        Args:
            symbol: Trading pair symbol.
            amount: Amount to release.
            
        Returns:
            True if release successful.
        """
        if symbol in self.reserved_allocations:
            self.reserved_allocations[symbol] = max(0.0, self.reserved_allocations[symbol] - amount)
        return True


# Legacy compatibility - map old class to new implementation
PortfolioManager = ProductionPortfolioManager
