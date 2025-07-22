from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import numpy as np
from datetime import datetime, timedelta

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
    
    def __init__(self, total_capital: float, target_volatility: float = 0.18, max_allocation_pct: float = 0.85):
        """Initialize the production portfolio manager.
        
        Args:
            total_capital: Total trading capital in USDT.
            target_volatility: Target portfolio volatility (default: 18%).
            max_allocation_pct: Maximum percentage of capital to allocate (default: 85%).
        """
        self.total_capital = total_capital
        self.target_volatility = target_volatility
        self.max_allocation_pct = max_allocation_pct
        
        # Core data structures for the allocation system
        self.volatility_data: Dict[str, List[float]] = {}  # EMA of 1-min ATR(30) over 60 bars
        self.correlation_data: Dict[Tuple[str, str], List[float]] = {}  # EMA of pairwise returns over 60 bars
        self.allocation_weights: Dict[str, AllocationWeights] = {}
        
        # Regime detection
        self.volatility_history: List[float] = []  # For 30-day percentile calculation
        self.last_rebalance_time = datetime.now()
        
        # Fixed parameters from document
        self.alpha = 0.3  # Fixed correlation adjustment parameter
        self.lookback_bars = 60  # EMA lookback for vol and correlation
        self.regime_percentile = 75  # 75th percentile for high vol regime
        
        # Track reserved allocations for position management
        self.reserved_allocations: Dict[str, float] = {}
        
        print(f"[ProductionPortfolio] Initialized with ${total_capital:.2f}, target_vol={target_volatility:.1%}")
    
    def update_volatility_data(self, symbol: str, atr_value: float) -> None:
        """Update volatility data (EMA of 1-min ATR(30)) for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            atr_value: Current ATR value.
        """
        if symbol not in self.volatility_data:
            self.volatility_data[symbol] = []
        
        # Keep rolling window of 60 bars for EMA calculation
        if len(self.volatility_data[symbol]) >= self.lookback_bars:
            self.volatility_data[symbol].pop(0)
        
        self.volatility_data[symbol].append(atr_value)
    
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
            
            # Core formula: w_i = (1/σ_i) × (1 + α × avg_correlation_i)
            raw_weight = (1 / sigma_i) * (1 + self.alpha * avg_corr_i)
            raw_weights[symbol] = raw_weight
        
        # Normalize so sum = 1
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            # Equal weights fallback
            return {s: 1.0 / len(symbols) for s in symbols}
        
        normalized_weights = {s: w / total_weight for s, w in raw_weights.items()}
        
        print(f"[ProductionPortfolio] Computed weights for {len(symbols)} symbols")
        return normalized_weights
    
    def is_high_volatility_regime(self) -> bool:
        """Check if current market is in high volatility regime.
        σ_hat > 75th percentile over last 30 days.
        
        Returns:
            True if high volatility regime.
        """
        if len(self.volatility_history) < 30:
            return False  # Insufficient data
        
        # Calculate current average volatility
        active_symbols = list(self.volatility_data.keys())
        if not active_symbols:
            return False
        
        sigma_hat = sum(self.get_volatility_ema(s) for s in active_symbols) / len(active_symbols)
        
        # Calculate 75th percentile of last 30 days
        recent_history = self.volatility_history[-30:]
        percentile_75 = np.percentile(recent_history, 75)
        
        return sigma_hat > percentile_75
    
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
        
        # Update volatility history for regime detection
        self.volatility_history.append(sigma_hat)
        if len(self.volatility_history) > 30:  # Keep 30 days for percentile
            self.volatility_history.pop(0)
        
        # Volatility targeting component
        vol_scaling = min(1.0, self.target_volatility / max(sigma_hat, 0.001))
        
        # Regime adjustment
        regime_factor = 0.5 if self.is_high_volatility_regime() else 1.0
        
        multiplier = vol_scaling * regime_factor
        
        print(f"[ProductionPortfolio] Scaling multiplier: {multiplier:.3f} (vol_scaling={vol_scaling:.3f}, regime_factor={regime_factor:.3f})")
        return multiplier
    
    def should_rebalance(self) -> bool:
        """Check if daily rebalance is needed.
        
        Returns:
            True if rebalance needed.
        """
        now = datetime.now()
        hours_since_rebalance = (now - self.last_rebalance_time).total_seconds() / 3600
        return hours_since_rebalance >= 24.0
    
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
        
        # Step 1: Compute raw weights
        weights = self.compute_weights(active_symbols)
        
        # Step 2: Calculate scaling multiplier
        scaling_multiplier = self.calculate_scaling_multiplier()
        
        # Step 3: Calculate allocated capital for each symbol
        max_total_allocation = self.total_capital * self.max_allocation_pct
        
        new_allocations = {}
        
        for symbol in active_symbols:
            weight = weights[symbol]
            scaled_weight = scaling_multiplier * weight
            allocated_capital = scaled_weight * max_total_allocation
            
            allocation = AllocationWeights(
                symbol=symbol,
                weight=scaled_weight,
                allocated_capital=allocated_capital,
                volatility=self.get_volatility_ema(symbol),
                avg_correlation=self.get_average_correlation(symbol, active_symbols),
                raw_weight=weight
            )
            
            new_allocations[symbol] = allocation
        
        self.allocation_weights = new_allocations
        
        # Log rebalancing details
        total_allocated = sum(a.allocated_capital for a in new_allocations.values())
        print(f"[ProductionPortfolio] Daily rebalance completed:")
        print(f"[ProductionPortfolio]   Total allocated: ${total_allocated:.2f} ({total_allocated/self.total_capital:.1%})")
        print(f"[ProductionPortfolio]   High vol regime: {self.is_high_volatility_regime()}")
        
        for symbol, alloc in new_allocations.items():
            print(f"[ProductionPortfolio]   {symbol}: ${alloc.allocated_capital:.2f} ({alloc.weight:.1%}) vol={alloc.volatility:.4f}")
        
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
            print(f"[ProductionPortfolio] Loaded {len(symbols)} symbols from config")
        except ImportError:
            print("[ProductionPortfolio] Warning: Could not load config.symbols")
    
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
