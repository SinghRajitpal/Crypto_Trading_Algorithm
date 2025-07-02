from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SymbolAllocation:
    """Represents allocation and tier information for a trading symbol.
    
    Attributes:
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        tier (int): Tier level (1, 2, or 3) based on market cap.
        max_allocation_pct (float): Maximum percentage of total capital for this symbol.
        current_allocation (float): Current allocation in USDT.
        is_active (bool): Whether this symbol has an active position.
    """
    symbol: str
    tier: int
    max_allocation_pct: float
    current_allocation: float = 0.0
    is_active: bool = False


class PortfolioManager:
    """Portfolio Manager for crypto futures trading.
    
    Manages capital allocation across multiple symbols using a tier-based system.
    Sets allocation limits based on symbol tiers and monitors overall exposure.
    
    Attributes:
        total_capital (float): Total trading capital in USDT.
        max_allocation_pct (float): Maximum percentage of capital to allocate.
        symbol_allocations (Dict[str, SymbolAllocation]): Allocations by symbol.
        max_tier_allocation_pct (Dict[int, float]): Maximum allocation per tier.
        max_concurrent_positions (Dict[int, int]): Maximum positions per tier.
    """
    
    # Default tier assignments for top coins by market cap
    # These can be overridden when adding symbols
    DEFAULT_TIERS = {
        "BTCUSDT": 1, "ETHUSDT": 1,
        "BNBUSDT": 2, "SOLUSDT": 2, "XRPUSDT": 2, "ADAUSDT": 2, "AVAXUSDT": 2,
        # All other symbols default to tier 3
    }
    
    # Default maximum allocation percentages per tier
    DEFAULT_MAX_ALLOCATIONS = {
        1: 0.10,  # 10% for Tier 1
        2: 0.06,  # 6% for Tier 2
        3: 0.03   # 3% for Tier 3
    }
    
    def __init__(self, total_capital: float, max_allocation_pct: float = 0.5):
        """Initialize the portfolio manager.
        
        Args:
            total_capital: Total trading capital in USDT.
            max_allocation_pct: Maximum percentage of total capital to allocate (default: 0.5 or 50%).
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if total_capital <= 0:
            raise ValueError("Total capital must be positive")
        if not (0 < max_allocation_pct <= 1):
            raise ValueError("Max allocation percentage must be between 0 and 1")
            
        self.total_capital = total_capital
        self.max_allocation_pct = max_allocation_pct
        self.symbol_allocations: Dict[str, SymbolAllocation] = {}
        
        # Maximum allocation percentage per tier (of total capital)
        self.max_tier_allocation_pct = {
            1: 0.20,  # 20% for Tier 1 coins combined
            2: 0.20,  # 20% for Tier 2 coins combined
            3: 0.15   # 15% for Tier 3 coins combined
        }
        
        # Maximum concurrent positions per tier
        self.max_concurrent_positions = {
            1: 2,  # Max 2 Tier 1 positions at once
            2: 3,  # Max 3 Tier 2 positions at once
            3: 4   # Max 4 Tier 3 positions at once
        }
        
        print(f"[Portfolio] Initialized with {total_capital} USDT capital.")
    
    def add_symbol(self, symbol: str, tier: Optional[int] = None, max_allocation_pct: Optional[float] = None) -> None:
        """Add a symbol to the portfolio.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT").
            tier: Tier level (1, 2, or 3) - defaults to preset tier or tier 3.
            max_allocation_pct: Maximum percentage allocation for this symbol.
                If not provided, uses default based on tier.
                
        Raises:
            ValueError: If parameters are invalid.
        """
        # Determine tier if not provided
        if tier is None:
            tier = self.DEFAULT_TIERS.get(symbol, 3)
            
        # Validate tier
        if tier not in [1, 2, 3]:
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")
            
        # Use default allocation percentage if not provided
        if max_allocation_pct is None:
            max_allocation_pct = self.DEFAULT_MAX_ALLOCATIONS[tier]
        
        # Validate allocation percentage
        if not (0 < max_allocation_pct <= 1):
            raise ValueError(f"Maximum allocation percentage must be between 0 and 1, got {max_allocation_pct}")
            
        # Create and store allocation
        self.symbol_allocations[symbol] = SymbolAllocation(
            symbol=symbol,
            tier=tier,
            max_allocation_pct=max_allocation_pct
        )
        
        print(f"[Portfolio] Added {symbol} to portfolio as Tier {tier} with {max_allocation_pct:.1%} max allocation")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the portfolio.
        
        Args:
            symbol: Trading pair symbol to remove.
            
        Raises:
            KeyError: If the symbol doesn't exist in the portfolio.
        """
        if symbol not in self.symbol_allocations:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
            
        if self.symbol_allocations[symbol].current_allocation > 0:
            print(f"[Portfolio] WARNING: Removing {symbol} with non-zero allocation: {self.symbol_allocations[symbol].current_allocation} USDT")
            
        del self.symbol_allocations[symbol]
        print(f"[Portfolio] Removed {symbol} from portfolio")
    
    def get_tier(self, symbol: str) -> int:
        """Get the tier for a specific symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            Tier level (1, 2, or 3).
            
        Raises:
            KeyError: If the symbol isn't in the portfolio.
        """
        if symbol not in self.symbol_allocations:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
            
        return self.symbol_allocations[symbol].tier
    
    def get_allocation_limit(self, symbol: str) -> float:
        """Get the maximum allocation amount for a symbol in USDT.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            Maximum allocation amount in USDT.
            
        Raises:
            KeyError: If the symbol isn't in the portfolio.
        """
        if symbol not in self.symbol_allocations:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
            
        allocation = self.symbol_allocations[symbol]
        
        # Calculate symbol-specific limit
        symbol_limit = self.total_capital * allocation.max_allocation_pct
        
        # Calculate tier limit (remaining allocation for this tier)
        tier = allocation.tier
        current_tier_allocation = self._get_tier_allocation(tier)
        max_tier_allocation = self.total_capital * self.max_tier_allocation_pct[tier]
        tier_limit = max(0, max_tier_allocation - current_tier_allocation)
        
        # Calculate global limit (remaining allocation for entire portfolio)
        current_total_allocation = self._get_total_allocation()
        max_total_allocation = self.total_capital * self.max_allocation_pct
        global_limit = max(0, max_total_allocation - current_total_allocation)
        
        # Return minimum of all limits
        return min(symbol_limit, tier_limit, global_limit)
    
    def can_allocate(self, symbol: str, amount: float) -> bool:
        """Check if an allocation is possible for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            amount: Amount to allocate in USDT.
            
        Returns:
            True if allocation is possible, False otherwise.
            
        Raises:
            KeyError: If the symbol isn't in the portfolio.
        """
        if symbol not in self.symbol_allocations:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
            
        # Check if amount is positive
        if amount <= 0:
            print(f"[Portfolio] Cannot allocate non-positive amount: {amount}")
            return False
            
        # Check against allocation limit
        if amount > self.get_allocation_limit(symbol):
            return False
            
        # Check concurrent position limit for this tier
        tier = self.symbol_allocations[symbol].tier
        active_tier_positions = sum(1 for s in self.symbol_allocations.values() 
                                  if s.tier == tier and s.is_active)
                                  
        # If we already have a position in this symbol, don't count it toward the limit
        if self.symbol_allocations[symbol].is_active:
            active_tier_positions -= 1
            
        if active_tier_positions >= self.max_concurrent_positions[tier]:
            print(f"[Portfolio] Cannot allocate: maximum concurrent positions ({self.max_concurrent_positions[tier]}) reached for Tier {tier}")
            return False
            
        return True
    
    def reserve_allocation(self, symbol: str, amount: float) -> bool:
        """Reserve capital allocation for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            amount: Amount to allocate in USDT.
            
        Returns:
            True if allocation was successful, False otherwise.
            
        Raises:
            KeyError: If the symbol isn't in the portfolio.
        """
        if not self.can_allocate(symbol, amount):
            print(f"[Portfolio] Cannot allocate {amount} USDT to {symbol}")
            return False
            
        # Update allocation
        self.symbol_allocations[symbol].current_allocation += amount
        self.symbol_allocations[symbol].is_active = True
        
        print(f"[Portfolio] Reserved {amount} USDT for {symbol} (total: {self.symbol_allocations[symbol].current_allocation} USDT)")
        return True
    
    def release_allocation(self, symbol: str, amount: Optional[float] = None) -> float:
        """Release capital allocation for a symbol.
        
        Args:
            symbol: Trading pair symbol.
            amount: Amount to release in USDT. If None, releases all allocation.
            
        Returns:
            Amount actually released in USDT.
            
        Raises:
            KeyError: If the symbol isn't in the portfolio.
        """
        if symbol not in self.symbol_allocations:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
            
        allocation = self.symbol_allocations[symbol]
        
        # Determine amount to release
        if amount is None or amount >= allocation.current_allocation:
            amount_to_release = allocation.current_allocation
            allocation.current_allocation = 0.0
            allocation.is_active = False
        else:
            amount_to_release = amount
            allocation.current_allocation -= amount
            
        print(f"[Portfolio] Released {amount_to_release} USDT from {symbol} (remaining: {allocation.current_allocation} USDT)")
        return amount_to_release
    
    def get_active_allocations(self) -> Dict[str, float]:
        """Get all active allocations with their amounts.
        
        Returns:
            Dictionary mapping symbols to allocation amounts in USDT.
        """
        return {symbol: alloc.current_allocation 
                for symbol, alloc in self.symbol_allocations.items() 
                if alloc.current_allocation > 0}
    
    def get_active_symbols(self) -> List[str]:
        """Get list of symbols with active positions.
        
        Returns:
            List of symbols with active positions.
        """
        return [symbol for symbol, alloc in self.symbol_allocations.items() 
                if alloc.is_active]
    
    def get_available_capital(self) -> float:
        """Get available capital for new allocations.
        
        Returns:
            Available capital in USDT.
        """
        current_allocation = self._get_total_allocation()
        max_allocation = self.total_capital * self.max_allocation_pct
        return max(0, max_allocation - current_allocation)
    
    def _get_total_allocation(self) -> float:
        """Get total allocated capital across all symbols.
        
        Returns:
            Total allocated capital in USDT.
        """
        return sum(alloc.current_allocation for alloc in self.symbol_allocations.values())
    
    def _get_tier_allocation(self, tier: int) -> float:
        """Get total allocated capital for a specific tier.
        
        Args:
            tier: Tier level (1, 2, or 3).
            
        Returns:
            Total allocated capital for the tier in USDT.
        """
        return sum(alloc.current_allocation for alloc in self.symbol_allocations.values() 
                  if alloc.tier == tier)
    
    def get_allocation_percentage(self) -> float:
        """Get current percentage of total capital that is allocated.
        
        Returns:
            Percentage of total capital allocated (0.0-1.0).
        """
        return self._get_total_allocation() / self.total_capital if self.total_capital > 0 else 0.0
    
    def get_symbol_allocation(self, symbol: str) -> float:
        """Get current allocation for a specific symbol.
        
        Args:
            symbol: Trading pair symbol.
            
        Returns:
            Current allocation in USDT.
            
        Raises:
            KeyError: If the symbol isn't in the portfolio.
        """
        if symbol not in self.symbol_allocations:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
            
        return self.symbol_allocations[symbol].current_allocation
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get summary of portfolio state.
        
        Returns:
            Dictionary with portfolio summary information.
        """
        tier_allocations = {tier: self._get_tier_allocation(tier) for tier in range(1, 4)}
        tier_percentages = {tier: amount / self.total_capital for tier, amount in tier_allocations.items()}
        
        return {
            "total_capital": self.total_capital,
            "allocated_capital": self._get_total_allocation(),
            "allocation_percentage": self.get_allocation_percentage(),
            "tier_allocations": tier_allocations,
            "tier_percentages": tier_percentages,
            "active_positions": len(self.get_active_symbols())
        }
    
    def process_all_symbols(self) -> None:
        """Process and add all symbols from config.
        
        This method is a convenience function to automatically add all 
        symbols from the config.py file with default tier assignments.
        """
        import config
        
        for symbol, _ in config.symbols:
            if symbol not in self.symbol_allocations:
                self.add_symbol(symbol)
                
        print(f"[Portfolio] Processed {len(config.symbols)} symbols from config")
