"""
Centralized logging configuration for the Crypto Trading Algorithm.

This module provides a unified logging setup that:
1. Creates a proper folder structure under logs/
2. Separates different types of logs into dedicated files
3. Provides console output for print-like messages
4. Saves detailed logs to files only for internal operations
"""

import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

def setup_logging_structure():
    """Create the logging directory structure."""
    # Main logs directory
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Subdirectories for different components
    (LOGS_DIR / "data").mkdir(exist_ok=True)
    (LOGS_DIR / "algorithm").mkdir(exist_ok=True)
    (LOGS_DIR / "execution").mkdir(exist_ok=True)
    (LOGS_DIR / "backtest").mkdir(exist_ok=True)
    (LOGS_DIR / "errors").mkdir(exist_ok=True)

def configure_logger():
    """Configure the centralized logger with proper handlers."""
    # Remove any existing handlers
    logger.remove()
    
    # Create logs directory structure
    setup_logging_structure()
    
    # Console handler for user-facing messages (print replacements)
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        filter=lambda record: record["extra"].get("console", False)
    )
    
    # Main trading log - comprehensive activity
    logger.add(
        LOGS_DIR / "trading_main.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        compression="zip"
    )
    
    # Data engine logs
    logger.add(
        LOGS_DIR / "data" / "data_engine.log",
        rotation="5 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        filter=lambda record: "data" in record["name"].lower(),
        compression="zip"
    )
    
    # Algorithm engine logs
    logger.add(
        LOGS_DIR / "algorithm" / "algo_engine.log",
        rotation="5 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        filter=lambda record: "algorithm" in record["name"].lower() or "strategy" in record["name"].lower(),
        compression="zip"
    )
    
    # Execution engine logs
    logger.add(
        LOGS_DIR / "execution" / "execution_engine.log",
        rotation="5 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        filter=lambda record: "execution" in record["name"].lower(),
        compression="zip"
    )
    
    # Backtest logs
    logger.add(
        LOGS_DIR / "backtest" / "backtest.log",
        rotation="10 MB",
        retention="14 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        filter=lambda record: "backtest" in record["name"].lower(),
        compression="zip"
    )
    
    # Error-only log for critical issues
    logger.add(
        LOGS_DIR / "errors" / "errors.log",
        rotation="10 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        compression="zip"
    )

def get_logger(name: str):
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)

def console_log(message: str, level: str = "INFO"):
    """Log a message that should be displayed on console (replacement for print).
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, etc.)
    """
    logger.bind(console=True).log(level, message)

# Initialize logging when module is imported
configure_logger()
