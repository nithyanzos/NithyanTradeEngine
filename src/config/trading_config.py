"""
Trading Configuration Module
Comprehensive configuration for institutional-grade algorithmic trading system
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from decimal import Decimal
import logging

@dataclass
class TradingConfig:
    """
    Core trading configuration with NON-NEGOTIABLE capital management rules
    
    CRITICAL CONSTRAINTS:
    - NEVER exceed 50% capital deployment
    - Position sizes: 3-8% of DEPLOYABLE capital (not total capital)
    - Daily loss limit: 2% of total portfolio
    - Maximum drawdown: 8% emergency stop
    """
    
    # ========== CAPITAL MANAGEMENT (NON-NEGOTIABLE) ==========
    TOTAL_CAPITAL: float = 5000000.0  # ₹50 Lakh default
    MAX_CAPITAL_UTILIZATION: float = 0.50  # NEVER exceed 50%
    CASH_RESERVE_RATIO: float = 0.50  # Always maintain 50% cash
    
    # Position Management
    MAX_POSITIONS: int = 15
    MIN_POSITIONS: int = 10
    MAX_POSITION_SIZE: float = 0.08  # 8% of DEPLOYABLE capital
    MIN_POSITION_SIZE: float = 0.03  # 3% of DEPLOYABLE capital
    
    # ========== RISK LIMITS (HARD CONSTRAINTS) ==========
    DAILY_LOSS_LIMIT: float = -0.02  # 2% daily loss triggers full stop
    MAX_DRAWDOWN_LIMIT: float = -0.08  # 8% drawdown triggers emergency stop
    ATR_STOP_MULTIPLIER: float = 2.5
    TRAIL_STOP_MULTIPLIER: float = 1.5
    MAX_HOLDING_DAYS: int = 30
    
    # ========== TRADE FILTERING ==========
    COMPOSITE_SCORE_THRESHOLD: float = 0.70  # Minimum score to trade
    FUNDAMENTAL_WEIGHT: float = 0.25
    TECHNICAL_WEIGHT: float = 0.30
    QUANTITATIVE_WEIGHT: float = 0.25
    MACRO_WEIGHT: float = 0.20
    
    # ========== SECTOR LIMITS ==========
    MAX_SECTOR_ALLOCATION: float = 0.30  # 30% max per sector
    MAX_STOCKS_PER_SECTOR: int = 3
    
    # ========== API CONFIGURATION ==========
    ZERODHA_API_KEY: str = field(default_factory=lambda: os.getenv('ZERODHA_API_KEY', ''))
    ZERODHA_ACCESS_TOKEN: str = field(default_factory=lambda: os.getenv('ZERODHA_ACCESS_TOKEN', ''))
    ZERODHA_API_SECRET: str = field(default_factory=lambda: os.getenv('ZERODHA_API_SECRET', ''))
    
    # Sonar API for sentiment analysis
    SONAR_API_KEY: str = field(default_factory=lambda: os.getenv('SONAR_API_KEY', ''))
    SONAR_BASE_URL: str = "https://api.sonar.com"
    
    # ========== DATABASE CONFIGURATION ==========
    DATABASE_URL: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 
        'postgresql://trading_user:trading_pass@localhost:5432/nifty_trading'))
    REDIS_URL: str = field(default_factory=lambda: os.getenv('REDIS_URL', 
        'redis://localhost:6379/0'))
    
    # ========== EXECUTION PARAMETERS ==========
    ORDER_TIMEOUT_SECONDS: int = 30
    MAX_SLIPPAGE_BPS: int = 20  # 20 basis points
    PARTICIPATION_RATE: float = 0.10  # 10% of average volume
    
    # ========== MONITORING & ALERTS ==========
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    ALERT_EMAIL: str = field(default_factory=lambda: os.getenv('ALERT_EMAIL', ''))
    TELEGRAM_BOT_TOKEN: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', ''))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', ''))
    
    # ========== UNIVERSE CONFIGURATION ==========
    NIFTY_INDICES: List[str] = field(default_factory=lambda: [
        'NIFTY50', 'NIFTY_NEXT50', 'NIFTY_MIDCAP100', 'NIFTY_SMALLCAP100'
    ])
    
    MIN_MARKET_CAP: float = 1000000000  # ₹1000 Cr minimum market cap
    MIN_DAILY_VOLUME: int = 1000000  # Minimum 10 lakh shares daily volume
    MIN_DAILY_VALUE: float = 100000000  # Minimum ₹10 Cr daily value
    
    @property
    def deployable_capital(self) -> float:
        """Calculate deployable capital (50% of total)"""
        return self.TOTAL_CAPITAL * self.MAX_CAPITAL_UTILIZATION
    
    @property
    def max_position_value(self) -> float:
        """Maximum position value in rupees"""
        return self.deployable_capital * self.MAX_POSITION_SIZE
    
    @property
    def min_position_value(self) -> float:
        """Minimum position value in rupees"""
        return self.deployable_capital * self.MIN_POSITION_SIZE
    
    def validate_config(self) -> bool:
        """
        Validate configuration constraints
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If any critical constraint is violated
        """
        
        # Critical capital management constraints
        if self.MAX_CAPITAL_UTILIZATION > 0.50:
            raise ValueError("VIOLATION: Cannot exceed 50% capital deployment")
        
        if self.MAX_POSITION_SIZE > 0.08:
            raise ValueError("VIOLATION: Position size cannot exceed 8%")
        
        if self.DAILY_LOSS_LIMIT > -0.02:
            raise ValueError("VIOLATION: Daily loss limit too aggressive")
        
        if self.MAX_DRAWDOWN_LIMIT > -0.08:
            raise ValueError("VIOLATION: Maximum drawdown limit too aggressive")
        
        # Validate scoring weights sum to 1.0
        total_weight = (self.FUNDAMENTAL_WEIGHT + self.TECHNICAL_WEIGHT + 
                       self.QUANTITATIVE_WEIGHT + self.MACRO_WEIGHT)
        
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"VIOLATION: Scoring weights must sum to 1.0, got {total_weight}")
        
        # Validate API credentials are present
        if not self.ZERODHA_API_KEY:
            logging.warning("Zerodha API key not configured")
        
        if not self.ZERODHA_ACCESS_TOKEN:
            logging.warning("Zerodha access token not configured")
        
        # Validate position limits
        if self.MAX_POSITIONS < self.MIN_POSITIONS:
            raise ValueError("VIOLATION: MAX_POSITIONS must be >= MIN_POSITIONS")
        
        if self.MIN_POSITION_SIZE >= self.MAX_POSITION_SIZE:
            raise ValueError("VIOLATION: MIN_POSITION_SIZE must be < MAX_POSITION_SIZE")
        
        logging.info("✅ Trading configuration validated successfully")
        return True
    
    def get_position_size_range(self) -> Dict[str, float]:
        """Get position sizing range in rupees"""
        return {
            'min_position_inr': self.min_position_value,
            'max_position_inr': self.max_position_value,
            'deployable_capital': self.deployable_capital,
            'total_capital': self.TOTAL_CAPITAL
        }
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get all risk limits"""
        return {
            'daily_loss_limit_inr': self.TOTAL_CAPITAL * abs(self.DAILY_LOSS_LIMIT),
            'max_drawdown_limit_inr': self.TOTAL_CAPITAL * abs(self.MAX_DRAWDOWN_LIMIT),
            'max_sector_allocation_inr': self.deployable_capital * self.MAX_SECTOR_ALLOCATION,
            'max_position_value_inr': self.max_position_value
        }

@dataclass
class BacktestConfig:
    """Configuration for backtesting engine"""
    
    INITIAL_CAPITAL: float = 5000000.0
    START_DATE: str = "2020-01-01"
    END_DATE: str = "2024-12-31"
    BENCHMARK: str = "NIFTY50"
    
    # Transaction costs
    BROKERAGE_RATE: float = 0.0003  # 0.03% (Zerodha equity delivery)
    STT_RATE: float = 0.001  # 0.1% on sell side
    SLIPPAGE_BPS: int = 5  # 5 basis points
    
    # Performance calculation
    RISK_FREE_RATE: float = 0.06  # 6% annual risk-free rate
    
@dataclass
class ExecutionConfig:
    """Configuration for order execution"""
    
    # Order size thresholds for execution strategy
    SMALL_ORDER_THRESHOLD: float = 500000  # ₹5 Lakh
    MEDIUM_ORDER_THRESHOLD: float = 2500000  # ₹25 Lakh
    
    # TWAP parameters
    TWAP_DURATION_MINUTES: int = 30
    TWAP_INTERVALS: int = 6
    
    # Iceberg parameters
    MAX_CHUNK_SIZE_RATIO: float = 0.10  # 10% of average volume
    INTER_CHUNK_DELAY_SECONDS: int = 60
    
    # Market impact limits
    MAX_MARKET_IMPACT_BPS: int = 20
    IMPACT_DECAY_MINUTES: int = 15


# Singleton pattern for global configuration
_config_instance = None

def get_trading_config() -> TradingConfig:
    """Get singleton trading configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = TradingConfig()
        _config_instance.validate_config()
    
    return _config_instance

def get_backtest_config() -> BacktestConfig:
    """Get backtesting configuration"""
    return BacktestConfig()

def get_execution_config() -> ExecutionConfig:
    """Get execution configuration"""
    return ExecutionConfig()

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging for trading system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    import os
    from datetime import datetime
    
    # Create logs directory if it doesn't exist
    log_dir = "/workspaces/NithyanTradeEngine/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('trading_system')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with daily rotation
    log_filename = f'{log_dir}/trading_system_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Detailed formatter for files
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Simple formatter for console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"✅ Logging system initialized - Level: {log_level}")
    logger.info(f"Log file: {log_filename}")
    
    return logger
