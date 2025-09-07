"""
Database Configuration and Models
SQLAlchemy models and database setup for trading system
"""

import os
from datetime import datetime
from decimal import Decimal
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, BigInteger, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()

class Stock(Base):
    """Stock universe table"""
    __tablename__ = 'stocks'
    
    symbol = Column(String(20), primary_key=True)
    company_name = Column(String(200), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap_category = Column(String(20))  # LARGE, MID, SMALL
    market_cap = Column(BigInteger)  # Market cap in rupees
    
    # Index memberships
    is_nifty50 = Column(Boolean, default=False)
    is_nifty_next50 = Column(Boolean, default=False)
    is_nifty_midcap100 = Column(Boolean, default=False)
    is_nifty_smallcap100 = Column(Boolean, default=False)
    
    # Trading eligibility
    is_active = Column(Boolean, default=True)
    listing_date = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Liquidity metrics
    avg_daily_volume = Column(BigInteger)  # 30-day average
    avg_daily_value = Column(BigInteger)   # 30-day average value
    
    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', company='{self.company_name}')>"

class MarketData(Base):
    """OHLCV market data table"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 1d
    
    open = Column(Float(precision=4), nullable=False)
    high = Column(Float(precision=4), nullable=False)
    low = Column(Float(precision=4), nullable=False)
    close = Column(Float(precision=4), nullable=False)
    volume = Column(BigInteger, nullable=False)
    trades_count = Column(Integer)
    vwap = Column(Float(precision=4))
    
    # Technical indicators (calculated and stored)
    sma_20 = Column(Float(precision=4))
    sma_50 = Column(Float(precision=4))
    sma_200 = Column(Float(precision=4))
    rsi = Column(Float(precision=4))
    atr = Column(Float(precision=4))
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"

class FundamentalData(Base):
    """Fundamental analysis data"""
    __tablename__ = 'fundamental_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    report_date = Column(DateTime, nullable=False)
    quarter = Column(String(10))  # Q1FY24, Q2FY24, etc.
    
    # Valuation ratios
    pe_ratio = Column(Float(precision=4))
    pb_ratio = Column(Float(precision=4))
    ps_ratio = Column(Float(precision=4))
    ev_ebitda = Column(Float(precision=4))
    
    # Profitability metrics
    roe = Column(Float(precision=6))  # Return on Equity
    roa = Column(Float(precision=6))  # Return on Assets
    roic = Column(Float(precision=6))  # Return on Invested Capital
    gross_margin = Column(Float(precision=6))
    operating_margin = Column(Float(precision=6))
    net_margin = Column(Float(precision=6))
    
    # Financial health
    debt_equity = Column(Float(precision=6))
    current_ratio = Column(Float(precision=4))
    quick_ratio = Column(Float(precision=4))
    interest_coverage = Column(Float(precision=4))
    
    # Growth metrics
    revenue_growth_yoy = Column(Float(precision=6))
    profit_growth_yoy = Column(Float(precision=6))
    eps_growth_yoy = Column(Float(precision=6))
    
    # Ownership data
    promoter_holding = Column(Float(precision=6))
    institutional_holding = Column(Float(precision=6))
    retail_holding = Column(Float(precision=6))
    
    # Raw financial data (JSON for flexibility)
    financial_data = Column(JSONB)
    
    def __repr__(self):
        return f"<FundamentalData(symbol='{self.symbol}', date='{self.report_date}', roe={self.roe})>"

class Trade(Base):
    """Individual trade records"""
    __tablename__ = 'trades'
    
    trade_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    
    # Order details
    side = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float(precision=4))
    exit_price = Column(Float(precision=4))
    
    # Timing
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    holding_period_days = Column(Integer)
    
    # P&L calculation
    pnl_gross = Column(Float(precision=2))
    pnl_net = Column(Float(precision=2))
    commission = Column(Float(precision=2))
    slippage = Column(Float(precision=4))
    
    # Trade status
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    exit_reason = Column(String(50))  # STOP_LOSS, TAKE_PROFIT, TIME_EXIT, FUNDAMENTAL_DETERIORATION
    
    # Entry scores (for analysis)
    composite_score = Column(Float(precision=3))
    fundamental_score = Column(Float(precision=3))
    technical_score = Column(Float(precision=3))
    quantitative_score = Column(Float(precision=3))
    macro_score = Column(Float(precision=3))
    conviction_level = Column(Float(precision=3))
    
    # Risk management
    initial_stop_loss = Column(Float(precision=4))
    trailing_stop_loss = Column(Float(precision=4))
    position_size_pct = Column(Float(precision=4))  # % of portfolio
    
    # Additional metadata
    market_regime = Column(String(20))  # BULL, BEAR, VOLATILE, NORMAL
    volatility_at_entry = Column(Float(precision=4))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Trade(id='{self.trade_id}', symbol='{self.symbol}', side='{self.side}', pnl={self.pnl_net})>"

class Portfolio(Base):
    """Portfolio positions tracking"""
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    
    # Position details
    quantity = Column(Integer, nullable=False)
    average_price = Column(Float(precision=4), nullable=False)
    current_price = Column(Float(precision=4))
    market_value = Column(Float(precision=2))
    
    # P&L tracking
    unrealized_pnl = Column(Float(precision=2))
    realized_pnl = Column(Float(precision=2))
    
    # Risk management
    stop_loss_price = Column(Float(precision=4))
    trailing_stop_price = Column(Float(precision=4))
    
    # Metadata
    entry_date = Column(DateTime, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Reference to original trade
    trade_id = Column(String(36), ForeignKey('trades.trade_id'))
    trade = relationship("Trade", backref="portfolio_position")
    
    def __repr__(self):
        return f"<Portfolio(symbol='{self.symbol}', quantity={self.quantity}, value={self.market_value})>"

class PerformanceMetrics(Base):
    """Daily portfolio performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    
    # Portfolio values
    total_value = Column(Float(precision=2), nullable=False)
    cash_balance = Column(Float(precision=2), nullable=False)
    invested_value = Column(Float(precision=2), nullable=False)
    
    # Daily P&L
    daily_pnl = Column(Float(precision=2))
    daily_pnl_pct = Column(Float(precision=6))
    
    # Cumulative returns
    total_return = Column(Float(precision=6))
    total_return_pct = Column(Float(precision=6))
    
    # Risk metrics
    volatility = Column(Float(precision=6))
    sharpe_ratio = Column(Float(precision=4))
    max_drawdown = Column(Float(precision=6))
    current_drawdown = Column(Float(precision=6))
    
    # Position metrics
    position_count = Column(Integer)
    capital_utilization = Column(Float(precision=4))
    
    # Trade statistics
    trades_today = Column(Integer, default=0)
    win_rate_30d = Column(Float(precision=4))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PerformanceMetrics(date='{self.date}', value={self.total_value}, pnl={self.daily_pnl})>"

class AlertLog(Base):
    """System alerts and notifications"""
    __tablename__ = 'alert_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False)  # RISK_BREACH, POSITION_ALERT, SYSTEM_ERROR
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Context data
    symbol = Column(String(20))
    trade_id = Column(String(36))
    portfolio_value = Column(Float(precision=2))
    
    # Alert status
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AlertLog(type='{self.alert_type}', severity='{self.severity}', title='{self.title}')>"

class SystemLog(Base):
    """System operations and audit log"""
    __tablename__ = 'system_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    log_level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    module = Column(String(100), nullable=False)
    function = Column(String(100))
    
    message = Column(Text, nullable=False)
    details = Column(JSONB)  # Additional structured data
    
    # Request context
    session_id = Column(String(100))
    user_id = Column(String(100))
    
    def __repr__(self):
        return f"<SystemLog(level='{self.log_level}', module='{self.module}', message='{self.message[:50]}...')>"


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        
    def initialize(self):
        """Initialize database connection and create tables"""
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=os.getenv('SQL_ECHO', 'False').lower() == 'true'
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_engine(self):
        """Get database engine"""
        return self.engine


# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get singleton database manager"""
    global _db_manager
    
    if _db_manager is None:
        from .trading_config import get_trading_config
        config = get_trading_config()
        _db_manager = DatabaseManager(config.DATABASE_URL)
        _db_manager.initialize()
    
    return _db_manager

def get_db_session():
    """Get database session for dependency injection"""
    db_manager = get_database_manager()
    return next(db_manager.get_session())
