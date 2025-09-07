# GitHub Copilot Instructions - Nifty Universe Trading System
## Complete Development Context & Code Generation Guidelines

## 🎯 Project Context & Core Principles
You are building an institutional-grade algorithmic trading system for the Indian Nifty universe (large, mid, small cap stocks) with the following non-negotiable principles:

### Capital Management Philosophy
- **NEVER exceed 50% capital deployment** - Always maintain 50% cash reserve
- **Position limits**: 3-8% of deployable capital per position (NOT total capital)
- **Maximum 15 positions**, minimum 10 positions at any time
- **Sector limits**: Maximum 30% allocation to any single sector
- **Daily loss limit**: Stop all trading if portfolio loses >2% in single day
- **Maximum drawdown**: Emergency stop if portfolio drawdown exceeds 8%

### Trade Selection Philosophy  
- **Quality over quantity**: Filter ~300 Nifty stocks to top 10-15 opportunities only
- **Composite scoring**: 25% fundamental + 30% technical + 25% quantitative + 20% macro
- **Minimum score threshold**: Only trade opportunities with composite score ≥ 0.70
- **Risk-adjusted sizing**: Position size based on volatility (ATR), conviction, and correlation

## 🏗️ Code Architecture Standards

### 1. Directory Structure (ALWAYS follow this)
```
nifty-trading-system/
├── src/
│   ├── config/
│   │   ├── trading_config.py
│   │   ├── api_config.py
│   │   └── database_config.py
│   ├── data_pipeline/
│   │   ├── zerodha_connector.py
│   │   ├── sonar_connector.py
│   │   ├── data_processor.py
│   │   └── market_data_manager.py
│   ├── analytics/
│   │   ├── fundamental_analyzer.py
│   │   ├── technical_analyzer.py
│   │   ├── quantitative_engine.py
│   │   └── sentiment_analyzer.py
│   ├── strategy/
│   │   ├── trade_filter.py
│   │   ├── position_manager.py
│   │   ├── risk_manager.py
│   │   └── strategy_executor.py
│   ├── backtesting/
│   │   ├── backtest_engine.py
│   │   ├── performance_analyzer.py
│   │   └── optimization_engine.py
│   ├── trading/
│   │   ├── order_manager.py
│   │   ├── portfolio_tracker.py
│   │   └── execution_engine.py
│   ├── frontend/
│   │   ├── dashboard.py
│   │   ├── charts.py
│   │   └── controls.py
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── backtests/
```

### 2. Essential Imports Template
```python
# ALWAYS include these imports for trading system files
import pandas as pd
import numpy as np
import talib as ta
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from decimal import Decimal
import uuid

# Trading specific
from kiteconnect import KiteConnect
import redis
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

# Configuration
from config.trading_config import TradingConfig
```

### 3. Configuration Management Pattern
```python
# ALWAYS use dataclass configuration pattern
@dataclass
class TradingConfig:
    # Capital Management - NEVER change these core principles
    TOTAL_CAPITAL: float = 5000000  # 50 lakh default
    MAX_CAPITAL_UTILIZATION: float = 0.50  # NEVER exceed 50%
    CASH_RESERVE_RATIO: float = 0.50  # Always maintain 50% cash
    
    # Position Management
    MAX_POSITIONS: int = 15
    MIN_POSITIONS: int = 10
    MAX_POSITION_SIZE: float = 0.08  # 8% of DEPLOYABLE capital
    MIN_POSITION_SIZE: float = 0.03  # 3% of DEPLOYABLE capital
    
    # Risk Limits (NON-NEGOTIABLE)
    DAILY_LOSS_LIMIT: float = -0.02  # 2% daily loss triggers full stop
    MAX_DRAWDOWN_LIMIT: float = -0.08  # 8% drawdown triggers emergency stop
    ATR_STOP_MULTIPLIER: float = 2.5
    TRAIL_STOP_MULTIPLIER: float = 1.5
    MAX_HOLDING_DAYS: int = 30
    
    # Trade Filtering
    COMPOSITE_SCORE_THRESHOLD: float = 0.70  # Minimum score to trade
    FUNDAMENTAL_WEIGHT: float = 0.25
    TECHNICAL_WEIGHT: float = 0.30
    QUANTITATIVE_WEIGHT: float = 0.25
    MACRO_WEIGHT: float = 0.20
    
    # Sector Limits
    MAX_SECTOR_ALLOCATION: float = 0.30  # 30% max per sector
    MAX_STOCKS_PER_SECTOR: int = 3
    
    @property
    def deployable_capital(self) -> float:
        """Calculate deployable capital (50% of total)"""
        return self.TOTAL_CAPITAL * self.MAX_CAPITAL_UTILIZATION
    
    def validate_config(self) -> bool:
        """Validate configuration constraints"""
        assert self.MAX_CAPITAL_UTILIZATION <= 0.50, "Cannot exceed 50% capital deployment"
        assert self.MAX_POSITION_SIZE <= 0.08, "Position size too large"
        assert self.DAILY_LOSS_LIMIT >= -0.02, "Daily loss limit too aggressive"
        return True
```

## 🔍 Trade Filtering Implementation Patterns

### 1. Trade Filter Class Template
```python
class TradeFilter:
    """
    ALWAYS implement trade filtering with these exact principles:
    - Filter ~300 Nifty stocks to exactly 10-15 opportunities
    - Use 4-factor composite scoring (Fundamental 25%, Technical 30%, Quantitative 25%, Macro 20%)
    - Apply minimum score threshold of 0.70
    - Include detailed rationale for each selection
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.score_weights = {
            'fundamental': config.FUNDAMENTAL_WEIGHT,
            'technical': config.TECHNICAL_WEIGHT,
            'quantitative': config.QUANTITATIVE_WEIGHT,
            'macro': config.MACRO_WEIGHT
        }
        self.logger = logging.getLogger(__name__)
    
    def calculate_composite_score(self, symbol: str) -> Dict[str, Union[float, Dict]]:
        """
        ALWAYS implement composite scoring with this structure:
        1. Calculate individual factor scores (0-1 scale)
        2. Apply weights to get composite score
        3. Return detailed breakdown for transparency
        4. Include investment rationale
        """
        
        # Individual factor scores
        fund_score = self.calculate_fundamental_score(symbol)
        tech_score = self.calculate_technical_score(symbol)
        quant_score = self.calculate_quantitative_score(symbol)
        macro_score = self.calculate_macro_score(symbol)
        
        # Weighted composite
        composite = (
            fund_score['score'] * self.score_weights['fundamental'] +
            tech_score['score'] * self.score_weights['technical'] +
            quant_score['score'] * self.score_weights['quantitative'] +
            macro_score['score'] * self.score_weights['macro']
        )
        
        return {
            'symbol': symbol,
            'composite_score': round(composite, 3),
            'individual_scores': {
                'fundamental': fund_score,
                'technical': tech_score,  
                'quantitative': quant_score,
                'macro': macro_score
            },
            'meets_threshold': composite >= self.config.COMPOSITE_SCORE_THRESHOLD,
            'conviction_level': self._calculate_conviction(composite),
            'investment_rationale': self._generate_rationale(symbol, fund_score, tech_score, quant_score, macro_score)
        }
    
    def filter_top_trades(self) -> List[Dict]:
        """
        ALWAYS implement filtering with these steps:
        1. Get tradeable universe (~300 Nifty stocks)
        2. Score all stocks using composite scoring
        3. Filter by minimum threshold (0.70)
        4. Apply quality screens (liquidity, volatility, news sentiment)
        5. Apply diversification limits (sector concentration)
        6. Return exactly 10-15 top opportunities
        """
        
        universe = self.get_tradeable_universe()  # ~300 stocks
        
        # Score all stocks
        scored_opportunities = []
        for symbol in universe:
            try:
                score_result = self.calculate_composite_score(symbol)
                if score_result['meets_threshold']:
                    scored_opportunities.append(score_result)
            except Exception as e:
                self.logger.warning(f"Scoring failed for {symbol}: {e}")
        
        # Sort by composite score
        scored_opportunities.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Apply quality filters
        quality_filtered = self._apply_quality_filters(scored_opportunities)
        
        # Apply diversification filters
        diversified = self._apply_diversification_filters(quality_filtered)
        
        # Return top 10-15
        final_selection = diversified[:self.config.MAX_POSITIONS]
        
        self.logger.info(f"Filtered {len(universe)} stocks to {len(final_selection)} opportunities")
        return final_selection
    
    def calculate_fundamental_score(self, symbol: str) -> Dict:
        """
        ALWAYS implement fundamental scoring with these components:
        
        Profitability (35 points):
        - ROE > 15% (15 points)
        - Operating margin > 10% (10 points)  
        - ROIC > 12% (10 points)
        
        Financial Health (30 points):
        - Debt/Equity < 0.5 (15 points)
        - Interest coverage > 5x (10 points)
        - Current ratio > 1.2 (5 points)
        
        Growth (25 points):
        - Revenue CAGR > 10% (10 points)
        - EPS growth consistency (10 points)
        - Book value growth (5 points)
        
        Governance (10 points):
        - Promoter holding > 50% (5 points)
        - Institutional growth (5 points)
        
        Return: {'score': 0.0-1.0, 'breakdown': {...}, 'raw_total': 0-100}
        """
        
        financials = self.get_financial_data(symbol)
        score = 0
        breakdown = {}
        
        # Profitability scoring
        if financials.get('roe', 0) > 0.15:
            score += 15
            breakdown['roe'] = 15
        
        if financials.get('operating_margin', 0) > 0.10:
            score += 10
            breakdown['operating_margin'] = 10
        
        # Add remaining scoring logic following same pattern...
        
        normalized_score = score / 100.0  # Convert to 0-1 scale
        
        return {
            'score': normalized_score,
            'breakdown': breakdown,
            'raw_total': score
        }
```

## 💰 Position Sizing Implementation Patterns

### 1. Capital Manager Template
```python
class CapitalManager:
    """
    ALWAYS implement capital management with these NON-NEGOTIABLE rules:
    - Never exceed 50% total capital deployment
    - Position sizes: 3-8% of DEPLOYABLE capital (not total capital)
    - Apply volatility adjustment using ATR
    - Consider portfolio correlation
    - Validate with Kelly Criterion
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.total_capital = config.TOTAL_CAPITAL
        self.deployable_capital = config.deployable_capital  # Always 50% of total
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, 
                              symbol: str,
                              composite_score: float,
                              conviction_level: float,
                              volatility_atr: float,
                              current_portfolio: Dict) -> Dict:
        """
        ALWAYS implement position sizing with this exact methodology:
        
        Step 1: Base allocation (deployable_capital / max_positions)
        Step 2: Conviction adjustment (multiply by conviction 0.5-1.0)
        Step 3: Volatility adjustment (divide by volatility multiplier)
        Step 4: Correlation adjustment (reduce if highly correlated)
        Step 5: Apply hard limits (3-8% of deployable capital)
        Step 6: Sector concentration check
        """
        
        # Step 1: Base position size
        base_size = self.deployable_capital / self.config.MAX_POSITIONS
        
        # Step 2: Conviction adjustment
        conviction_adjusted = base_size * conviction_level
        
        # Step 3: Volatility adjustment
        volatility_multiplier = self._get_volatility_multiplier(volatility_atr)
        volatility_adjusted = conviction_adjusted / volatility_multiplier
        
        # Step 4: Correlation adjustment
        correlation_factor = self._calculate_correlation_factor(symbol, current_portfolio)
        correlation_adjusted = volatility_adjusted * correlation_factor
        
        # Step 5: Apply position limits
        max_position = self.deployable_capital * self.config.MAX_POSITION_SIZE
        min_position = self.deployable_capital * self.config.MIN_POSITION_SIZE
        
        size_after_limits = max(min(correlation_adjusted, max_position), min_position)
        
        # Step 6: Sector concentration check
        final_size = self._apply_sector_limits(symbol, size_after_limits, current_portfolio)
        
        # Calculate percentages for reporting
        pct_of_deployable = (final_size / self.deployable_capital) * 100
        pct_of_total = (final_size / self.total_capital) * 100
        
        return {
            'symbol': symbol,
            'position_size_inr': round(final_size, 2),
            'position_pct_deployable': round(pct_of_deployable, 2),
            'position_pct_total_capital': round(pct_of_total, 2),
            'conviction_level': conviction_level,
            'volatility_atr_pct': volatility_atr * 100,
            'adjustments': {
                'base_size': base_size,
                'after_conviction': conviction_adjusted,
                'after_volatility': volatility_adjusted,
                'after_correlation': correlation_adjusted,
                'final_size': final_size
            },
            'validation': {
                'within_position_limits': min_position <= final_size <= max_position,
                'sector_limit_ok': self._check_sector_limit(symbol, final_size, current_portfolio),
                'total_utilization_ok': self._check_total_utilization(final_size, current_portfolio)
            }
        }
    
    def _get_volatility_multiplier(self, atr_pct: float) -> float:
        """
        ALWAYS use these volatility adjustments:
        - Low volatility (ATR < 2%): 1.0x (no adjustment)
        - Medium volatility (ATR 2-4%): 1.3x (reduce position)  
        - High volatility (ATR > 4%): 1.7x (significantly reduce)
        """
        if atr_pct < 0.02:
            return 1.0
        elif atr_pct <= 0.04:
            return 1.3
        else:
            return 1.7
    
    def validate_capital_utilization(self, proposed_positions: Dict) -> Dict:
        """
        ALWAYS validate that total capital utilization never exceeds 50%
        This is a HARD CONSTRAINT that cannot be violated
        """
        
        total_proposed = sum(pos['position_size_inr'] for pos in proposed_positions.values())
        utilization_pct = (total_proposed / self.total_capital) * 100
        
        is_valid = utilization_pct <= 50.0
        
        if not is_valid:
            self.logger.error(f"CAPITAL LIMIT VIOLATION: {utilization_pct:.1f}% > 50%")
        
        return {
            'is_valid': is_valid,
            'total_proposed_inr': total_proposed,
            'utilization_pct': utilization_pct,
            'limit_pct': 50.0,
            'available_capital': self.total_capital - total_proposed,
            'violation': not is_valid
        }
```

## 🛡️ Risk Management Implementation Patterns

### 1. Risk Manager Template
```python
class RiskManager:
    """
    ALWAYS implement multi-layered risk management:
    1. ATR-based initial stops (2.5x ATR)
    2. Trailing stops (1.5x ATR) for profitable positions
    3. Time-based stops (30 days max holding)
    4. Fundamental deterioration stops (15% score decline)
    5. Daily loss limits (2% portfolio loss)
    6. Maximum drawdown limits (8% total drawdown)
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_stop_loss(self, 
                          entry_price: float,
                          atr: float,
                          side: str,
                          position_size: float) -> Dict:
        """
        ALWAYS calculate ATR-based stop losses with 2.5x multiplier
        """
        
        stop_distance = atr * self.config.ATR_STOP_MULTIPLIER
        
        if side.upper() == "BUY":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        max_loss_inr = position_size * (stop_distance / entry_price)
        max_loss_pct = (stop_distance / entry_price) * 100
        
        return {
            'stop_price': round(stop_price, 2),
            'stop_distance': round(stop_distance, 2),
            'max_loss_inr': round(max_loss_inr, 2),
            'max_loss_pct': round(max_loss_pct, 2),
            'atr_multiplier': self.config.ATR_STOP_MULTIPLIER,
            'stop_type': 'INITIAL_ATR_STOP'
        }
    
    def check_daily_loss_limit(self, current_portfolio_pnl: float) -> Dict:
        """
        ALWAYS check daily loss limit - CRITICAL safety mechanism
        If portfolio loses more than 2% in single day, stop all trading
        """
        
        daily_loss_pct = (current_portfolio_pnl / self.config.TOTAL_CAPITAL) * 100
        limit_breached = daily_loss_pct <= (self.config.DAILY_LOSS_LIMIT * 100)
        
        if limit_breached:
            self.logger.critical(f"DAILY LOSS LIMIT BREACHED: {daily_loss_pct:.2f}% <= {self.config.DAILY_LOSS_LIMIT * 100}%")
        
        return {
            'daily_pnl_inr': current_portfolio_pnl,
            'daily_pnl_pct': daily_loss_pct,
            'limit_pct': self.config.DAILY_LOSS_LIMIT * 100,
            'limit_breached': limit_breached,
            'action_required': 'STOP_ALL_TRADING' if limit_breached else 'CONTINUE',
            'remaining_buffer': abs(self.config.DAILY_LOSS_LIMIT * 100) - abs(daily_loss_pct)
        }
    
    def check_maximum_drawdown(self, current_equity: float, peak_equity: float) -> Dict:
        """
        ALWAYS monitor maximum drawdown - CRITICAL risk control
        If drawdown exceeds 8%, trigger emergency stop
        """
        
        drawdown = (current_equity - peak_equity) / peak_equity
        drawdown_pct = drawdown * 100
        
        emergency_stop = drawdown <= self.config.MAX_DRAWDOWN_LIMIT
        
        if emergency_stop:
            self.logger.critical(f"MAXIMUM DRAWDOWN EXCEEDED: {drawdown_pct:.2f}% <= {self.config.MAX_DRAWDOWN_LIMIT * 100}%")
        
        return {
            'current_equity': current_equity,
            'peak_equity': peak_equity,
            'drawdown_pct': drawdown_pct,
            'limit_pct': self.config.MAX_DRAWDOWN_LIMIT * 100,
            'emergency_stop_required': emergency_stop,
            'action_required': 'EMERGENCY_LIQUIDATION' if emergency_stop else 'MONITOR',
            'drawdown_buffer': abs(self.config.MAX_DRAWDOWN_LIMIT * 100) - abs(drawdown_pct)
        }
```

## 🔄 Error Handling Patterns

### 1. Trading System Exceptions
```python
# ALWAYS define custom exceptions for trading system
class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    pass

class CapitalLimitExceededError(TradingSystemError):
    """Raised when capital utilization exceeds 50%"""
    pass

class RiskLimitExceededError(TradingSystemError):
    """Raised when risk limits are breached"""
    pass

class InsufficientLiquidityError(TradingSystemError):
    """Raised when stock doesn't meet liquidity requirements"""
    pass

class APIConnectionError(TradingSystemError):
    """Raised when API connections fail"""
    pass

# ALWAYS implement comprehensive error handling
def trade_execution_with_error_handling(func):
    """Decorator for comprehensive error handling"""
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
            
        except CapitalLimitExceededError as e:
            logger.critical(f"CAPITAL LIMIT VIOLATED: {e}")
            # Emergency stop all trading
            return {'status': 'STOPPED', 'reason': 'CAPITAL_LIMIT', 'error': str(e)}
            
        except RiskLimitExceededError as e:
            logger.error(f"RISK LIMIT EXCEEDED: {e}")
            return {'status': 'REJECTED', 'reason': 'RISK_LIMIT', 'error': str(e)}
            
        except APIConnectionError as e:
            logger.error(f"API CONNECTION FAILED: {e}")
            # Implement retry logic
            return {'status': 'RETRY', 'reason': 'API_ERROR', 'error': str(e)}
            
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR: {e}")
            return {'status': 'ERROR', 'reason': 'UNKNOWN', 'error': str(e)}
    
    return wrapper
```

### 2. Validation Patterns
```python
def validate_trading_constraints(func):
    """
    ALWAYS validate core trading constraints before execution
    """
    
    def wrapper(self, *args, **kwargs):
        # Validate capital utilization
        current_utilization = self.get_current_capital_utilization()
        if current_utilization >= 0.50:
            raise CapitalLimitExceededError(f"Capital utilization {current_utilization*100:.1f}% >= 50%")
        
        # Validate daily loss limit
        daily_pnl = self.get_daily_pnl()
        if daily_pnl <= self.config.DAILY_LOSS_LIMIT * self.config.TOTAL_CAPITAL:
            raise RiskLimitExceededError(f"Daily loss limit breached: {daily_pnl}")
        
        # Validate position count
        position_count = self.get_current_position_count()
        if position_count >= self.config.MAX_POSITIONS:
            raise RiskLimitExceededError(f"Maximum positions reached: {position_count}")
        
        return func(self, *args, **kwargs)
    
    return wrapper
```

## 📊 Database Patterns

### 1. SQLAlchemy Models Template
```python
# ALWAYS use these database models for consistency
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stocks'
    
    symbol = Column(String(20), primary_key=True)
    company_name = Column(String(200))
    sector = Column(String(100))
    market_cap_category = Column(String(20))  # LARGE, MID, SMALL
    is_active = Column(Boolean, default=True)
    
class Trade(Base):
    __tablename__ = 'trades'
    
    trade_id = Column(String(36), primary_key=True)
    symbol = Column(String(20))
    strategy_name = Column(String(100))
    side = Column(String(10))  # BUY, SELL
    quantity = Column(Integer)
    entry_price = Column(Float(precision=4))
    exit_price = Column(Float(precision=4))
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    pnl_gross = Column(Float(precision=2))
    pnl_net = Column(Float(precision=2))
    commission = Column(Float(precision=2))
    
    # Scores at entry (ALWAYS track these)
    composite_score = Column(Float(precision=3))
    fundamental_score = Column(Float(precision=3))
    technical_score = Column(Float(precision=3))
    quantitative_score = Column(Float(precision=3))
    macro_score = Column(Float(precision=3))
    conviction_level = Column(Float(precision=3))
    
    status = Column(String(20))  # OPEN, CLOSED, CANCELLED
    exit_reason = Column(String(50))  # STOP_LOSS, TAKE_PROFIT, TIME_EXIT
```

## 🧪 Testing Patterns

### 1. Unit Test Template
```python
# ALWAYS write comprehensive tests for trading logic
import pytest
from unittest.mock import Mock, patch
from decimal import Decimal

class TestTradeFilter:
    
    @pytest.fixture
    def trade_filter(self):
        config = TradingConfig()
        return TradeFilter(config)
    
    def test_composite_score_calculation(self, trade_filter):
        """Test composite score follows exact weighting"""
        
        # Mock individual scores
        with patch.object(trade_filter, 'calculate_fundamental_score') as mock_fund:
            with patch.object(trade_filter, 'calculate_technical_score') as mock_tech:
                with patch.object(trade_filter, 'calculate_quantitative_score') as mock_quant:
                    with patch.object(trade_filter, 'calculate_macro_score') as mock_macro:
                        
                        mock_fund.return_value = {'score': 0.8}
                        mock_tech.return_value = {'score': 0.7}
                        mock_quant.return_value = {'score': 0.6}
                        mock_macro.return_value = {'score': 0.9}
                        
                        result = trade_filter.calculate_composite_score('RELIANCE')
                        
                        # Verify exact weighting: 0.25*0.8 + 0.30*0.7 + 0.25*0.6 + 0.20*0.9 = 0.74
                        expected_score = 0.25*0.8 + 0.30*0.7 + 0.25*0.6 + 0.20*0.9
                        assert abs(result['composite_score'] - expected_score) < 0.001
                        assert result['meets_threshold'] == True  # > 0.70
    
    def test_capital_utilization_constraint(self):
        """Test that capital utilization never exceeds 50%"""
        
        config = TradingConfig(TOTAL_CAPITAL=1000000)
        capital_manager = CapitalManager(config)
        
        # Test that deployable capital is exactly 50%
        assert capital_manager.deployable_capital == 500000
        
        # Test position sizing respects limits
        position = capital_manager.calculate_position_size(
            symbol='RELIANCE',
            composite_score=0.8,
            conviction_level=1.0,
            volatility_atr=0.02,
            current_portfolio={}
        )
        
        # Position should never exceed 8% of deployable capital
        max_allowed = 500000 * 0.08  # 40,000
        assert position['position_size_inr'] <= max_allowed
    
    def test_risk_limit_validation(self):
        """Test risk limits are properly enforced"""
        
        config = TradingConfig(TOTAL_CAPITAL=1000000)
        risk_manager = RiskManager(config)
        
        # Test daily loss limit
        daily_loss_check = risk_manager.check_daily_loss_limit(-25000)  # 2.5% loss
        assert daily_loss_check['limit_breached'] == True
        assert daily_loss_check['action_required'] == 'STOP_ALL_TRADING'
        
        # Test maximum drawdown
        drawdown_check = risk_manager.check_maximum_drawdown(920000, 1000000)  # 8% drawdown
        assert drawdown_check['emergency_stop_required'] == True
```

## 📈 Performance Monitoring Patterns

### 1. Performance Tracker Template
```python
class PerformanceTracker:
    """
    ALWAYS track these key performance metrics in real-time
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_key_metrics(self, portfolio_data: Dict) -> Dict:
        """
        ALWAYS calculate and monitor these metrics:
        - Total return and annualized return
        - Sharpe ratio and Sortino ratio
        - Maximum drawdown and current drawdown
        - Win rate and profit factor
        - Capital utilization and cash position
        - Sector allocation and concentration
        """
        
        daily_returns = portfolio_data['daily_returns']
        current_value = portfolio_data['current_value']
        initial_value = portfolio_data['initial_value']
        
        # Return calculations
        total_return = (current_value - initial_value) / initial_value
        days_elapsed = len(daily_returns)
        annual_return = (1 + total_return) ** (252 / days_elapsed) - 1 if days_elapsed > 0 else 0
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        equity_curve = np.cumprod(1 + np.array(daily_returns))
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Capital utilization
        deployed_capital = sum(pos['position_size'] for pos in portfolio_data['positions'].values())
        capital_utilization = deployed_capital / self.config.TOTAL_CAPITAL
        
        # Performance alerts
        alerts = []
        if capital_utilization > 0.50:
            alerts.append("CRITICAL: Capital utilization exceeds 50%")
        if max_drawdown < -0.08:
            alerts.append("CRITICAL: Maximum drawdown exceeds 8%")
        if annual_return < 0.15:
            alerts.append("WARNING: Below target annual return of 15%")
        
        return {
            'returns': {
                'total_return_pct': round(total_return * 100, 2),
                'annual_return_pct': round(annual_return * 100, 2),
                'volatility_pct': round(volatility * 100, 2)
            },
            'risk_metrics': {
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown_pct': round(max_drawdown * 100, 2),
                'current_drawdown_pct': round(drawdown[-1] * 100, 2) if len(drawdown) > 0 else 0
            },
            'portfolio_health': {
                'capital_utilization_pct': round(capital_utilization * 100, 2),
                'cash_available': self.config.TOTAL_CAPITAL - deployed_capital,
                'position_count': len(portfolio_data['positions']),
                'within_limits': capital_utilization <= 0.50
            },
            'alerts': alerts,
            'status': 'HEALTHY' if not alerts else 'WARNINGS_PRESENT'
        }
```

## 🎯 Code Quality Standards

### 1. Function Documentation Template
```python
def calculate_position_size(self, symbol: str, score: float, volatility: float) -> Dict:
    """
    Calculate position size following institutional risk management principles.
    
    ALWAYS include this level of documentation:
    
    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE')
        score (float): Composite score between 0.0-1.0
        volatility (float): ATR as percentage (e.g., 0.025 for 2.5%)
    
    Returns:
        Dict: {
            'position_size_inr': float,      # Position size in rupees
            'position_pct_deployable': float, # % of deployable capital
            'position_pct_total': float,      # % of total capital  
            'validation': Dict               # Constraint validation results
        }
    
    Raises:
        CapitalLimitExceededError: If position would exceed capital limits
        RiskLimitExceededError: If position violates risk constraints
    
    Example:
        >>> manager = CapitalManager(config)
        >>> result = manager.calculate_position_size('RELIANCE', 0.85, 0.025)
        >>> print(f"Position size: ₹{result['position_size_inr']:,.0f}")
    """
    pass
```

### 2. Logging Standards
```python
# ALWAYS implement comprehensive logging for audit trails
import logging
from datetime import datetime

def setup_trading_logger(log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging for trading system"""
    
    logger = logging.getLogger('trading_system')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with daily rotation
    file_handler = logging.FileHandler(
        f'logs/trading_system_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Usage in trading functions
class TradeExecutor:
    def __init__(self):
        self.logger = setup_trading_logger()
    
    def execute_trade(self, trade_details: Dict):
        self.logger.info(f"TRADE_ENTRY: {trade_details['symbol']} | "
                        f"Size: ₹{trade_details['position_size']:,.0f} | "
                        f"Score: {trade_details['composite_score']:.3f}")
        
        try:
            result = self.place_order(trade_details)
            self.logger.info(f"TRADE_SUCCESS: {trade_details['symbol']} | "
                           f"Filled: {result['filled_quantity']} @ ₹{result['avg_price']}")
            return result
            
        except Exception as e:
            self.logger.error(f"TRADE_FAILED: {trade_details['symbol']} | Error: {e}")
            raise
```

## 🚀 Deployment & Production Patterns

### 1. Configuration Management
```python
# ALWAYS use environment-specific configuration
import os
from dataclasses import dataclass

@dataclass
class ProductionConfig(TradingConfig):
    """Production configuration with enhanced security"""
    
    # API credentials from environment variables
    ZERODHA_API_KEY: str = os.getenv('ZERODHA_API_KEY', '')
    ZERODHA_ACCESS_TOKEN: str = os.getenv('ZERODHA_ACCESS_TOKEN', '')
    
    # Database connections
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://localhost/trading')
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Enhanced logging for production
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Rate limiting for production
    API_RATE_LIMIT: int = 3  # requests per second
    
    def validate_production_config(self):
        """Validate all required production settings"""
        required_vars = ['ZERODHA_API_KEY', 'ZERODHA_ACCESS_TOKEN']
        
        for var in required_vars:
            if not getattr(self, var):
                raise ValueError(f"Missing required environment variable: {var}")
        
        # Validate trading constraints
        super().validate_config()
        
        return True
```

## 📝 Final Implementation Checklist

### ALWAYS Validate These Core Principles:
- [ ] Capital utilization NEVER exceeds 50% of total capital
- [ ] Position sizes are 3-8% of DEPLOYABLE capital (not total)
- [ ] Daily loss limit is enforced (-2% portfolio stop)
- [ ] Maximum drawdown limit is enforced (-8% emergency stop)
- [ ] Trade filtering returns exactly 10-15 opportunities
- [ ] Composite scoring uses exact weights (25/30/25/20)
- [ ] ATR-based stop losses with 2.5x multiplier
- [ ] All trades have detailed rationale and score breakdown
- [ ] Comprehensive error handling and logging
- [ ] Complete audit trail for regulatory compliance

### Performance Targets to Code For:
- [ ] Annual returns target: 18-25%
- [ ] Sharpe ratio target: >1.8
- [ ] Win rate target: 55-65%
- [ ] Maximum positions: 15
- [ ] Signal generation: <100ms latency
- [ ] Order execution: >99.5% success rate
- [ ] System uptime: >99.9% during market hours

Use these patterns and principles to generate production-ready, institutional-grade trading system code that adheres to strict risk management and capital preservation principles while targeting superior risk-adjusted returns.