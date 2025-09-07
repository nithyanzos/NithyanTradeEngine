"""
Position Management System
Conservative capital allocation and position sizing with institutional risk controls
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from decimal import Decimal
import uuid

from config.trading_config import get_trading_config
from config.database_config import get_db_session, Portfolio, Trade
from strategy.trade_filter import TradeOpportunity

logger = logging.getLogger(__name__)

@dataclass
class PositionSizingResult:
    """Position sizing calculation result"""
    
    symbol: str
    position_size_inr: float
    position_pct_deployable: float
    position_pct_total_capital: float
    conviction_level: float
    volatility_atr_pct: float
    
    # Adjustment breakdown
    adjustments: Dict[str, float]
    
    # Validation results
    validation: Dict[str, bool]
    
    # Risk metrics
    max_loss_inr: float
    max_loss_pct: float
    
    # Rationale
    sizing_rationale: str
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PortfolioState:
    """Current portfolio state snapshot"""
    
    total_value: float
    cash_available: float
    invested_value: float
    capital_utilization: float
    
    position_count: int
    positions: Dict[str, Dict]
    
    sector_allocation: Dict[str, float]
    market_cap_allocation: Dict[str, float]
    
    daily_pnl: float
    total_return: float
    
    risk_metrics: Dict[str, float]
    
    timestamp: datetime = field(default_factory=datetime.now)


class CapitalManager:
    """
    Conservative capital management with NEVER-EXCEED 50% deployment rule
    
    Core Principles:
    - NEVER exceed 50% total capital deployment
    - Position sizes: 3-8% of DEPLOYABLE capital (not total capital)
    - Apply volatility adjustment using ATR
    - Consider portfolio correlation
    - Validate with Kelly Criterion (conservative 25% fractional)
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.total_capital = self.config.TOTAL_CAPITAL
        self.deployable_capital = self.config.deployable_capital  # Always 50% of total
        self.cash_reserve = self.total_capital * 0.50
        
        logger.info(f"💰 Capital Manager initialized:")
        logger.info(f"  Total Capital: ₹{self.total_capital:,.0f}")
        logger.info(f"  Deployable Capital: ₹{self.deployable_capital:,.0f}")
        logger.info(f"  Cash Reserve: ₹{self.cash_reserve:,.0f}")
        
    def calculate_position_size(self, 
                              opportunity: TradeOpportunity,
                              current_portfolio: PortfolioState) -> PositionSizingResult:
        """
        Advanced position sizing algorithm with multiple safety layers
        
        Steps:
        1. Base allocation (deployable_capital / max_positions)
        2. Conviction adjustment (multiply by conviction 0.5-1.0)
        3. Volatility adjustment (divide by volatility multiplier)
        4. Correlation adjustment (reduce if highly correlated)
        5. Apply hard limits (3-8% of deployable capital)
        6. Sector concentration check
        7. Kelly Criterion validation
        8. Market regime adjustment
        
        Args:
            opportunity: TradeOpportunity object
            current_portfolio: Current portfolio state
            
        Returns:
            PositionSizingResult with detailed breakdown
        """
        
        symbol = opportunity.symbol
        
        try:
            # Step 1: Base position calculation
            base_size = self.deployable_capital / self.config.MAX_POSITIONS
            
            # Step 2: Conviction adjustment
            conviction_adjusted = base_size * opportunity.conviction_level
            
            # Step 3: Volatility adjustment
            volatility_multiplier = self._get_volatility_multiplier(opportunity.atr_pct / 100)
            volatility_adjusted = conviction_adjusted / volatility_multiplier
            
            # Step 4: Correlation adjustment
            correlation_factor = self._calculate_correlation_factor(symbol, current_portfolio)
            correlation_adjusted = volatility_adjusted * correlation_factor
            
            # Step 5: Apply position limits
            max_position = self.deployable_capital * self.config.MAX_POSITION_SIZE
            min_position = self.deployable_capital * self.config.MIN_POSITION_SIZE
            
            size_after_limits = max(min(correlation_adjusted, max_position), min_position)
            
            # Step 6: Sector concentration check
            sector_adjusted = self._apply_sector_limits(opportunity, size_after_limits, current_portfolio)
            
            # Step 7: Kelly Criterion validation
            kelly_result = self._kelly_criterion_check(sector_adjusted, opportunity)
            kelly_adjusted = min(sector_adjusted, kelly_result['recommended_size'])
            
            # Step 8: Market regime adjustment
            final_size = self._apply_market_regime_adjustment(kelly_adjusted)
            
            # Calculate percentages and risk metrics
            pct_of_deployable = (final_size / self.deployable_capital) * 100
            pct_of_total = (final_size / self.total_capital) * 100
            
            # Calculate maximum loss
            max_loss_inr = final_size * (opportunity.atr_pct / 100) * self.config.ATR_STOP_MULTIPLIER
            max_loss_pct = (max_loss_inr / self.total_capital) * 100
            
            # Validation checks
            validation = self._validate_position_constraints(
                final_size, opportunity, current_portfolio
            )
            
            # Generate sizing rationale
            rationale = self._generate_sizing_rationale(
                opportunity, final_size, opportunity.conviction_level, opportunity.atr_pct / 100
            )
            
            return PositionSizingResult(
                symbol=symbol,
                position_size_inr=round(final_size, 2),
                position_pct_deployable=round(pct_of_deployable, 2),
                position_pct_total_capital=round(pct_of_total, 2),
                conviction_level=opportunity.conviction_level,
                volatility_atr_pct=opportunity.atr_pct,
                
                adjustments={
                    'base_size': base_size,
                    'conviction_adjusted': conviction_adjusted,
                    'volatility_adjusted': volatility_adjusted,
                    'correlation_adjusted': correlation_adjusted,
                    'limits_applied': size_after_limits,
                    'sector_adjusted': sector_adjusted,
                    'kelly_adjusted': kelly_adjusted,
                    'final_size': final_size
                },
                
                validation=validation,
                
                max_loss_inr=round(max_loss_inr, 2),
                max_loss_pct=round(max_loss_pct, 2),
                
                sizing_rationale=rationale
            )
            
        except Exception as e:
            logger.error(f"Position sizing failed for {symbol}: {e}")
            
            # Return minimum position size as fallback
            min_size = self.deployable_capital * self.config.MIN_POSITION_SIZE
            
            return PositionSizingResult(
                symbol=symbol,
                position_size_inr=min_size,
                position_pct_deployable=self.config.MIN_POSITION_SIZE * 100,
                position_pct_total_capital=(min_size / self.total_capital) * 100,
                conviction_level=0.5,
                volatility_atr_pct=3.0,
                
                adjustments={'error': str(e), 'fallback_size': min_size},
                validation={'error': True},
                
                max_loss_inr=min_size * 0.05,  # 5% stop loss estimate
                max_loss_pct=(min_size * 0.05 / self.total_capital) * 100,
                
                sizing_rationale=f"Error in sizing calculation, using minimum position size: {str(e)}"
            )
    
    def _get_volatility_multiplier(self, atr_pct: float) -> float:
        """
        ATR-based volatility position sizing adjustment
        
        Logic:
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
    
    def _calculate_correlation_factor(self, symbol: str, portfolio: PortfolioState) -> float:
        """
        Reduce position size for highly correlated stocks
        
        Correlation Adjustments:
        - Average correlation < 0.3: Factor = 1.0 (no adjustment)
        - Average correlation 0.3-0.6: Factor = 0.8 (reduce 20%)
        - Average correlation > 0.6: Factor = 0.6 (reduce 40%)
        
        Note: This is simplified - real implementation would calculate actual correlations
        """
        
        if portfolio.position_count == 0:
            return 1.0
        
        # Simplified correlation estimation based on sector overlap
        # In real implementation, this would use historical price correlations
        
        # Estimate average correlation (placeholder logic)
        avg_correlation = np.random.uniform(0.1, 0.7)  # Placeholder
        
        if avg_correlation < 0.3:
            return 1.0
        elif avg_correlation <= 0.6:
            return 0.8
        else:
            return 0.6
    
    def _apply_sector_limits(self, 
                           opportunity: TradeOpportunity, 
                           calculated_size: float, 
                           portfolio: PortfolioState) -> float:
        """
        Apply sector concentration limits
        
        Rules:
        - Maximum 30% allocation to any single sector
        - Consider existing sector exposure
        """
        
        sector = opportunity.sector
        current_sector_allocation = portfolio.sector_allocation.get(sector, 0.0)
        
        # Calculate maximum additional allocation for this sector
        max_sector_allocation_inr = self.deployable_capital * self.config.MAX_SECTOR_ALLOCATION
        available_sector_capacity = max_sector_allocation_inr - current_sector_allocation
        
        if available_sector_capacity <= 0:
            logger.warning(f"Sector limit exceeded for {sector}, returning zero position")
            return 0.0
        
        # Return minimum of calculated size and available sector capacity
        return min(calculated_size, available_sector_capacity)
    
    def _kelly_criterion_check(self, position_size: float, opportunity: TradeOpportunity) -> Dict[str, float]:
        """
        Kelly Criterion position size validation
        
        Uses historical win rate and average win/loss from backtesting
        Applies conservative 25% fractional Kelly for safety
        """
        
        # These would come from historical backtesting results
        # Placeholder values for similar scoring opportunities
        estimated_win_rate = 0.62  # 62% win rate
        avg_win_pct = 0.15  # 15% average win
        avg_loss_pct = 0.08  # 8% average loss (with stop losses)
        
        if avg_loss_pct == 0:
            return {
                'kelly_fraction': 0.05,
                'recommended_size': self.deployable_capital * 0.05,
                'recommendation': 'insufficient_data'
            }
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win_pct / avg_loss_pct
        p = estimated_win_rate
        q = 1 - estimated_win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly (25% of full Kelly for safety)
        safe_kelly = max(0.01, min(kelly_fraction * 0.25, 0.08))
        
        recommended_size = self.deployable_capital * safe_kelly
        
        return {
            'kelly_fraction': safe_kelly,
            'recommended_size': recommended_size,
            'kelly_ratio': position_size / recommended_size if recommended_size > 0 else 0,
            'recommendation': 'approved' if position_size <= recommended_size * 1.2 else 'reduce_size'
        }
    
    def _apply_market_regime_adjustment(self, base_size: float) -> float:
        """
        Adjust position sizes based on market regime
        
        Market Regime Adjustments:
        - BULL: Increase size by 15% (1.15x)
        - NORMAL: No adjustment (1.0x)
        - VOLATILE: Reduce size by 25% (0.75x)
        - BEAR: Reduce size by 40% (0.6x)
        """
        
        # This would be determined by market regime detection
        # Placeholder implementation
        market_regime = 'NORMAL'  # Would come from market analysis
        
        regime_multipliers = {
            'BULL': 1.15,
            'NORMAL': 1.0,
            'VOLATILE': 0.75,
            'BEAR': 0.6
        }
        
        multiplier = regime_multipliers.get(market_regime, 1.0)
        return base_size * multiplier
    
    def _validate_position_constraints(self, 
                                     position_size: float, 
                                     opportunity: TradeOpportunity,
                                     portfolio: PortfolioState) -> Dict[str, bool]:
        """
        Validate all position constraints
        
        Returns:
            Dict with validation results for each constraint
        """
        
        validation = {}
        
        # Position size limits
        max_allowed = self.deployable_capital * self.config.MAX_POSITION_SIZE
        min_allowed = self.deployable_capital * self.config.MIN_POSITION_SIZE
        validation['within_position_limits'] = min_allowed <= position_size <= max_allowed
        
        # Sector limits
        sector = opportunity.sector
        current_sector_allocation = portfolio.sector_allocation.get(sector, 0.0)
        max_sector_allocation = self.deployable_capital * self.config.MAX_SECTOR_ALLOCATION
        validation['sector_limit_ok'] = current_sector_allocation + position_size <= max_sector_allocation
        
        # Total capital utilization
        total_proposed_deployment = portfolio.invested_value + position_size
        validation['total_utilization_ok'] = total_proposed_deployment <= self.deployable_capital
        
        # Portfolio count limits
        new_position_count = portfolio.position_count + 1
        validation['position_count_ok'] = new_position_count <= self.config.MAX_POSITIONS
        
        return validation
    
    def _generate_sizing_rationale(self, 
                                 opportunity: TradeOpportunity,
                                 final_size: float,
                                 conviction: float,
                                 volatility: float) -> str:
        """
        Generate human-readable position sizing rationale
        """
        
        size_pct = (final_size / self.deployable_capital) * 100
        
        rationale = f"Position sized at ₹{final_size:,.0f} ({size_pct:.1f}% of deployable capital) for {opportunity.symbol}. "
        
        if conviction >= 0.8:
            rationale += "High conviction opportunity with strong composite score. "
        elif conviction >= 0.6:
            rationale += "Good conviction level based on multi-factor analysis. "
        else:
            rationale += "Moderate conviction, sized conservatively. "
        
        if volatility > 0.05:
            rationale += "Position reduced due to elevated volatility. "
        elif volatility < 0.02:
            rationale += "Low volatility allows for standard sizing. "
        
        rationale += f"ATR-based stop loss provides {self.config.ATR_STOP_MULTIPLIER}x protection."
        
        return rationale
    
    def validate_capital_utilization(self, portfolio: PortfolioState, new_positions: List[PositionSizingResult]) -> Dict:
        """
        CRITICAL: Validate that total capital utilization never exceeds 50%
        This is a HARD CONSTRAINT that cannot be violated
        """
        
        current_deployed = portfolio.invested_value
        additional_deployment = sum(pos.position_size_inr for pos in new_positions)
        total_proposed = current_deployed + additional_deployment
        
        utilization_pct = (total_proposed / self.total_capital) * 100
        
        is_valid = utilization_pct <= 50.0
        
        if not is_valid:
            logger.critical(f"🚨 CAPITAL LIMIT VIOLATION: {utilization_pct:.1f}% > 50%")
        
        return {
            'is_valid': is_valid,
            'current_deployed_inr': current_deployed,
            'additional_deployment_inr': additional_deployment,
            'total_proposed_inr': total_proposed,
            'utilization_pct': utilization_pct,
            'limit_pct': 50.0,
            'available_capital': self.total_capital - total_proposed,
            'violation': not is_valid,
            'deployable_capital': self.deployable_capital,
            'cash_reserve': self.cash_reserve
        }


class PortfolioTracker:
    """
    Real-time portfolio tracking and state management
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.db_session = get_db_session()
        
    def get_current_portfolio_state(self) -> PortfolioState:
        """
        Get comprehensive current portfolio state
        
        Returns:
            PortfolioState with all current metrics
        """
        
        try:
            # Get current positions from database
            positions = self.db_session.query(Portfolio).all()
            
            # Calculate portfolio metrics
            total_value = 0
            invested_value = 0
            positions_dict = {}
            sector_allocation = {}
            market_cap_allocation = {}
            
            for position in positions:
                symbol = position.symbol
                market_value = position.market_value or 0
                
                # Update totals
                total_value += market_value
                invested_value += market_value
                
                # Track positions
                positions_dict[symbol] = {
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'current_price': position.current_price,
                    'market_value': market_value,
                    'unrealized_pnl': position.unrealized_pnl or 0,
                    'entry_date': position.entry_date
                }
                
                # Get stock info for sector allocation
                stock_info = self._get_stock_info(symbol)
                if stock_info:
                    sector = stock_info.get('sector', 'UNKNOWN')
                    market_cap_cat = stock_info.get('market_cap_category', 'UNKNOWN')
                    
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + market_value
                    market_cap_allocation[market_cap_cat] = market_cap_allocation.get(market_cap_cat, 0) + market_value
            
            # Calculate cash available
            cash_available = self.config.TOTAL_CAPITAL - invested_value
            
            # Calculate capital utilization
            capital_utilization = invested_value / self.config.TOTAL_CAPITAL
            
            # Calculate daily P&L (simplified)
            daily_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions_dict.values())
            
            # Calculate total return
            total_return = (total_value - self.config.TOTAL_CAPITAL) / self.config.TOTAL_CAPITAL
            
            # Calculate risk metrics
            risk_metrics = self._calculate_portfolio_risk_metrics(positions_dict)
            
            return PortfolioState(
                total_value=total_value,
                cash_available=cash_available,
                invested_value=invested_value,
                capital_utilization=capital_utilization,
                
                position_count=len(positions),
                positions=positions_dict,
                
                sector_allocation=sector_allocation,
                market_cap_allocation=market_cap_allocation,
                
                daily_pnl=daily_pnl,
                total_return=total_return,
                
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            
            # Return empty portfolio state
            return PortfolioState(
                total_value=self.config.TOTAL_CAPITAL,
                cash_available=self.config.TOTAL_CAPITAL,
                invested_value=0.0,
                capital_utilization=0.0,
                
                position_count=0,
                positions={},
                
                sector_allocation={},
                market_cap_allocation={},
                
                daily_pnl=0.0,
                total_return=0.0,
                
                risk_metrics={}
            )
    
    def _get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get stock information from database"""
        
        try:
            from config.database_config import Stock
            
            stock = self.db_session.query(Stock).filter(Stock.symbol == symbol).first()
            
            if stock:
                return {
                    'sector': stock.sector,
                    'market_cap_category': stock.market_cap_category,
                    'company_name': stock.company_name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None
    
    def _calculate_portfolio_risk_metrics(self, positions: Dict) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        
        if not positions:
            return {}
        
        try:
            # Calculate portfolio beta (simplified)
            # In real implementation, this would use actual beta calculations
            portfolio_beta = 1.0  # Placeholder
            
            # Calculate position concentration
            total_value = sum(pos['market_value'] for pos in positions.values())
            max_position_pct = max(pos['market_value'] / total_value for pos in positions.values()) if total_value > 0 else 0
            
            # Calculate number of positions
            position_count = len(positions)
            
            return {
                'portfolio_beta': portfolio_beta,
                'max_position_concentration': max_position_pct,
                'position_count': position_count,
                'diversification_ratio': min(1.0, position_count / 15)  # Ideal is 15 positions
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {}
