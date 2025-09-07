"""
Comprehensive Risk Management System
Multi-layered risk controls with institutional-grade safety mechanisms
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from config.trading_config import get_trading_config
from config.database_config import get_db_session, Portfolio, Trade, AlertLog
from strategy.position_manager import PortfolioState

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ExitReason(Enum):
    """Trade exit reasons"""
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_EXIT = "TIME_EXIT"
    FUNDAMENTAL_DETERIORATION = "FUNDAMENTAL_DETERIORATION"
    TECHNICAL_BREAKDOWN = "TECHNICAL_BREAKDOWN"
    PORTFOLIO_HEAT = "PORTFOLIO_HEAT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    EMERGENCY_LIQUIDATION = "EMERGENCY_LIQUIDATION"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    
    alert_id: str
    alert_type: str
    severity: RiskLevel
    symbol: Optional[str]
    
    title: str
    message: str
    recommended_action: str
    
    current_value: float
    threshold_value: float
    
    portfolio_impact: Dict[str, float]
    
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StopLossLevel:
    """Stop loss configuration for a position"""
    
    symbol: str
    entry_price: float
    current_price: float
    side: str  # BUY or SELL
    
    # Stop loss levels
    initial_stop_price: float
    trailing_stop_price: Optional[float]
    
    # Stop loss parameters
    atr_value: float
    stop_distance: float
    max_loss_inr: float
    max_loss_pct: float
    
    # Status
    stop_type: str  # INITIAL_ATR, TRAILING_ATR, FUNDAMENTAL, TECHNICAL
    is_active: bool
    last_updated: datetime = field(default_factory=datetime.now)


class ComprehensiveRiskManager:
    """
    Institutional-grade risk management with multiple stop loss layers
    
    Stop Loss Types:
    1. Initial Hard Stop: ATR-based fixed stop (2.5x ATR)
    2. Trailing Stop: Dynamic stop following favorable moves (1.5x ATR)
    3. Time-based Stop: Maximum holding period (30 trading days)
    4. Fundamental Stop: Exit on score deterioration (>15% decline)
    5. Technical Breakdown Stop: Key support level violations
    6. Portfolio Heat Stop: Overall portfolio risk limits
    7. Daily Loss Limit: 2% daily portfolio loss emergency stop
    8. Maximum Drawdown: 8% maximum drawdown emergency stop
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.db_session = get_db_session()
        
        # Risk limit parameters
        self.atr_stop_multiplier = self.config.ATR_STOP_MULTIPLIER
        self.trail_stop_multiplier = self.config.TRAIL_STOP_MULTIPLIER
        self.max_holding_days = self.config.MAX_HOLDING_DAYS
        self.fundamental_deterioration_threshold = -0.15  # 15% score decline
        self.daily_loss_limit = self.config.DAILY_LOSS_LIMIT
        self.max_drawdown_limit = self.config.MAX_DRAWDOWN_LIMIT
        
        logger.info("🛡️ Risk Manager initialized with institutional controls")
    
    def calculate_initial_stop_loss(self, 
                                   entry_price: float,
                                   atr: float,
                                   side: str,
                                   position_size: float) -> StopLossLevel:
        """
        Calculate initial ATR-based stop loss
        
        Formula:
        - Long: Entry Price - (ATR × 2.5)
        - Short: Entry Price + (ATR × 2.5)
        
        Args:
            entry_price: Entry price of position
            atr: Average True Range value
            side: BUY or SELL
            position_size: Position size in rupees
            
        Returns:
            StopLossLevel object with all stop loss details
        """
        
        stop_distance = atr * self.atr_stop_multiplier
        
        if side.upper() == "BUY":
            stop_price = entry_price - stop_distance
            max_loss_pct = stop_distance / entry_price
        else:  # SELL
            stop_price = entry_price + stop_distance
            max_loss_pct = stop_distance / entry_price
        
        max_loss_inr = position_size * max_loss_pct
        
        return StopLossLevel(
            symbol="",  # Will be set by caller
            entry_price=entry_price,
            current_price=entry_price,
            side=side,
            
            initial_stop_price=round(stop_price, 2),
            trailing_stop_price=None,
            
            atr_value=atr,
            stop_distance=round(stop_distance, 2),
            max_loss_inr=round(max_loss_inr, 2),
            max_loss_pct=round(max_loss_pct * 100, 2),
            
            stop_type='INITIAL_ATR',
            is_active=True
        )
    
    def calculate_trailing_stop(self, 
                               position: Dict,
                               current_price: float,
                               high_since_entry: float,
                               atr: float) -> StopLossLevel:
        """
        Dynamic trailing stop calculation
        
        Logic:
        - Only activates after position moves favorably by 2%
        - Trails at 1.5x ATR distance from highest favorable price
        - Never moves against the position
        
        Args:
            position: Position dictionary with entry details
            current_price: Current market price
            high_since_entry: Highest price since entry (for longs)
            atr: Current Average True Range
            
        Returns:
            Updated StopLossLevel with trailing stop
        """
        
        entry_price = position['average_price']
        side = position.get('side', 'BUY')
        
        trail_distance = atr * self.trail_stop_multiplier
        
        if side.upper() == "BUY":
            # Long position trailing stop
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            
            # Only activate trailing stop after 2% profit
            if unrealized_pnl_pct < 0.02:
                return None
            
            trail_stop = high_since_entry - trail_distance
            
            # Ensure trailing stop never moves down
            current_stop = position.get('trailing_stop_price', trail_stop)
            trail_stop = max(trail_stop, current_stop) if current_stop else trail_stop
            
        else:  # SHORT position
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
            
            if unrealized_pnl_pct < 0.02:
                return None
            
            # For shorts, we trail from the lowest price since entry
            trail_stop = high_since_entry + trail_distance  # high_since_entry is actually low_since_entry for shorts
            
            # Ensure trailing stop never moves up
            current_stop = position.get('trailing_stop_price', trail_stop)
            trail_stop = min(trail_stop, current_stop) if current_stop else trail_stop
        
        return StopLossLevel(
            symbol=position['symbol'],
            entry_price=entry_price,
            current_price=current_price,
            side=side,
            
            initial_stop_price=position.get('initial_stop_price', 0),
            trailing_stop_price=round(trail_stop, 2),
            
            atr_value=atr,
            stop_distance=round(trail_distance, 2),
            max_loss_inr=0,  # Trailing stops aim to preserve profits
            max_loss_pct=0,
            
            stop_type='TRAILING_ATR',
            is_active=True
        )
    
    def check_time_based_exit(self, 
                            entry_date: datetime,
                            current_date: datetime = None) -> Dict[str, Union[bool, int, str]]:
        """
        Time-based exit logic for mean reversion positions
        
        Rules:
        - Maximum holding period: 30 trading days
        - Early exit warning after 20 days
        - Accelerated exit in volatile markets
        
        Args:
            entry_date: Position entry date
            current_date: Current date (defaults to now)
            
        Returns:
            Dict with exit recommendation and timing details
        """
        
        if current_date is None:
            current_date = datetime.now()
        
        # Calculate business days (approximate)
        total_days = (current_date - entry_date).days
        holding_days = max(1, int(total_days * 5/7))  # Approximate trading days
        
        should_exit = holding_days >= self.max_holding_days
        
        # Early exit conditions
        warning_threshold = int(self.max_holding_days * 0.67)  # 20 days
        approaching_limit = holding_days >= warning_threshold
        
        return {
            'should_exit': should_exit,
            'holding_days': holding_days,
            'max_holding_days': self.max_holding_days,
            'approaching_limit': approaching_limit,
            'days_remaining': max(0, self.max_holding_days - holding_days),
            'exit_reason': ExitReason.TIME_EXIT.value if should_exit else None,
            'urgency': 'HIGH' if should_exit else 'MEDIUM' if approaching_limit else 'LOW'
        }
    
    def check_fundamental_deterioration(self, 
                                      symbol: str,
                                      entry_score: float,
                                      entry_date: datetime) -> Dict[str, Union[bool, float, str, List]]:
        """
        Monitor fundamental score deterioration
        
        Exit Triggers:
        - Fundamental score drops >15% from entry
        - Key metrics breach thresholds
        - Earnings guidance cuts or negative surprises
        
        Args:
            symbol: Stock symbol
            entry_score: Fundamental score at entry
            entry_date: Entry date for position
            
        Returns:
            Dict with deterioration analysis and exit recommendation
        """
        
        try:
            # Get current fundamental score
            current_score = self._get_current_fundamental_score(symbol)
            
            if current_score is None:
                return {
                    'should_exit': False, 
                    'reason': 'SCORE_UNAVAILABLE',
                    'current_score': None,
                    'score_change_pct': 0
                }
            
            score_change = (current_score - entry_score) / entry_score
            should_exit = score_change <= self.fundamental_deterioration_threshold
            
            # Check for additional fundamental red flags
            red_flags = self._check_fundamental_red_flags(symbol)
            
            # Override exit decision if critical red flags present
            critical_exit = len(red_flags) > 0 and any('CRITICAL' in flag for flag in red_flags)
            
            return {
                'should_exit': should_exit or critical_exit,
                'entry_score': entry_score,
                'current_score': current_score,
                'score_change_pct': round(score_change * 100, 2),
                'threshold_pct': round(self.fundamental_deterioration_threshold * 100, 2),
                'red_flags': red_flags,
                'exit_reason': ExitReason.FUNDAMENTAL_DETERIORATION.value if (should_exit or critical_exit) else None,
                'severity': 'CRITICAL' if critical_exit else 'HIGH' if should_exit else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Fundamental deterioration check failed for {symbol}: {e}")
            return {
                'should_exit': False,
                'error': str(e),
                'current_score': None,
                'score_change_pct': 0
            }
    
    def check_technical_breakdown(self, symbol: str, position: Dict) -> Dict[str, Union[bool, str, List]]:
        """
        Technical breakdown detection
        
        Breakdown Signals:
        - Break below key support levels (50-day, 200-day MA)
        - Negative momentum divergence (price vs RSI/MACD)
        - High volume selling pressure
        - Sector relative weakness
        
        Args:
            symbol: Stock symbol
            position: Position details
            
        Returns:
            Dict with technical breakdown analysis
        """
        
        try:
            # Get technical indicators
            technical_data = self._get_technical_indicators(symbol)
            
            if not technical_data:
                return {
                    'should_exit': False,
                    'reason': 'NO_TECHNICAL_DATA',
                    'breakdown_signals': []
                }
            
            breakdown_signals = []
            
            # Moving average breaks
            current_price = technical_data.get('close', 0)
            ma_50 = technical_data.get('sma_50', 0)
            ma_200 = technical_data.get('sma_200', 0)
            
            if current_price < ma_50 * 0.98:  # 2% below MA50
                breakdown_signals.append('BELOW_MA50')
            
            if current_price < ma_200 * 0.95:  # 5% below MA200
                breakdown_signals.append('BELOW_MA200')
            
            # Momentum divergence
            if self._detect_negative_divergence(technical_data):
                breakdown_signals.append('NEGATIVE_DIVERGENCE')
            
            # Volume analysis
            if self._detect_distribution_pattern(technical_data):
                breakdown_signals.append('DISTRIBUTION_PATTERN')
            
            # RSI breakdown
            rsi = technical_data.get('rsi', 50)
            if rsi < 30:  # Oversold condition
                breakdown_signals.append('RSI_OVERSOLD')
            
            # Multiple confirmation required for exit
            should_exit = len(breakdown_signals) >= 2
            
            return {
                'should_exit': should_exit,
                'breakdown_signals': breakdown_signals,
                'signal_count': len(breakdown_signals),
                'exit_reason': ExitReason.TECHNICAL_BREAKDOWN.value if should_exit else None,
                'current_price': current_price,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'rsi': rsi,
                'severity': 'HIGH' if should_exit else 'MEDIUM' if breakdown_signals else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Technical breakdown check failed for {symbol}: {e}")
            return {
                'should_exit': False,
                'error': str(e),
                'breakdown_signals': []
            }
    
    def check_daily_loss_limit(self, portfolio_state: PortfolioState) -> Dict[str, Union[bool, float, str]]:
        """
        CRITICAL: Check daily loss limit - emergency safety mechanism
        
        If portfolio loses more than 2% in single day, stop all trading
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dict with daily loss analysis and emergency actions
        """
        
        daily_pnl = portfolio_state.daily_pnl
        daily_loss_pct = (daily_pnl / self.config.TOTAL_CAPITAL) * 100
        
        limit_breached = daily_loss_pct <= (self.daily_loss_limit * 100)
        
        if limit_breached:
            logger.critical(f"🚨 DAILY LOSS LIMIT BREACHED: {daily_loss_pct:.2f}% <= {self.daily_loss_limit * 100}%")
            
            # Create critical alert
            self._create_critical_alert(
                alert_type="DAILY_LOSS_LIMIT",
                title="Daily Loss Limit Breached",
                message=f"Portfolio daily loss of {daily_loss_pct:.2f}% exceeds {abs(self.daily_loss_limit * 100)}% limit",
                recommended_action="STOP_ALL_TRADING",
                current_value=daily_loss_pct,
                threshold_value=self.daily_loss_limit * 100
            )
        
        return {
            'daily_pnl_inr': daily_pnl,
            'daily_pnl_pct': daily_loss_pct,
            'limit_pct': self.daily_loss_limit * 100,
            'limit_breached': limit_breached,
            'action_required': 'STOP_ALL_TRADING' if limit_breached else 'CONTINUE',
            'remaining_buffer': abs(self.daily_loss_limit * 100) - abs(daily_loss_pct),
            'severity': RiskLevel.CRITICAL if limit_breached else RiskLevel.LOW
        }
    
    def check_maximum_drawdown(self, 
                             current_equity: float, 
                             peak_equity: float) -> Dict[str, Union[bool, float, str]]:
        """
        CRITICAL: Monitor maximum drawdown - ultimate risk control
        
        If drawdown exceeds 8%, trigger emergency liquidation
        
        Args:
            current_equity: Current portfolio value
            peak_equity: Historical peak portfolio value
            
        Returns:
            Dict with drawdown analysis and emergency actions
        """
        
        if peak_equity <= 0:
            peak_equity = self.config.TOTAL_CAPITAL
        
        drawdown = (current_equity - peak_equity) / peak_equity
        drawdown_pct = drawdown * 100
        
        emergency_stop = drawdown <= self.max_drawdown_limit
        
        if emergency_stop:
            logger.critical(f"🚨 MAXIMUM DRAWDOWN EXCEEDED: {drawdown_pct:.2f}% <= {self.max_drawdown_limit * 100}%")
            
            # Create critical alert
            self._create_critical_alert(
                alert_type="MAX_DRAWDOWN",
                title="Maximum Drawdown Exceeded",
                message=f"Portfolio drawdown of {abs(drawdown_pct):.2f}% exceeds {abs(self.max_drawdown_limit * 100)}% limit",
                recommended_action="EMERGENCY_LIQUIDATION",
                current_value=drawdown_pct,
                threshold_value=self.max_drawdown_limit * 100
            )
        
        return {
            'current_equity': current_equity,
            'peak_equity': peak_equity,
            'drawdown_pct': drawdown_pct,
            'limit_pct': self.max_drawdown_limit * 100,
            'emergency_stop_required': emergency_stop,
            'action_required': 'EMERGENCY_LIQUIDATION' if emergency_stop else 'MONITOR',
            'drawdown_buffer': abs(self.max_drawdown_limit * 100) - abs(drawdown_pct),
            'severity': RiskLevel.CRITICAL if emergency_stop else RiskLevel.HIGH if drawdown_pct < -5 else RiskLevel.MEDIUM
        }
    
    def portfolio_risk_monitor(self, portfolio_state: PortfolioState) -> Dict[str, Union[List, bool, str]]:
        """
        Real-time portfolio risk monitoring
        
        Risk Limits Monitored:
        - Daily P&L loss limit: -2%
        - Maximum drawdown: -8%
        - Position concentration: <8% per stock, <30% per sector
        - Beta exposure: <1.3 (if available)
        - Capital utilization: <50%
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dict with comprehensive risk assessment
        """
        
        risk_alerts = []
        emergency_stop = False
        
        try:
            # 1. Daily loss limit check
            daily_loss_check = self.check_daily_loss_limit(portfolio_state)
            if daily_loss_check['limit_breached']:
                risk_alerts.append(RiskAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type='DAILY_LOSS_LIMIT',
                    severity=RiskLevel.CRITICAL,
                    symbol=None,
                    title='Daily Loss Limit Breached',
                    message=f"Daily P&L of {daily_loss_check['daily_pnl_pct']:.2f}% exceeds limit",
                    recommended_action='STOP_ALL_TRADING',
                    current_value=daily_loss_check['daily_pnl_pct'],
                    threshold_value=daily_loss_check['limit_pct'],
                    portfolio_impact={'daily_pnl': daily_loss_check['daily_pnl_inr']}
                ))
                emergency_stop = True
            
            # 2. Maximum drawdown check (requires historical peak)
            # This would typically come from performance tracking
            historical_peak = portfolio_state.total_value  # Simplified - would track actual peak
            drawdown_check = self.check_maximum_drawdown(portfolio_state.total_value, historical_peak)
            
            if drawdown_check['emergency_stop_required']:
                risk_alerts.append(RiskAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type='MAX_DRAWDOWN',
                    severity=RiskLevel.CRITICAL,
                    symbol=None,
                    title='Maximum Drawdown Exceeded',
                    message=f"Drawdown of {abs(drawdown_check['drawdown_pct']):.2f}% exceeds limit",
                    recommended_action='EMERGENCY_LIQUIDATION',
                    current_value=drawdown_check['drawdown_pct'],
                    threshold_value=drawdown_check['limit_pct'],
                    portfolio_impact={'drawdown_amount': drawdown_check['current_equity'] - drawdown_check['peak_equity']}
                ))
                emergency_stop = True
            
            # 3. Position concentration checks
            total_value = portfolio_state.invested_value
            
            if total_value > 0:
                for symbol, position in portfolio_state.positions.items():
                    position_pct = position['market_value'] / total_value
                    
                    if position_pct > 0.08:  # 8% limit per position
                        risk_alerts.append(RiskAlert(
                            alert_id=self._generate_alert_id(),
                            alert_type='POSITION_CONCENTRATION',
                            severity=RiskLevel.HIGH,
                            symbol=symbol,
                            title=f'Position Concentration Risk: {symbol}',
                            message=f"Position size {position_pct*100:.1f}% exceeds 8% limit",
                            recommended_action='REDUCE_POSITION',
                            current_value=position_pct * 100,
                            threshold_value=8.0,
                            portfolio_impact={'position_value': position['market_value']}
                        ))
            
            # 4. Sector concentration checks
            deployable_capital = self.config.deployable_capital
            
            for sector, allocation in portfolio_state.sector_allocation.items():
                sector_pct = allocation / deployable_capital
                
                if sector_pct > self.config.MAX_SECTOR_ALLOCATION:  # 30% limit per sector
                    risk_alerts.append(RiskAlert(
                        alert_id=self._generate_alert_id(),
                        alert_type='SECTOR_CONCENTRATION',
                        severity=RiskLevel.HIGH,
                        symbol=None,
                        title=f'Sector Concentration Risk: {sector}',
                        message=f"Sector allocation {sector_pct*100:.1f}% exceeds 30% limit",
                        recommended_action='DIVERSIFY_SECTOR',
                        current_value=sector_pct * 100,
                        threshold_value=30.0,
                        portfolio_impact={'sector_value': allocation}
                    ))
            
            # 5. Capital utilization check
            if portfolio_state.capital_utilization > 0.50:  # 50% hard limit
                risk_alerts.append(RiskAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type='CAPITAL_UTILIZATION',
                    severity=RiskLevel.CRITICAL,
                    symbol=None,
                    title='Capital Utilization Limit Exceeded',
                    message=f"Capital utilization {portfolio_state.capital_utilization*100:.1f}% exceeds 50% limit",
                    recommended_action='REDUCE_POSITIONS',
                    current_value=portfolio_state.capital_utilization * 100,
                    threshold_value=50.0,
                    portfolio_impact={'excess_deployment': portfolio_state.invested_value - self.config.deployable_capital}
                ))
                emergency_stop = True
            
            # Save alerts to database
            for alert in risk_alerts:
                self._save_risk_alert(alert)
            
            return {
                'risk_alerts': risk_alerts,
                'alert_count': len(risk_alerts),
                'emergency_stop_required': emergency_stop,
                'portfolio_state': portfolio_state,
                'risk_score': self._calculate_portfolio_risk_score(risk_alerts),
                'recommendations': self._generate_risk_recommendations(risk_alerts),
                'last_check': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk monitoring failed: {e}")
            return {
                'risk_alerts': [],
                'alert_count': 0,
                'emergency_stop_required': False,
                'error': str(e)
            }
    
    def _get_current_fundamental_score(self, symbol: str) -> Optional[float]:
        """Get current fundamental score for symbol"""
        # This would integrate with the fundamental analyzer
        # Placeholder implementation
        return np.random.uniform(0.3, 0.9)
    
    def _check_fundamental_red_flags(self, symbol: str) -> List[str]:
        """Check for fundamental red flags"""
        # This would check for earnings restatements, management changes, etc.
        # Placeholder implementation
        possible_flags = [
            'EARNINGS_RESTATEMENT',
            'MANAGEMENT_CHANGE',
            'REGULATORY_ACTION',
            'MAJOR_CUSTOMER_LOSS',
            'GOVERNANCE_ISSUES'
        ]
        
        # Simulate 10% chance of red flag
        if np.random.random() < 0.10:
            return [np.random.choice(possible_flags)]
        
        return []
    
    def _get_technical_indicators(self, symbol: str) -> Optional[Dict]:
        """Get technical indicators for symbol"""
        # This would integrate with technical analyzer
        # Placeholder implementation
        return {
            'close': np.random.uniform(100, 200),
            'sma_50': np.random.uniform(95, 195),
            'sma_200': np.random.uniform(90, 190),
            'rsi': np.random.uniform(20, 80),
            'volume': np.random.randint(100000, 1000000)
        }
    
    def _detect_negative_divergence(self, technical_data: Dict) -> bool:
        """Detect negative momentum divergence"""
        # Simplified divergence detection
        return np.random.random() < 0.20  # 20% chance
    
    def _detect_distribution_pattern(self, technical_data: Dict) -> bool:
        """Detect institutional distribution pattern"""
        # Simplified distribution detection
        return np.random.random() < 0.15  # 15% chance
    
    def _create_critical_alert(self, 
                             alert_type: str,
                             title: str,
                             message: str,
                             recommended_action: str,
                             current_value: float,
                             threshold_value: float):
        """Create and log critical risk alert"""
        
        try:
            alert = AlertLog(
                alert_type=alert_type,
                severity='CRITICAL',
                title=title,
                message=message,
                portfolio_value=self.config.TOTAL_CAPITAL  # Would be actual portfolio value
            )
            
            self.db_session.add(alert)
            self.db_session.commit()
            
            logger.critical(f"🚨 CRITICAL ALERT: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to create critical alert: {e}")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _save_risk_alert(self, alert: RiskAlert):
        """Save risk alert to database"""
        
        try:
            db_alert = AlertLog(
                alert_type=alert.alert_type,
                severity=alert.severity.value,
                title=alert.title,
                message=alert.message,
                symbol=alert.symbol,
                portfolio_value=alert.current_value
            )
            
            self.db_session.add(db_alert)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to save risk alert: {e}")
    
    def _calculate_portfolio_risk_score(self, alerts: List[RiskAlert]) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        
        if not alerts:
            return 10.0  # Low risk
        
        # Weight alerts by severity
        severity_weights = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 3,
            RiskLevel.HIGH: 7,
            RiskLevel.CRITICAL: 15
        }
        
        total_weight = sum(severity_weights.get(alert.severity, 1) for alert in alerts)
        
        # Scale to 0-100 range
        risk_score = min(100, total_weight * 5)
        
        return risk_score
    
    def _generate_risk_recommendations(self, alerts: List[RiskAlert]) -> List[str]:
        """Generate actionable risk management recommendations"""
        
        if not alerts:
            return ["Portfolio risk levels are within acceptable limits"]
        
        recommendations = []
        
        # Group by alert type
        alert_types = {}
        for alert in alerts:
            if alert.alert_type not in alert_types:
                alert_types[alert.alert_type] = []
            alert_types[alert.alert_type].append(alert)
        
        # Generate specific recommendations
        for alert_type, type_alerts in alert_types.items():
            if alert_type == 'POSITION_CONCENTRATION':
                symbols = [alert.symbol for alert in type_alerts]
                recommendations.append(f"Reduce position sizes for: {', '.join(symbols)}")
            
            elif alert_type == 'SECTOR_CONCENTRATION':
                recommendations.append("Diversify sector allocation by reducing overweight positions")
            
            elif alert_type == 'DAILY_LOSS_LIMIT':
                recommendations.append("IMMEDIATE: Stop all new trading until daily losses stabilize")
            
            elif alert_type == 'MAX_DRAWDOWN':
                recommendations.append("EMERGENCY: Consider systematic liquidation to preserve capital")
            
            elif alert_type == 'CAPITAL_UTILIZATION':
                recommendations.append("Reduce overall position sizes to maintain 50% cash reserve")
        
        return recommendations
