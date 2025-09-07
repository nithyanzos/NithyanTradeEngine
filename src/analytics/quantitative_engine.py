"""
Quantitative Analysis Engine
Factor-based quantitative scoring and macro sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import asyncio
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config.trading_config import get_trading_config
from config.api_config import get_api_manager
from config.database_config import get_db_session, Stock, FundamentalData, MarketData
from analytics.fundamental_analyzer import ScoringResult

logger = logging.getLogger(__name__)

class QuantitativeEngine:
    """
    Factor-based quantitative scoring (25% weight)
    
    Scoring Components (100 points total):
    1. Momentum Factors (40 points)
    2. Quality Factors (35 points)
    3. Value Factors (25 points)
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.db_session = get_db_session()
        self.scaler = StandardScaler()
    
    async def calculate_quantitative_score(self, symbol: str) -> ScoringResult:
        """
        Calculate comprehensive quantitative factor score
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ScoringResult with quantitative factor analysis
        """
        
        try:
            # Get price and fundamental data
            price_data = await self._get_price_data(symbol, days=365)
            fundamental_data = self._get_fundamental_data(symbol)
            
            if price_data is None or len(price_data) < 100:
                logger.warning(f"Insufficient data for quantitative analysis: {symbol}")
                return ScoringResult(
                    score=0.0,
                    breakdown={'error': 'Insufficient data'},
                    raw_total=0.0
                )
            
            score_breakdown = {}
            total_score = 0
            
            # 1. Momentum Factors (40 points)
            momentum_score = await self._score_momentum_factors(symbol, price_data, score_breakdown)
            total_score += momentum_score
            
            # 2. Quality Factors (35 points)
            quality_score = self._score_quality_factors(fundamental_data, score_breakdown)
            total_score += quality_score
            
            # 3. Value Factors (25 points)
            value_score = self._score_value_factors(symbol, fundamental_data, price_data, score_breakdown)
            total_score += value_score
            
            # Normalize to 0-1 scale
            normalized_score = total_score / 100.0
            
            logger.debug(f"Quantitative score for {symbol}: {normalized_score:.3f} ({total_score}/100)")
            
            return ScoringResult(
                score=normalized_score,
                breakdown=score_breakdown,
                raw_total=total_score
            )
            
        except Exception as e:
            logger.error(f"Quantitative scoring failed for {symbol}: {e}")
            return ScoringResult(
                score=0.0,
                breakdown={'error': str(e)},
                raw_total=0.0
            )
    
    async def _score_momentum_factors(self, symbol: str, price_data: pd.DataFrame, breakdown: Dict) -> float:
        """
        Score momentum factors (40 points total)
        
        Components:
        - 1-month return (10 points)
        - 3-month return (10 points)
        - 6-month return (10 points)
        - 12-month return (10 points)
        """
        
        score = 0
        
        try:
            # Calculate returns for different periods
            returns = self._calculate_period_returns(price_data)
            
            # Get universe percentiles for comparison
            universe_returns = await self._get_universe_returns()
            
            # 1-month return scoring (10 points)
            return_1m = returns.get('1m', 0)
            percentile_1m = self._calculate_percentile(return_1m, universe_returns.get('1m', []))
            
            if percentile_1m >= 90:  # Top 10%
                score_1m = 10
            elif percentile_1m >= 75:  # Top 25%
                score_1m = 8
            elif percentile_1m >= 60:  # Top 40%
                score_1m = 6
            elif percentile_1m >= 50:  # Above median
                score_1m = 4
            elif percentile_1m >= 25:  # Above bottom 25%
                score_1m = 2
            else:
                score_1m = 0
            
            breakdown['return_1m_score'] = score_1m
            breakdown['return_1m_value'] = return_1m
            breakdown['return_1m_percentile'] = percentile_1m
            score += score_1m
            
            # 3-month return scoring (10 points)
            return_3m = returns.get('3m', 0)
            percentile_3m = self._calculate_percentile(return_3m, universe_returns.get('3m', []))
            
            if percentile_3m >= 90:
                score_3m = 10
            elif percentile_3m >= 75:
                score_3m = 8
            elif percentile_3m >= 60:
                score_3m = 6
            elif percentile_3m >= 50:
                score_3m = 4
            elif percentile_3m >= 25:
                score_3m = 2
            else:
                score_3m = 0
            
            breakdown['return_3m_score'] = score_3m
            breakdown['return_3m_value'] = return_3m
            breakdown['return_3m_percentile'] = percentile_3m
            score += score_3m
            
            # 6-month return scoring (10 points)
            return_6m = returns.get('6m', 0)
            percentile_6m = self._calculate_percentile(return_6m, universe_returns.get('6m', []))
            
            if percentile_6m >= 90:
                score_6m = 10
            elif percentile_6m >= 75:
                score_6m = 8
            elif percentile_6m >= 60:
                score_6m = 6
            elif percentile_6m >= 50:
                score_6m = 4
            elif percentile_6m >= 25:
                score_6m = 2
            else:
                score_6m = 0
            
            breakdown['return_6m_score'] = score_6m
            breakdown['return_6m_value'] = return_6m
            breakdown['return_6m_percentile'] = percentile_6m
            score += score_6m
            
            # 12-month return scoring (10 points)
            return_12m = returns.get('12m', 0)
            percentile_12m = self._calculate_percentile(return_12m, universe_returns.get('12m', []))
            
            if percentile_12m >= 90:
                score_12m = 10
            elif percentile_12m >= 75:
                score_12m = 8
            elif percentile_12m >= 60:
                score_12m = 6
            elif percentile_12m >= 50:
                score_12m = 4
            elif percentile_12m >= 25:
                score_12m = 2
            else:
                score_12m = 0
            
            breakdown['return_12m_score'] = score_12m
            breakdown['return_12m_value'] = return_12m
            breakdown['return_12m_percentile'] = percentile_12m
            score += score_12m
            
        except Exception as e:
            logger.warning(f"Momentum factor calculation failed for {symbol}: {e}")
            score = 20  # Default moderate score
            breakdown['momentum_error'] = str(e)
        
        breakdown['momentum_factors_total'] = score
        return score
    
    def _score_quality_factors(self, fundamental_data: Dict, breakdown: Dict) -> float:
        """
        Score quality factors (35 points total)
        
        Components:
        - Earnings quality score (15 points)
        - Balance sheet strength (10 points)
        - Cash flow consistency (10 points)
        """
        
        score = 0
        
        if not fundamental_data:
            breakdown['quality_factors_total'] = 15  # Default moderate score
            return 15
        
        # Earnings quality scoring (15 points)
        # Based on ROE consistency, profit margins, and earnings growth
        roe = fundamental_data.get('roe', 0)
        net_margin = fundamental_data.get('net_margin', 0)
        eps_growth = fundamental_data.get('eps_growth_yoy', 0)
        
        earnings_quality = 0
        if roe > 0.15 and net_margin > 0.10:  # Strong profitability
            earnings_quality += 7
        elif roe > 0.10 and net_margin > 0.05:  # Good profitability
            earnings_quality += 5
        elif roe > 0.05:  # Acceptable profitability
            earnings_quality += 3
        
        if eps_growth > 0.15:  # Strong earnings growth
            earnings_quality += 8
        elif eps_growth > 0.10:  # Good earnings growth
            earnings_quality += 6
        elif eps_growth > 0.05:  # Moderate earnings growth
            earnings_quality += 4
        elif eps_growth > 0:  # Positive earnings growth
            earnings_quality += 2
        
        earnings_quality = min(earnings_quality, 15)  # Cap at 15 points
        breakdown['earnings_quality_score'] = earnings_quality
        score += earnings_quality
        
        # Balance sheet strength scoring (10 points)
        debt_equity = fundamental_data.get('debt_equity', 999)
        current_ratio = fundamental_data.get('current_ratio', 0)
        
        balance_sheet_score = 0
        if debt_equity < 0.3 and current_ratio > 1.5:  # Very strong
            balance_sheet_score = 10
        elif debt_equity < 0.5 and current_ratio > 1.2:  # Strong
            balance_sheet_score = 8
        elif debt_equity < 0.8 and current_ratio > 1.0:  # Good
            balance_sheet_score = 6
        elif debt_equity < 1.0:  # Acceptable
            balance_sheet_score = 3
        
        breakdown['balance_sheet_score'] = balance_sheet_score
        score += balance_sheet_score
        
        # Cash flow consistency (10 points) - simplified
        # This would require historical cash flow data analysis
        cash_flow_score = 6  # Default moderate score
        breakdown['cash_flow_score'] = cash_flow_score
        score += cash_flow_score
        
        breakdown['quality_factors_total'] = score
        return score
    
    def _score_value_factors(self, symbol: str, fundamental_data: Dict, price_data: pd.DataFrame, breakdown: Dict) -> float:
        """
        Score value factors (25 points total)
        
        Components:
        - P/E relative to sector (10 points)
        - P/B relative to history (8 points)
        - EV/EBITDA attractiveness (7 points)
        """
        
        score = 0
        
        if not fundamental_data:
            breakdown['value_factors_total'] = 12  # Default moderate score
            return 12
        
        # P/E relative scoring (10 points)
        pe_ratio = fundamental_data.get('pe_ratio', 999)
        
        if pe_ratio < 10:  # Very cheap
            pe_score = 10
        elif pe_ratio < 15:  # Cheap
            pe_score = 8
        elif pe_ratio < 20:  # Fair value
            pe_score = 6
        elif pe_ratio < 25:  # Slightly expensive
            pe_score = 3
        elif pe_ratio < 30:  # Expensive
            pe_score = 1
        else:  # Very expensive
            pe_score = 0
        
        breakdown['pe_score'] = pe_score
        breakdown['pe_value'] = pe_ratio
        score += pe_score
        
        # P/B relative scoring (8 points)
        pb_ratio = fundamental_data.get('pb_ratio', 999)
        
        if pb_ratio < 1.0:  # Trading below book value
            pb_score = 8
        elif pb_ratio < 1.5:  # Reasonable P/B
            pb_score = 6
        elif pb_ratio < 2.0:  # Fair P/B
            pb_score = 4
        elif pb_ratio < 3.0:  # High P/B
            pb_score = 2
        else:  # Very high P/B
            pb_score = 0
        
        breakdown['pb_score'] = pb_score
        breakdown['pb_value'] = pb_ratio
        score += pb_score
        
        # EV/EBITDA scoring (7 points)
        ev_ebitda = fundamental_data.get('ev_ebitda', 999)
        
        if ev_ebitda < 8:  # Very attractive
            ev_score = 7
        elif ev_ebitda < 12:  # Attractive
            ev_score = 5
        elif ev_ebitda < 15:  # Fair
            ev_score = 3
        elif ev_ebitda < 20:  # Expensive
            ev_score = 1
        else:  # Very expensive
            ev_score = 0
        
        breakdown['ev_ebitda_score'] = ev_score
        breakdown['ev_ebitda_value'] = ev_ebitda
        score += ev_score
        
        breakdown['value_factors_total'] = score
        return score
    
    def _calculate_period_returns(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate returns for different time periods"""
        
        returns = {}
        
        try:
            current_price = price_data['close'].iloc[-1]
            
            # 1-month return
            if len(price_data) >= 22:  # ~22 trading days in a month
                price_1m_ago = price_data['close'].iloc[-22]
                returns['1m'] = (current_price - price_1m_ago) / price_1m_ago
            
            # 3-month return
            if len(price_data) >= 66:  # ~66 trading days in 3 months
                price_3m_ago = price_data['close'].iloc[-66]
                returns['3m'] = (current_price - price_3m_ago) / price_3m_ago
            
            # 6-month return
            if len(price_data) >= 132:  # ~132 trading days in 6 months
                price_6m_ago = price_data['close'].iloc[-132]
                returns['6m'] = (current_price - price_6m_ago) / price_6m_ago
            
            # 12-month return
            if len(price_data) >= 252:  # ~252 trading days in a year
                price_12m_ago = price_data['close'].iloc[-252]
                returns['12m'] = (current_price - price_12m_ago) / price_12m_ago
            
        except Exception as e:
            logger.error(f"Return calculation error: {e}")
        
        return returns
    
    def _calculate_percentile(self, value: float, universe_values: List[float]) -> float:
        """Calculate percentile rank of value in universe"""
        
        if not universe_values:
            return 50.0  # Default to median if no universe data
        
        try:
            percentile = stats.percentileofscore(universe_values, value)
            return percentile
        except:
            return 50.0
    
    async def _get_universe_returns(self) -> Dict[str, List[float]]:
        """Get returns for the entire tradeable universe for comparison"""
        
        # This would ideally be calculated and cached regularly
        # For now, return placeholder data
        return {
            '1m': list(np.random.normal(0.02, 0.08, 300)),  # 2% mean, 8% std
            '3m': list(np.random.normal(0.06, 0.15, 300)),  # 6% mean, 15% std
            '6m': list(np.random.normal(0.12, 0.25, 300)),  # 12% mean, 25% std
            '12m': list(np.random.normal(0.20, 0.35, 300))  # 20% mean, 35% std
        }
    
    async def _get_price_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Get historical price data for symbol"""
        
        try:
            # Query from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            market_data = self.db_session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == '1d',
                MarketData.timestamp >= start_date,
                MarketData.timestamp <= end_date
            ).order_by(MarketData.timestamp).all()
            
            if not market_data:
                return None
            
            # Convert to DataFrame
            data = []
            for record in market_data:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open,
                    'high': record.high,
                    'low': record.low,
                    'close': record.close,
                    'volume': record.volume
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def _get_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """Get latest fundamental data for symbol"""
        
        try:
            latest_data = self.db_session.query(FundamentalData).filter(
                FundamentalData.symbol == symbol
            ).order_by(FundamentalData.report_date.desc()).first()
            
            if not latest_data:
                return None
            
            return {
                'pe_ratio': latest_data.pe_ratio,
                'pb_ratio': latest_data.pb_ratio,
                'ev_ebitda': latest_data.ev_ebitda,
                'roe': latest_data.roe,
                'net_margin': latest_data.net_margin,
                'eps_growth_yoy': latest_data.eps_growth_yoy,
                'debt_equity': latest_data.debt_equity,
                'current_ratio': latest_data.current_ratio
            }
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return None


class MacroSentimentAnalyzer:
    """
    Macro sentiment and positioning analysis (20% weight)
    
    Scoring Components (100 points total):
    1. Sector Rotation (35 points)
    2. Market Environment (30 points)
    3. Economic Sensitivity (25 points)
    4. Policy Impact (10 points)
    """
    
    def __init__(self):
        self.config = get_trading_config()
    
    async def calculate_macro_score(self, symbol: str) -> ScoringResult:
        """
        Calculate macro sentiment and positioning score
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ScoringResult with macro analysis
        """
        
        try:
            # Get stock sector information
            stock_sector = await self._get_stock_sector(symbol)
            
            score_breakdown = {}
            total_score = 0
            
            # 1. Sector Rotation (35 points)
            sector_score = await self._score_sector_rotation(stock_sector, score_breakdown)
            total_score += sector_score
            
            # 2. Market Environment (30 points)
            market_score = await self._score_market_environment(score_breakdown)
            total_score += market_score
            
            # 3. Economic Sensitivity (25 points)
            economic_score = await self._score_economic_sensitivity(symbol, stock_sector, score_breakdown)
            total_score += economic_score
            
            # 4. Policy Impact (10 points)
            policy_score = await self._score_policy_impact(stock_sector, score_breakdown)
            total_score += policy_score
            
            # Normalize to 0-1 scale
            normalized_score = total_score / 100.0
            
            logger.debug(f"Macro score for {symbol}: {normalized_score:.3f} ({total_score}/100)")
            
            return ScoringResult(
                score=normalized_score,
                breakdown=score_breakdown,
                raw_total=total_score
            )
            
        except Exception as e:
            logger.error(f"Macro scoring failed for {symbol}: {e}")
            return ScoringResult(
                score=0.5,  # Default neutral score
                breakdown={'error': str(e)},
                raw_total=50.0
            )
    
    async def _score_sector_rotation(self, sector: str, breakdown: Dict) -> float:
        """
        Score sector rotation and momentum (35 points total)
        
        Components:
        - Sector momentum vs market (20 points)
        - FII/DII sector flows (15 points)
        """
        
        score = 0
        
        try:
            # Get sector performance data
            api_manager = await get_api_manager()
            sonar = api_manager.get_sonar()
            
            # Get sector flows
            sector_flows = await sonar.get_sector_flows()
            
            # Sector momentum scoring (20 points)
            sector_momentum = self._calculate_sector_momentum(sector, sector_flows)
            
            if sector_momentum > 0.10:  # Strong positive momentum
                momentum_score = 20
            elif sector_momentum > 0.05:  # Good momentum
                momentum_score = 15
            elif sector_momentum > 0:  # Positive momentum
                momentum_score = 10
            elif sector_momentum > -0.05:  # Slight negative momentum
                momentum_score = 5
            else:  # Strong negative momentum
                momentum_score = 0
            
            breakdown['sector_momentum_score'] = momentum_score
            breakdown['sector_momentum_value'] = sector_momentum
            score += momentum_score
            
            # FII/DII flows scoring (15 points)
            fii_flows = sector_flows.get('fii_flows', {}).get(sector, 0)
            dii_flows = sector_flows.get('dii_flows', {}).get(sector, 0)
            
            total_flows = fii_flows + dii_flows
            
            if total_flows > 1000:  # Strong inflows (in crores)
                flows_score = 15
            elif total_flows > 500:  # Good inflows
                flows_score = 12
            elif total_flows > 0:  # Positive inflows
                flows_score = 8
            elif total_flows > -500:  # Minor outflows
                flows_score = 4
            else:  # Strong outflows
                flows_score = 0
            
            breakdown['sector_flows_score'] = flows_score
            breakdown['fii_flows'] = fii_flows
            breakdown['dii_flows'] = dii_flows
            score += flows_score
            
        except Exception as e:
            logger.warning(f"Sector rotation scoring error: {e}")
            score = 17  # Default moderate score
            breakdown['sector_rotation_error'] = str(e)
        
        breakdown['sector_rotation_total'] = score
        return score
    
    async def _score_market_environment(self, breakdown: Dict) -> float:
        """
        Score market environment (30 points total)
        
        Components:
        - VIX positioning (15 points)
        - Market breadth indicators (15 points)
        """
        
        score = 0
        
        try:
            # Get market sentiment data
            api_manager = await get_api_manager()
            sonar = api_manager.get_sonar()
            
            market_sentiment = await sonar.get_market_sentiment()
            
            # VIX positioning scoring (15 points)
            vix_level = market_sentiment.get('vix_level', 20)
            
            if vix_level < 15:  # Very low volatility - good for risk-taking
                vix_score = 15
            elif vix_level < 20:  # Low volatility
                vix_score = 12
            elif vix_level < 25:  # Normal volatility
                vix_score = 8
            elif vix_level < 30:  # Elevated volatility
                vix_score = 4
            else:  # High volatility - risk-off environment
                vix_score = 0
            
            breakdown['vix_score'] = vix_score
            breakdown['vix_level'] = vix_level
            score += vix_score
            
            # Market breadth scoring (15 points)
            market_breadth = market_sentiment.get('market_breadth', 0.5)
            
            if market_breadth > 0.7:  # Strong breadth
                breadth_score = 15
            elif market_breadth > 0.6:  # Good breadth
                breadth_score = 12
            elif market_breadth > 0.5:  # Neutral breadth
                breadth_score = 8
            elif market_breadth > 0.4:  # Weak breadth
                breadth_score = 4
            else:  # Very weak breadth
                breadth_score = 0
            
            breakdown['market_breadth_score'] = breadth_score
            breakdown['market_breadth_value'] = market_breadth
            score += breadth_score
            
        except Exception as e:
            logger.warning(f"Market environment scoring error: {e}")
            score = 15  # Default moderate score
            breakdown['market_environment_error'] = str(e)
        
        breakdown['market_environment_total'] = score
        return score
    
    async def _score_economic_sensitivity(self, symbol: str, sector: str, breakdown: Dict) -> float:
        """
        Score economic sensitivity (25 points total)
        
        Components:
        - Interest rate sensitivity (15 points)
        - Currency exposure impact (10 points)
        """
        
        score = 0
        
        # Interest rate sensitivity scoring (15 points)
        # Based on sector characteristics
        rate_sensitivity = self._get_sector_rate_sensitivity(sector)
        
        # In rising rate environment, prefer low sensitivity stocks
        # In falling rate environment, prefer high sensitivity stocks
        current_rate_trend = await self._get_rate_trend()
        
        if current_rate_trend == 'FALLING':
            # Prefer rate-sensitive sectors (banking, real estate)
            if rate_sensitivity == 'HIGH':
                rate_score = 15
            elif rate_sensitivity == 'MEDIUM':
                rate_score = 10
            else:
                rate_score = 8
        elif current_rate_trend == 'RISING':
            # Prefer rate-insensitive sectors
            if rate_sensitivity == 'LOW':
                rate_score = 15
            elif rate_sensitivity == 'MEDIUM':
                rate_score = 10
            else:
                rate_score = 5
        else:  # STABLE
            rate_score = 10  # Neutral score
        
        breakdown['rate_sensitivity_score'] = rate_score
        breakdown['rate_sensitivity'] = rate_sensitivity
        breakdown['rate_trend'] = current_rate_trend
        score += rate_score
        
        # Currency exposure scoring (10 points)
        currency_exposure = self._get_currency_exposure(sector)
        usd_inr_trend = await self._get_currency_trend()
        
        if usd_inr_trend == 'WEAKENING':  # INR strengthening
            # Prefer import-heavy sectors
            if currency_exposure == 'IMPORT_HEAVY':
                currency_score = 10
            elif currency_exposure == 'NEUTRAL':
                currency_score = 6
            else:
                currency_score = 3
        elif usd_inr_trend == 'STRENGTHENING':  # INR weakening
            # Prefer export-heavy sectors
            if currency_exposure == 'EXPORT_HEAVY':
                currency_score = 10
            elif currency_exposure == 'NEUTRAL':
                currency_score = 6
            else:
                currency_score = 3
        else:  # STABLE
            currency_score = 6  # Neutral score
        
        breakdown['currency_exposure_score'] = currency_score
        breakdown['currency_exposure'] = currency_exposure
        breakdown['usd_inr_trend'] = usd_inr_trend
        score += currency_score
        
        breakdown['economic_sensitivity_total'] = score
        return score
    
    async def _score_policy_impact(self, sector: str, breakdown: Dict) -> float:
        """
        Score policy impact (10 points total)
        
        Components:
        - Regulatory tailwinds/headwinds (10 points)
        """
        
        # This would be enhanced with real-time policy tracking
        # For now, using sector-based policy assessment
        
        policy_impact = self._get_sector_policy_impact(sector)
        
        if policy_impact == 'STRONG_POSITIVE':
            policy_score = 10
        elif policy_impact == 'POSITIVE':
            policy_score = 8
        elif policy_impact == 'NEUTRAL':
            policy_score = 6
        elif policy_impact == 'NEGATIVE':
            policy_score = 3
        else:  # STRONG_NEGATIVE
            policy_score = 0
        
        breakdown['policy_impact_score'] = policy_score
        breakdown['policy_impact'] = policy_impact
        
        return policy_score
    
    def _calculate_sector_momentum(self, sector: str, sector_flows: Dict) -> float:
        """Calculate sector momentum vs market"""
        # Simplified calculation - would be enhanced with real data
        return np.random.normal(0.03, 0.08)  # 3% mean with 8% volatility
    
    def _get_sector_rate_sensitivity(self, sector: str) -> str:
        """Get interest rate sensitivity for sector"""
        
        high_sensitivity = ['BANKING', 'REALTY', 'AUTO', 'CAPITAL_GOODS']
        medium_sensitivity = ['FMCG', 'TELECOM', 'POWER']
        
        if sector in high_sensitivity:
            return 'HIGH'
        elif sector in medium_sensitivity:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_currency_exposure(self, sector: str) -> str:
        """Get currency exposure for sector"""
        
        export_heavy = ['IT', 'PHARMA', 'TEXTILES', 'CHEMICALS']
        import_heavy = ['OIL_GAS', 'METALS', 'AUTO']
        
        if sector in export_heavy:
            return 'EXPORT_HEAVY'
        elif sector in import_heavy:
            return 'IMPORT_HEAVY'
        else:
            return 'NEUTRAL'
    
    def _get_sector_policy_impact(self, sector: str) -> str:
        """Get current policy impact for sector"""
        
        # This would be updated based on current policy environment
        # Placeholder implementation
        positive_policy = ['POWER', 'INFRASTRUCTURE', 'DEFENCE']
        neutral_policy = ['FMCG', 'IT', 'PHARMA']
        
        if sector in positive_policy:
            return 'POSITIVE'
        elif sector in neutral_policy:
            return 'NEUTRAL'
        else:
            return 'NEUTRAL'  # Default
    
    async def _get_stock_sector(self, symbol: str) -> str:
        """Get sector for stock symbol"""
        
        try:
            db_session = get_db_session()
            stock = db_session.query(Stock).filter(Stock.symbol == symbol).first()
            
            if stock and stock.sector:
                return stock.sector
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            logger.error(f"Error fetching sector for {symbol}: {e}")
            return 'UNKNOWN'
    
    async def _get_rate_trend(self) -> str:
        """Get current interest rate trend"""
        # This would be enhanced with real rate data
        return 'STABLE'  # Placeholder
    
    async def _get_currency_trend(self) -> str:
        """Get current USD/INR trend"""
        # This would be enhanced with real currency data
        return 'STABLE'  # Placeholder
