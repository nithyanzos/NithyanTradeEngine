"""
Advanced Trade Filtering Engine
Multi-factor scoring system for identifying top trading opportunities
"""

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

from config.trading_config import get_trading_config
from config.api_config import get_api_manager
from config.database_config import get_db_session, Stock, FundamentalData, MarketData

logger = logging.getLogger(__name__)

@dataclass
class ScoringResult:
    """Individual scoring component result"""
    score: float  # 0.0 to 1.0
    breakdown: Dict[str, Union[float, int]]
    raw_total: float
    timestamp: datetime = field(default_factory=datetime.now)

class FundamentalAnalyzer:
    """
    Comprehensive fundamental analysis scoring (25% weight)
    
    Scoring Components (100 points total):
    1. Profitability Metrics (35 points)
    2. Financial Health (30 points)  
    3. Growth Quality (25 points)
    4. Governance & Ownership (10 points)
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.db_session = get_db_session()
    
    def calculate_fundamental_score(self, symbol: str) -> ScoringResult:
        """
        Calculate comprehensive fundamental score
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            
        Returns:
            ScoringResult with score 0.0-1.0 and detailed breakdown
        """
        
        try:
            # Get latest fundamental data
            financial_data = self._get_latest_financials(symbol)
            
            if not financial_data:
                logger.warning(f"No fundamental data found for {symbol}")
                return ScoringResult(
                    score=0.0, 
                    breakdown={'error': 'No fundamental data'}, 
                    raw_total=0.0
                )
            
            score_breakdown = {}
            total_score = 0
            
            # 1. Profitability Metrics (35 points)
            profitability_score = self._score_profitability(financial_data, score_breakdown)
            total_score += profitability_score
            
            # 2. Financial Health (30 points)
            health_score = self._score_financial_health(financial_data, score_breakdown)
            total_score += health_score
            
            # 3. Growth Quality (25 points)
            growth_score = self._score_growth_quality(financial_data, score_breakdown)
            total_score += growth_score
            
            # 4. Governance & Ownership (10 points)
            governance_score = self._score_governance(financial_data, score_breakdown)
            total_score += governance_score
            
            # Normalize to 0-1 scale
            normalized_score = total_score / 100.0
            
            logger.debug(f"Fundamental score for {symbol}: {normalized_score:.3f} ({total_score}/100)")
            
            return ScoringResult(
                score=normalized_score,
                breakdown=score_breakdown,
                raw_total=total_score
            )
            
        except Exception as e:
            logger.error(f"Fundamental scoring failed for {symbol}: {e}")
            return ScoringResult(
                score=0.0,
                breakdown={'error': str(e)},
                raw_total=0.0
            )
    
    def _score_profitability(self, data: Dict, breakdown: Dict) -> float:
        """
        Score profitability metrics (35 points total)
        
        Components:
        - ROE > 15% (15 points)
        - Operating margin > 10% (10 points)
        - ROIC > 12% (10 points)
        """
        
        score = 0
        
        # ROE scoring (15 points)
        roe = data.get('roe', 0)
        if roe > 0.25:  # Exceptional ROE
            roe_score = 15
        elif roe > 0.20:  # Very good ROE
            roe_score = 12
        elif roe > 0.15:  # Good ROE
            roe_score = 10
        elif roe > 0.10:  # Average ROE
            roe_score = 5
        else:
            roe_score = 0
        
        breakdown['roe_score'] = roe_score
        breakdown['roe_value'] = roe
        score += roe_score
        
        # Operating margin scoring (10 points)
        op_margin = data.get('operating_margin', 0)
        if op_margin > 0.20:  # Exceptional margin
            margin_score = 10
        elif op_margin > 0.15:  # Very good margin
            margin_score = 8
        elif op_margin > 0.10:  # Good margin
            margin_score = 6
        elif op_margin > 0.05:  # Average margin
            margin_score = 3
        else:
            margin_score = 0
        
        breakdown['operating_margin_score'] = margin_score
        breakdown['operating_margin_value'] = op_margin
        score += margin_score
        
        # ROIC scoring (10 points)
        roic = data.get('roic', 0)
        if roic > 0.20:  # Exceptional ROIC
            roic_score = 10
        elif roic > 0.15:  # Very good ROIC
            roic_score = 8
        elif roic > 0.12:  # Good ROIC
            roic_score = 6
        elif roic > 0.08:  # Average ROIC
            roic_score = 3
        else:
            roic_score = 0
        
        breakdown['roic_score'] = roic_score
        breakdown['roic_value'] = roic
        score += roic_score
        
        breakdown['profitability_total'] = score
        return score
    
    def _score_financial_health(self, data: Dict, breakdown: Dict) -> float:
        """
        Score financial health metrics (30 points total)
        
        Components:
        - Debt/Equity < 0.5 (15 points)
        - Interest coverage > 5x (10 points)
        - Current ratio > 1.2 (5 points)
        """
        
        score = 0
        
        # Debt/Equity scoring (15 points)
        debt_equity = data.get('debt_equity', 999)  # Default to high value
        if debt_equity < 0.2:  # Very low debt
            de_score = 15
        elif debt_equity < 0.3:  # Low debt
            de_score = 12
        elif debt_equity < 0.5:  # Acceptable debt
            de_score = 8
        elif debt_equity < 0.8:  # High debt
            de_score = 3
        else:
            de_score = 0
        
        breakdown['debt_equity_score'] = de_score
        breakdown['debt_equity_value'] = debt_equity
        score += de_score
        
        # Interest coverage scoring (10 points)
        interest_coverage = data.get('interest_coverage', 0)
        if interest_coverage > 10:  # Very strong coverage
            coverage_score = 10
        elif interest_coverage > 7:  # Strong coverage
            coverage_score = 8
        elif interest_coverage > 5:  # Good coverage
            coverage_score = 6
        elif interest_coverage > 3:  # Acceptable coverage
            coverage_score = 3
        else:
            coverage_score = 0
        
        breakdown['interest_coverage_score'] = coverage_score
        breakdown['interest_coverage_value'] = interest_coverage
        score += coverage_score
        
        # Current ratio scoring (5 points)
        current_ratio = data.get('current_ratio', 0)
        if current_ratio > 2.0:  # Very strong liquidity
            current_score = 5
        elif current_ratio > 1.5:  # Strong liquidity
            current_score = 4
        elif current_ratio > 1.2:  # Good liquidity
            current_score = 3
        elif current_ratio > 1.0:  # Acceptable liquidity
            current_score = 1
        else:
            current_score = 0
        
        breakdown['current_ratio_score'] = current_score
        breakdown['current_ratio_value'] = current_ratio
        score += current_score
        
        breakdown['financial_health_total'] = score
        return score
    
    def _score_growth_quality(self, data: Dict, breakdown: Dict) -> float:
        """
        Score growth quality metrics (25 points total)
        
        Components:
        - Revenue CAGR > 10% (10 points)
        - EPS Growth Consistency (10 points)
        - Book Value Growth (5 points)
        """
        
        score = 0
        
        # Revenue growth scoring (10 points)
        revenue_growth = data.get('revenue_growth_yoy', 0)
        if revenue_growth > 0.25:  # Exceptional growth
            revenue_score = 10
        elif revenue_growth > 0.20:  # Very good growth
            revenue_score = 8
        elif revenue_growth > 0.15:  # Good growth
            revenue_score = 6
        elif revenue_growth > 0.10:  # Acceptable growth
            revenue_score = 4
        elif revenue_growth > 0.05:  # Slow growth
            revenue_score = 2
        else:
            revenue_score = 0
        
        breakdown['revenue_growth_score'] = revenue_score
        breakdown['revenue_growth_value'] = revenue_growth
        score += revenue_score
        
        # EPS growth scoring (10 points)
        eps_growth = data.get('eps_growth_yoy', 0)
        if eps_growth > 0.25:  # Exceptional EPS growth
            eps_score = 10
        elif eps_growth > 0.20:  # Very good EPS growth
            eps_score = 8
        elif eps_growth > 0.15:  # Good EPS growth
            eps_score = 6
        elif eps_growth > 0.10:  # Acceptable EPS growth
            eps_score = 4
        elif eps_growth > 0.05:  # Slow EPS growth
            eps_score = 2
        else:
            eps_score = 0
        
        breakdown['eps_growth_score'] = eps_score
        breakdown['eps_growth_value'] = eps_growth
        score += eps_score
        
        # Book value growth (5 points) - simplified for now
        # This would require historical book value data
        book_value_score = 3  # Default moderate score
        breakdown['book_value_score'] = book_value_score
        score += book_value_score
        
        breakdown['growth_quality_total'] = score
        return score
    
    def _score_governance(self, data: Dict, breakdown: Dict) -> float:
        """
        Score governance and ownership metrics (10 points total)
        
        Components:
        - Promoter holding > 50% (5 points)
        - Institutional holding growth (5 points)
        """
        
        score = 0
        
        # Promoter holding scoring (5 points)
        promoter_holding = data.get('promoter_holding', 0)
        if promoter_holding > 0.70:  # Very high promoter confidence
            promoter_score = 5
        elif promoter_holding > 0.60:  # High promoter confidence
            promoter_score = 4
        elif promoter_holding > 0.50:  # Good promoter confidence
            promoter_score = 3
        elif promoter_holding > 0.40:  # Moderate promoter confidence
            promoter_score = 2
        else:
            promoter_score = 0
        
        breakdown['promoter_holding_score'] = promoter_score
        breakdown['promoter_holding_value'] = promoter_holding
        score += promoter_score
        
        # Institutional holding (5 points) - simplified
        institutional_holding = data.get('institutional_holding', 0)
        if institutional_holding > 0.20:  # Good institutional interest
            institutional_score = 5
        elif institutional_holding > 0.15:  # Moderate institutional interest
            institutional_score = 3
        elif institutional_holding > 0.10:  # Some institutional interest
            institutional_score = 2
        else:
            institutional_score = 0
        
        breakdown['institutional_holding_score'] = institutional_score
        breakdown['institutional_holding_value'] = institutional_holding
        score += institutional_score
        
        breakdown['governance_total'] = score
        return score
    
    def _get_latest_financials(self, symbol: str) -> Optional[Dict]:
        """Get latest fundamental data for symbol"""
        
        try:
            # Query latest fundamental data from database
            latest_data = self.db_session.query(FundamentalData).filter(
                FundamentalData.symbol == symbol
            ).order_by(FundamentalData.report_date.desc()).first()
            
            if not latest_data:
                return None
            
            return {
                'roe': latest_data.roe,
                'operating_margin': latest_data.operating_margin,
                'roic': latest_data.roic,
                'debt_equity': latest_data.debt_equity,
                'interest_coverage': latest_data.interest_coverage,
                'current_ratio': latest_data.current_ratio,
                'revenue_growth_yoy': latest_data.revenue_growth_yoy,
                'eps_growth_yoy': latest_data.eps_growth_yoy,
                'promoter_holding': latest_data.promoter_holding,
                'institutional_holding': latest_data.institutional_holding,
                'report_date': latest_data.report_date
            }
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return None


class TechnicalAnalyzer:
    """
    Advanced technical analysis scoring (30% weight)
    
    Scoring Components (100 points total):
    1. Momentum Indicators (40 points)
    2. Trend Strength (25 points)
    3. Volume Analysis (20 points)
    4. Relative Strength (15 points)
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.db_session = get_db_session()
    
    async def calculate_technical_score(self, symbol: str) -> ScoringResult:
        """
        Calculate comprehensive technical score
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ScoringResult with detailed technical analysis
        """
        
        try:
            # Get historical price data
            price_data = await self._get_price_data(symbol, days=200)
            
            if price_data is None or len(price_data) < 50:
                logger.warning(f"Insufficient price data for {symbol}")
                return ScoringResult(
                    score=0.0,
                    breakdown={'error': 'Insufficient price data'},
                    raw_total=0.0
                )
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(price_data)
            
            score_breakdown = {}
            total_score = 0
            
            # 1. Momentum Indicators (40 points)
            momentum_score = self._score_momentum(indicators, score_breakdown)
            total_score += momentum_score
            
            # 2. Trend Strength (25 points)
            trend_score = self._score_trend_strength(indicators, score_breakdown)
            total_score += trend_score
            
            # 3. Volume Analysis (20 points)
            volume_score = self._score_volume_analysis(indicators, score_breakdown)
            total_score += volume_score
            
            # 4. Relative Strength (15 points)
            relative_score = await self._score_relative_strength(symbol, indicators, score_breakdown)
            total_score += relative_score
            
            # Normalize to 0-1 scale
            normalized_score = total_score / 100.0
            
            logger.debug(f"Technical score for {symbol}: {normalized_score:.3f} ({total_score}/100)")
            
            return ScoringResult(
                score=normalized_score,
                breakdown=score_breakdown,
                raw_total=total_score
            )
            
        except Exception as e:
            logger.error(f"Technical scoring failed for {symbol}: {e}")
            return ScoringResult(
                score=0.0,
                breakdown={'error': str(e)},
                raw_total=0.0
            )
    
    def _calculate_indicators(self, price_data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        
        # Extract OHLCV arrays
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values
        volume = price_data['volume'].values
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = ta.SMA(close, timeperiod=20)
        indicators['sma_50'] = ta.SMA(close, timeperiod=50)
        indicators['sma_200'] = ta.SMA(close, timeperiod=200)
        
        # Momentum indicators
        indicators['rsi'] = ta.RSI(close, timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = ta.MACD(close)
        
        # Trend indicators
        indicators['adx'] = ta.ADX(high, low, close, timeperiod=14)
        
        # Volatility indicators
        indicators['atr'] = ta.ATR(high, low, close, timeperiod=14)
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = ta.BBANDS(close)
        
        # Volume indicators
        indicators['obv'] = ta.OBV(close, volume)
        indicators['volume_sma'] = ta.SMA(volume, timeperiod=20)
        
        # Current values
        indicators['current_price'] = close[-1]
        indicators['current_volume'] = volume[-1]
        
        return indicators
    
    def _score_momentum(self, indicators: Dict, breakdown: Dict) -> float:
        """
        Score momentum indicators (40 points total)
        
        Components:
        - RSI in 40-70 range (10 points)
        - MACD bullish crossover (15 points)
        - Price above 50-day MA (15 points)
        """
        
        score = 0
        
        # RSI scoring (10 points)
        rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
        
        if 45 <= rsi <= 65:  # Ideal RSI range
            rsi_score = 10
        elif 40 <= rsi <= 70:  # Good RSI range
            rsi_score = 8
        elif 35 <= rsi <= 75:  # Acceptable RSI range
            rsi_score = 5
        elif 30 <= rsi <= 80:  # Marginal RSI range
            rsi_score = 2
        else:
            rsi_score = 0
        
        breakdown['rsi_score'] = rsi_score
        breakdown['rsi_value'] = rsi
        score += rsi_score
        
        # MACD scoring (15 points)
        macd = indicators['macd'][-1] if len(indicators['macd']) > 0 else 0
        macd_signal = indicators['macd_signal'][-1] if len(indicators['macd_signal']) > 0 else 0
        macd_hist = indicators['macd_hist'][-1] if len(indicators['macd_hist']) > 0 else 0
        
        if macd > macd_signal and macd_hist > 0:  # Bullish MACD
            if macd_hist > indicators['macd_hist'][-2]:  # Increasing momentum
                macd_score = 15
            else:
                macd_score = 10
        elif macd > macd_signal:  # MACD above signal
            macd_score = 8
        else:
            macd_score = 0
        
        breakdown['macd_score'] = macd_score
        breakdown['macd_value'] = macd
        breakdown['macd_signal_value'] = macd_signal
        score += macd_score
        
        # Price vs MA50 scoring (15 points)
        current_price = indicators['current_price']
        sma_50 = indicators['sma_50'][-1] if len(indicators['sma_50']) > 0 else current_price
        
        price_vs_ma50 = (current_price - sma_50) / sma_50
        
        if price_vs_ma50 > 0.05:  # 5%+ above MA50
            ma_score = 15
        elif price_vs_ma50 > 0.02:  # 2%+ above MA50
            ma_score = 12
        elif price_vs_ma50 > 0:  # Above MA50
            ma_score = 8
        elif price_vs_ma50 > -0.02:  # Slightly below MA50
            ma_score = 3
        else:
            ma_score = 0
        
        breakdown['price_vs_ma50_score'] = ma_score
        breakdown['price_vs_ma50_pct'] = price_vs_ma50 * 100
        score += ma_score
        
        breakdown['momentum_total'] = score
        return score
    
    def _score_trend_strength(self, indicators: Dict, breakdown: Dict) -> float:
        """
        Score trend strength indicators (25 points total)
        
        Components:
        - ADX > 25 (10 points)
        - Price above 200-day MA (10 points)
        - Higher highs/higher lows (5 points)
        """
        
        score = 0
        
        # ADX scoring (10 points)
        adx = indicators['adx'][-1] if len(indicators['adx']) > 0 else 20
        
        if adx > 40:  # Very strong trend
            adx_score = 10
        elif adx > 30:  # Strong trend
            adx_score = 8
        elif adx > 25:  # Good trend
            adx_score = 6
        elif adx > 20:  # Moderate trend
            adx_score = 3
        else:
            adx_score = 0
        
        breakdown['adx_score'] = adx_score
        breakdown['adx_value'] = adx
        score += adx_score
        
        # Price vs MA200 scoring (10 points)
        current_price = indicators['current_price']
        sma_200 = indicators['sma_200'][-1] if len(indicators['sma_200']) > 0 else current_price
        
        price_vs_ma200 = (current_price - sma_200) / sma_200
        
        if price_vs_ma200 > 0.10:  # 10%+ above MA200
            ma200_score = 10
        elif price_vs_ma200 > 0.05:  # 5%+ above MA200
            ma200_score = 8
        elif price_vs_ma200 > 0:  # Above MA200
            ma200_score = 6
        elif price_vs_ma200 > -0.05:  # Slightly below MA200
            ma200_score = 2
        else:
            ma200_score = 0
        
        breakdown['price_vs_ma200_score'] = ma200_score
        breakdown['price_vs_ma200_pct'] = price_vs_ma200 * 100
        score += ma200_score
        
        # Higher highs/higher lows pattern (5 points) - simplified
        # This would require more sophisticated pattern recognition
        pattern_score = 3  # Default moderate score
        breakdown['pattern_score'] = pattern_score
        score += pattern_score
        
        breakdown['trend_strength_total'] = score
        return score
    
    def _score_volume_analysis(self, indicators: Dict, breakdown: Dict) -> float:
        """
        Score volume analysis (20 points total)
        
        Components:
        - Volume above 20-day average (10 points)
        - On-Balance Volume trending up (10 points)
        """
        
        score = 0
        
        # Volume vs average (10 points)
        current_volume = indicators['current_volume']
        avg_volume = indicators['volume_sma'][-1] if len(indicators['volume_sma']) > 0 else current_volume
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 2.0:  # Very high volume
            volume_score = 10
        elif volume_ratio > 1.5:  # High volume
            volume_score = 8
        elif volume_ratio > 1.2:  # Above average volume
            volume_score = 6
        elif volume_ratio > 1.0:  # Average volume
            volume_score = 3
        else:
            volume_score = 0
        
        breakdown['volume_score'] = volume_score
        breakdown['volume_ratio'] = volume_ratio
        score += volume_score
        
        # OBV trend (10 points) - simplified
        obv = indicators['obv']
        if len(obv) >= 20:
            obv_recent = obv[-5:].mean()  # Recent 5-day average
            obv_older = obv[-20:-15].mean()  # Older 5-day average
            
            obv_trend = (obv_recent - obv_older) / obv_older if obv_older != 0 else 0
            
            if obv_trend > 0.05:  # Strong OBV uptrend
                obv_score = 10
            elif obv_trend > 0.02:  # Moderate OBV uptrend
                obv_score = 6
            elif obv_trend > 0:  # Slight OBV uptrend
                obv_score = 3
            else:
                obv_score = 0
        else:
            obv_score = 5  # Default score for insufficient data
        
        breakdown['obv_score'] = obv_score
        score += obv_score
        
        breakdown['volume_analysis_total'] = score
        return score
    
    async def _score_relative_strength(self, symbol: str, indicators: Dict, breakdown: Dict) -> float:
        """
        Score relative strength (15 points total)
        
        Components:
        - Outperforming Nifty 50 (1M) (8 points)
        - Sector relative strength (7 points)
        """
        
        score = 0
        
        try:
            # Get Nifty 50 performance for comparison
            nifty_performance = await self._get_nifty_performance()
            
            # Calculate stock performance (1 month)
            stock_performance = await self._calculate_stock_performance(symbol, days=30)
            
            if stock_performance and nifty_performance:
                relative_performance = stock_performance - nifty_performance
                
                if relative_performance > 0.10:  # 10%+ outperformance
                    relative_score = 8
                elif relative_performance > 0.05:  # 5%+ outperformance
                    relative_score = 6
                elif relative_performance > 0:  # Any outperformance
                    relative_score = 4
                elif relative_performance > -0.05:  # Minor underperformance
                    relative_score = 2
                else:
                    relative_score = 0
            else:
                relative_score = 4  # Default score
            
            breakdown['relative_performance_score'] = relative_score
            breakdown['relative_performance_pct'] = relative_performance * 100 if 'relative_performance' in locals() else 0
            score += relative_score
            
        except Exception as e:
            logger.warning(f"Relative strength calculation failed for {symbol}: {e}")
            breakdown['relative_performance_score'] = 4  # Default score
            score += 4
        
        # Sector relative strength (7 points) - simplified
        sector_score = 4  # Default moderate score
        breakdown['sector_strength_score'] = sector_score
        score += sector_score
        
        breakdown['relative_strength_total'] = score
        return score
    
    async def _get_price_data(self, symbol: str, days: int = 200) -> Optional[pd.DataFrame]:
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
                # Fallback to API if no database data
                api_manager = await get_api_manager()
                zerodha = api_manager.get_zerodha()
                
                historical_data = await zerodha.get_historical_data(
                    symbol, start_date, end_date, "day"
                )
                
                if historical_data:
                    df = pd.DataFrame(historical_data)
                    df['timestamp'] = pd.to_datetime(df['date'])
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
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
    
    async def _get_nifty_performance(self, days: int = 30) -> Optional[float]:
        """Get Nifty 50 performance over specified period"""
        # This would be implemented to fetch Nifty 50 index performance
        # For now, return a placeholder
        return 0.05  # 5% placeholder performance
    
    async def _calculate_stock_performance(self, symbol: str, days: int = 30) -> Optional[float]:
        """Calculate stock performance over specified period"""
        
        try:
            price_data = await self._get_price_data(symbol, days + 10)
            
            if price_data is None or len(price_data) < days:
                return None
            
            current_price = price_data['close'].iloc[-1]
            past_price = price_data['close'].iloc[-days]
            
            return (current_price - past_price) / past_price
            
        except Exception as e:
            logger.error(f"Performance calculation failed for {symbol}: {e}")
            return None
