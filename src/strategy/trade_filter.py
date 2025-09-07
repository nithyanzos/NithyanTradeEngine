"""
Advanced Trade Filtering Engine
Master trade selection system combining all scoring components
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from config.trading_config import get_trading_config
from config.database_config import get_db_session, Stock
from analytics.fundamental_analyzer import FundamentalAnalyzer, ScoringResult
from analytics.fundamental_analyzer import TechnicalAnalyzer
from analytics.quantitative_engine import QuantitativeEngine, MacroSentimentAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class TradeOpportunity:
    """Comprehensive trade opportunity data structure"""
    
    symbol: str
    company_name: str
    sector: str
    market_cap_category: str
    
    # Composite scoring
    composite_score: float
    conviction_level: float
    meets_threshold: bool
    
    # Individual scores
    fundamental_score: ScoringResult
    technical_score: ScoringResult
    quantitative_score: ScoringResult
    macro_score: ScoringResult
    
    # Market data
    current_price: float
    market_cap: Optional[int]
    avg_volume: Optional[int]
    atr_pct: float
    
    # Investment rationale
    investment_rationale: str
    key_strengths: List[str]
    key_risks: List[str]
    
    # Additional metrics
    quality_score: float
    liquidity_score: float
    volatility_score: float
    
    timestamp: datetime = field(default_factory=datetime.now)


class UniverseManager:
    """
    Manages the tradeable universe of Nifty stocks
    
    Target Universe: ~300 stocks from major Nifty indices
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.db_session = get_db_session()
        self.universe_cache = {}
        self.cache_expiry = None
    
    def get_tradeable_universe(self) -> List[str]:
        """
        Get filtered tradeable universe
        
        Returns:
            List of stock symbols that meet basic criteria
        """
        
        try:
            # Check cache first (valid for 1 hour)
            if (self.cache_expiry and 
                datetime.now() < self.cache_expiry and 
                self.universe_cache):
                logger.debug(f"Using cached universe: {len(self.universe_cache)} stocks")
                return list(self.universe_cache.keys())
            
            # Query active stocks from database
            stocks = self.db_session.query(Stock).filter(
                Stock.is_active == True
            ).all()
            
            tradeable_symbols = []
            
            for stock in stocks:
                # Apply basic filters
                if self._meets_basic_criteria(stock):
                    tradeable_symbols.append(stock.symbol)
                    
                    # Cache stock info
                    self.universe_cache[stock.symbol] = {
                        'company_name': stock.company_name,
                        'sector': stock.sector,
                        'market_cap_category': stock.market_cap_category,
                        'market_cap': stock.market_cap,
                        'avg_volume': stock.avg_daily_volume
                    }
            
            # Set cache expiry
            self.cache_expiry = datetime.now() + timedelta(hours=1)
            
            logger.info(f"Universe updated: {len(tradeable_symbols)} tradeable stocks")
            return tradeable_symbols
            
        except Exception as e:
            logger.error(f"Error building tradeable universe: {e}")
            return []
    
    def _meets_basic_criteria(self, stock: Stock) -> bool:
        """
        Check if stock meets basic trading criteria
        
        Criteria:
        - Listed in major Nifty indices
        - Minimum market cap
        - Minimum liquidity
        - Active listing status
        """
        
        # Must be in at least one major index
        in_major_index = (stock.is_nifty50 or 
                         stock.is_nifty_next50 or 
                         stock.is_nifty_midcap100 or 
                         stock.is_nifty_smallcap100)
        
        if not in_major_index:
            return False
        
        # Market cap filter
        if stock.market_cap and stock.market_cap < self.config.MIN_MARKET_CAP:
            return False
        
        # Liquidity filters
        if (stock.avg_daily_volume and 
            stock.avg_daily_volume < self.config.MIN_DAILY_VOLUME):
            return False
        
        if (stock.avg_daily_value and 
            stock.avg_daily_value < self.config.MIN_DAILY_VALUE):
            return False
        
        return True
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get cached stock information"""
        return self.universe_cache.get(symbol)


class QualityFilters:
    """
    Additional quality screening filters
    """
    
    def __init__(self):
        self.config = get_trading_config()
    
    async def liquidity_screen(self, symbol: str) -> Dict[str, Union[bool, float, str]]:
        """
        Comprehensive liquidity screening
        
        Returns:
            Dict with pass/fail status and metrics
        """
        
        try:
            # This would integrate with real-time data
            # Placeholder implementation
            
            # Simulate liquidity metrics
            avg_volume = np.random.uniform(500000, 5000000)  # 5L to 50L shares
            avg_value = np.random.uniform(50000000, 500000000)  # ₹5Cr to ₹50Cr
            bid_ask_spread = np.random.uniform(0.001, 0.008)  # 0.1% to 0.8%
            
            # Apply filters
            volume_ok = avg_volume >= self.config.MIN_DAILY_VOLUME
            value_ok = avg_value >= self.config.MIN_DAILY_VALUE
            spread_ok = bid_ask_spread <= 0.005  # 0.5% max spread
            
            passes = volume_ok and value_ok and spread_ok
            
            return {
                'passes': passes,
                'avg_volume': avg_volume,
                'avg_value': avg_value,
                'bid_ask_spread_pct': bid_ask_spread * 100,
                'volume_ok': volume_ok,
                'value_ok': value_ok,
                'spread_ok': spread_ok
            }
            
        except Exception as e:
            logger.error(f"Liquidity screening failed for {symbol}: {e}")
            return {'passes': False, 'error': str(e)}
    
    async def volatility_screen(self, symbol: str, market_cap_category: str) -> Dict[str, Union[bool, float]]:
        """
        Volatility-based screening by market cap
        
        ATR Acceptable Ranges:
        - Large Cap: 1-5% daily ATR
        - Mid Cap: 2-7% daily ATR
        - Small Cap: 3-10% daily ATR
        """
        
        try:
            # Simulate ATR calculation
            if market_cap_category == 'LARGE':
                atr_pct = np.random.uniform(0.01, 0.06)  # 1-6%
                min_atr, max_atr = 0.01, 0.05
            elif market_cap_category == 'MID':
                atr_pct = np.random.uniform(0.015, 0.08)  # 1.5-8%
                min_atr, max_atr = 0.02, 0.07
            else:  # SMALL
                atr_pct = np.random.uniform(0.02, 0.12)  # 2-12%
                min_atr, max_atr = 0.03, 0.10
            
            passes = min_atr <= atr_pct <= max_atr
            
            return {
                'passes': passes,
                'atr_pct': atr_pct * 100,
                'min_atr_pct': min_atr * 100,
                'max_atr_pct': max_atr * 100,
                'market_cap_category': market_cap_category
            }
            
        except Exception as e:
            logger.error(f"Volatility screening failed for {symbol}: {e}")
            return {'passes': False, 'error': str(e)}
    
    async def news_sentiment_screen(self, symbol: str) -> Dict[str, Union[bool, str, List]]:
        """
        News and sentiment quality check
        """
        
        try:
            # This would integrate with news APIs
            # Placeholder implementation
            
            # Simulate news sentiment
            sentiment_score = np.random.uniform(0.3, 0.8)  # 0.3 to 0.8
            
            red_flags = []
            
            # Simulate potential red flags
            if np.random.random() < 0.1:  # 10% chance of red flags
                possible_flags = [
                    'Management changes',
                    'Regulatory investigation',
                    'Earnings restatement',
                    'Major customer loss',
                    'Governance issues'
                ]
                red_flags = np.random.choice(possible_flags, 
                                           size=np.random.randint(1, 3), 
                                           replace=False).tolist()
            
            passes = sentiment_score >= 0.4 and len(red_flags) == 0
            
            return {
                'passes': passes,
                'sentiment_score': sentiment_score,
                'red_flags': red_flags,
                'sentiment_trend': 'POSITIVE' if sentiment_score > 0.6 else 'NEUTRAL' if sentiment_score > 0.4 else 'NEGATIVE'
            }
            
        except Exception as e:
            logger.error(f"News sentiment screening failed for {symbol}: {e}")
            return {'passes': False, 'error': str(e)}
    
    async def corporate_action_screen(self, symbol: str) -> Dict[str, Union[bool, str, List]]:
        """
        Corporate action and event screening
        """
        
        try:
            # This would check for pending corporate actions
            # Placeholder implementation
            
            # Simulate corporate actions
            pending_actions = []
            
            if np.random.random() < 0.05:  # 5% chance of pending actions
                possible_actions = [
                    'Dividend ex-date in 3 days',
                    'Stock split announced',
                    'Rights issue',
                    'Bonus issue',
                    'Merger announcement'
                ]
                pending_actions = np.random.choice(possible_actions, 
                                                 size=1, 
                                                 replace=False).tolist()
            
            passes = len(pending_actions) == 0
            
            return {
                'passes': passes,
                'pending_actions': pending_actions,
                'exclusion_reason': pending_actions[0] if pending_actions else None
            }
            
        except Exception as e:
            logger.error(f"Corporate action screening failed for {symbol}: {e}")
            return {'passes': False, 'error': str(e)}


class AdvancedTradeFilter:
    """
    Master trade filtering system
    
    Combines all scoring components to identify top trading opportunities
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.universe_manager = UniverseManager()
        self.quality_filters = QualityFilters()
        
        # Initialize analyzers
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.quantitative_engine = QuantitativeEngine()
        self.macro_analyzer = MacroSentimentAnalyzer()
        
        # Scoring weights
        self.score_weights = {
            'fundamental': self.config.FUNDAMENTAL_WEIGHT,
            'technical': self.config.TECHNICAL_WEIGHT,
            'quantitative': self.config.QUANTITATIVE_WEIGHT,
            'macro': self.config.MACRO_WEIGHT
        }
    
    async def filter_top_trades(self) -> List[TradeOpportunity]:
        """
        Core filtering algorithm returning top trading opportunities
        
        Process:
        1. Get tradeable universe (~300 stocks)
        2. Score all stocks using composite scoring
        3. Apply minimum score threshold (0.70)
        4. Apply quality screens
        5. Apply diversification limits
        6. Return top 10-15 opportunities
        
        Returns:
            List of TradeOpportunity objects
        """
        
        start_time = datetime.now()
        logger.info("Starting trade filtering process...")
        
        try:
            # Step 1: Get tradeable universe
            universe = self.universe_manager.get_tradeable_universe()
            logger.info(f"Tradeable universe: {len(universe)} stocks")
            
            if not universe:
                logger.warning("Empty tradeable universe")
                return []
            
            # Step 2: Score all stocks in parallel
            scored_opportunities = await self._score_universe_parallel(universe)
            logger.info(f"Scored {len(scored_opportunities)} opportunities")
            
            # Step 3: Apply minimum score threshold
            threshold_filtered = [
                opp for opp in scored_opportunities 
                if opp['composite_score'] >= self.config.COMPOSITE_SCORE_THRESHOLD
            ]
            logger.info(f"After threshold filter: {len(threshold_filtered)} opportunities")
            
            # Step 4: Apply quality screens
            quality_filtered = await self._apply_quality_filters(threshold_filtered)
            logger.info(f"After quality filter: {len(quality_filtered)} opportunities")
            
            # Step 5: Apply diversification filters
            diversified = await self._apply_diversification_filters(quality_filtered)
            logger.info(f"After diversification filter: {len(diversified)} opportunities")
            
            # Step 6: Create TradeOpportunity objects and sort
            trade_opportunities = await self._create_trade_opportunities(diversified)
            
            # Sort by composite score
            trade_opportunities.sort(key=lambda x: x.composite_score, reverse=True)
            
            # Return top opportunities
            final_selection = trade_opportunities[:self.config.MAX_POSITIONS]
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Trade filtering completed in {duration:.2f}s")
            logger.info(f"Final selection: {len(final_selection)} opportunities")
            
            # Log top opportunities
            for i, opp in enumerate(final_selection[:5], 1):
                logger.info(f"  {i}. {opp.symbol} ({opp.sector}) - Score: {opp.composite_score:.3f}")
            
            return final_selection
            
        except Exception as e:
            logger.error(f"Trade filtering failed: {e}")
            return []
    
    async def _score_universe_parallel(self, universe: List[str]) -> List[Dict]:
        """
        Score all stocks in parallel for efficiency
        """
        
        logger.info("Starting parallel scoring of universe...")
        
        # Create batches for parallel processing
        batch_size = 20  # Process 20 stocks at a time
        batches = [universe[i:i + batch_size] for i in range(0, len(universe), batch_size)]
        
        all_scored = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} ({len(batch)} stocks)")
            
            # Process batch in parallel
            tasks = [self._score_single_stock(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_results = [
                result for result in batch_results 
                if result is not None and not isinstance(result, Exception)
            ]
            
            all_scored.extend(valid_results)
            
            # Small delay between batches to avoid overwhelming APIs
            if i < len(batches) - 1:
                await asyncio.sleep(0.5)
        
        logger.info(f"Parallel scoring completed: {len(all_scored)} valid scores")
        return all_scored
    
    async def _score_single_stock(self, symbol: str) -> Optional[Dict]:
        """
        Score a single stock using all components
        """
        
        try:
            # Run all scoring components in parallel
            results = await asyncio.gather(
                self.fundamental_analyzer.calculate_fundamental_score(symbol),
                self.technical_analyzer.calculate_technical_score(symbol),
                self.quantitative_engine.calculate_quantitative_score(symbol),
                self.macro_analyzer.calculate_macro_score(symbol),
                return_exceptions=True
            )
            
            # Extract results or use defaults for exceptions
            fund_result = results[0] if not isinstance(results[0], Exception) else ScoringResult(0.0, {}, 0.0)
            tech_result = results[1] if not isinstance(results[1], Exception) else ScoringResult(0.0, {}, 0.0)
            quant_result = results[2] if not isinstance(results[2], Exception) else ScoringResult(0.0, {}, 0.0)
            macro_result = results[3] if not isinstance(results[3], Exception) else ScoringResult(0.5, {}, 50.0)
            
            # Calculate weighted composite score
            composite_score = (
                fund_result.score * self.score_weights['fundamental'] +
                tech_result.score * self.score_weights['technical'] +
                quant_result.score * self.score_weights['quantitative'] +
                macro_result.score * self.score_weights['macro']
            )
            
            # Calculate conviction level (based on score consistency)
            scores = [fund_result.score, tech_result.score, quant_result.score, macro_result.score]
            score_variance = np.var(scores)
            conviction_level = max(0.5, min(1.0, composite_score * (1 - score_variance)))
            
            return {
                'symbol': symbol,
                'composite_score': round(composite_score, 3),
                'conviction_level': round(conviction_level, 3),
                'individual_scores': {
                    'fundamental': fund_result,
                    'technical': tech_result,
                    'quantitative': quant_result,
                    'macro': macro_result
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.debug(f"Scoring failed for {symbol}: {e}")
            return None
    
    async def _apply_quality_filters(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Apply quality screening filters
        """
        
        quality_filtered = []
        
        for opp in opportunities:
            symbol = opp['symbol']
            stock_info = self.universe_manager.get_stock_info(symbol)
            
            if not stock_info:
                continue
            
            # Apply all quality screens in parallel
            quality_results = await asyncio.gather(
                self.quality_filters.liquidity_screen(symbol),
                self.quality_filters.volatility_screen(symbol, stock_info['market_cap_category']),
                self.quality_filters.news_sentiment_screen(symbol),
                self.quality_filters.corporate_action_screen(symbol),
                return_exceptions=True
            )
            
            # Check if all screens pass
            all_pass = True
            quality_breakdown = {}
            
            for i, (screen_name, result) in enumerate(zip(
                ['liquidity', 'volatility', 'sentiment', 'corporate_action'], 
                quality_results
            )):
                if isinstance(result, Exception):
                    logger.warning(f"Quality screen {screen_name} failed for {symbol}: {result}")
                    all_pass = False
                    quality_breakdown[screen_name] = {'passes': False, 'error': str(result)}
                else:
                    quality_breakdown[screen_name] = result
                    if not result.get('passes', False):
                        all_pass = False
            
            if all_pass:
                opp['quality_screens'] = quality_breakdown
                quality_filtered.append(opp)
            else:
                logger.debug(f"Quality filter failed for {symbol}")
        
        return quality_filtered
    
    async def _apply_diversification_filters(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Apply sector and market cap diversification rules
        
        Rules:
        - Maximum 30% allocation to any single sector
        - Maximum 3 stocks from same sector
        - Balanced allocation across market caps
        """
        
        # Sort by composite score first
        opportunities.sort(key=lambda x: x['composite_score'], reverse=True)
        
        diversified = []
        sector_count = {}
        sector_allocation = {}
        
        for opp in opportunities:
            symbol = opp['symbol']
            stock_info = self.universe_manager.get_stock_info(symbol)
            
            if not stock_info:
                continue
            
            sector = stock_info['sector']
            
            # Check sector limits
            current_sector_count = sector_count.get(sector, 0)
            current_sector_alloc = sector_allocation.get(sector, 0.0)
            
            # Calculate expected position size (simplified)
            expected_position_pct = self.config.MAX_POSITION_SIZE  # Use max for conservative estimate
            
            if (current_sector_count < self.config.MAX_STOCKS_PER_SECTOR and
                current_sector_alloc + expected_position_pct <= self.config.MAX_SECTOR_ALLOCATION):
                
                diversified.append(opp)
                sector_count[sector] = current_sector_count + 1
                sector_allocation[sector] = current_sector_alloc + expected_position_pct
                
                logger.debug(f"Added {symbol} ({sector}) - Sector count: {sector_count[sector]}")
            else:
                logger.debug(f"Diversification limit reached for {symbol} ({sector})")
        
        return diversified
    
    async def _create_trade_opportunities(self, scored_opportunities: List[Dict]) -> List[TradeOpportunity]:
        """
        Create TradeOpportunity objects with complete information
        """
        
        trade_opportunities = []
        
        for opp in scored_opportunities:
            symbol = opp['symbol']
            stock_info = self.universe_manager.get_stock_info(symbol)
            
            if not stock_info:
                continue
            
            try:
                # Generate investment rationale
                rationale = self._generate_investment_rationale(opp)
                
                # Calculate additional scores
                quality_score = self._calculate_quality_score(opp)
                
                # Create TradeOpportunity object
                trade_opp = TradeOpportunity(
                    symbol=symbol,
                    company_name=stock_info['company_name'],
                    sector=stock_info['sector'],
                    market_cap_category=stock_info['market_cap_category'],
                    
                    composite_score=opp['composite_score'],
                    conviction_level=opp['conviction_level'],
                    meets_threshold=opp['composite_score'] >= self.config.COMPOSITE_SCORE_THRESHOLD,
                    
                    fundamental_score=opp['individual_scores']['fundamental'],
                    technical_score=opp['individual_scores']['technical'],
                    quantitative_score=opp['individual_scores']['quantitative'],
                    macro_score=opp['individual_scores']['macro'],
                    
                    current_price=0.0,  # Would be fetched from real-time data
                    market_cap=stock_info.get('market_cap'),
                    avg_volume=stock_info.get('avg_volume'),
                    atr_pct=opp.get('quality_screens', {}).get('volatility', {}).get('atr_pct', 3.0),
                    
                    investment_rationale=rationale['summary'],
                    key_strengths=rationale['strengths'],
                    key_risks=rationale['risks'],
                    
                    quality_score=quality_score,
                    liquidity_score=0.8,  # Placeholder
                    volatility_score=0.7   # Placeholder
                )
                
                trade_opportunities.append(trade_opp)
                
            except Exception as e:
                logger.error(f"Failed to create TradeOpportunity for {symbol}: {e}")
                continue
        
        return trade_opportunities
    
    def _generate_investment_rationale(self, opportunity: Dict) -> Dict[str, Union[str, List[str]]]:
        """
        Generate comprehensive investment rationale
        """
        
        symbol = opportunity['symbol']
        scores = opportunity['individual_scores']
        
        strengths = []
        risks = []
        
        # Analyze fundamental strengths/risks
        fund_breakdown = scores['fundamental'].breakdown
        if fund_breakdown.get('roe_score', 0) >= 10:
            strengths.append(f"Strong ROE of {fund_breakdown.get('roe_value', 0)*100:.1f}%")
        if fund_breakdown.get('debt_equity_score', 0) >= 10:
            strengths.append("Conservative debt levels")
        
        # Analyze technical strengths/risks
        tech_breakdown = scores['technical'].breakdown
        if tech_breakdown.get('momentum_total', 0) >= 30:
            strengths.append("Strong technical momentum")
        if tech_breakdown.get('rsi_value', 50) > 70:
            risks.append("Overbought RSI levels")
        
        # Analyze quantitative factors
        quant_breakdown = scores['quantitative'].breakdown
        if quant_breakdown.get('momentum_factors_total', 0) >= 30:
            strengths.append("Superior price momentum")
        
        # Analyze macro factors
        macro_breakdown = scores['macro'].breakdown
        if macro_breakdown.get('sector_rotation_total', 0) >= 25:
            strengths.append("Favorable sector rotation")
        
        # Generate summary
        summary = f"High-conviction opportunity in {symbol} with composite score of {opportunity['composite_score']:.3f}. "
        
        if len(strengths) >= 3:
            summary += "Multiple positive factors align including strong fundamentals and technical momentum."
        elif len(strengths) >= 2:
            summary += "Good fundamental and technical setup with favorable macro environment."
        else:
            summary += "Selective opportunity based on quantitative factors."
        
        return {
            'summary': summary,
            'strengths': strengths[:5],  # Top 5 strengths
            'risks': risks[:3]  # Top 3 risks
        }
    
    def _calculate_quality_score(self, opportunity: Dict) -> float:
        """
        Calculate overall quality score (0-1)
        """
        
        quality_screens = opportunity.get('quality_screens', {})
        
        # Count passed screens
        passed_screens = sum(1 for screen in quality_screens.values() if screen.get('passes', False))
        total_screens = len(quality_screens)
        
        if total_screens == 0:
            return 0.5  # Default neutral score
        
        return passed_screens / total_screens
