"""
API Configuration and Connectors
Zerodha Kite Connect and Sonar API integration
"""

import os
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from kiteconnect import KiteConnect
import redis
import json

logger = logging.getLogger(__name__)

@dataclass
class APICredentials:
    """API credentials storage"""
    
    # Zerodha credentials
    zerodha_api_key: str
    zerodha_access_token: str
    zerodha_api_secret: str
    
    # Sonar credentials
    sonar_api_key: str
    sonar_base_url: str = "https://api.sonar.com"
    
    # Redis for caching
    redis_url: str = "redis://localhost:6379/0"


class ZerodhaConnector:
    """
    Zerodha Kite Connect API wrapper with rate limiting and error handling
    """
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        
        # Rate limiting: 3 requests per second
        self.request_queue = asyncio.Queue(maxsize=3)
        self.last_request_time = datetime.now()
        
        # Cache for reducing API calls
        self.redis_client = None
        self.cache_enabled = True
        
    async def initialize_cache(self, redis_url: str):
        """Initialize Redis cache"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await asyncio.to_thread(self.redis_client.ping)
            logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}")
            self.cache_enabled = False
    
    async def rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_request_time).total_seconds()
        
        if time_diff < 0.33:  # 3 requests per second = 0.33 seconds between requests
            await asyncio.sleep(0.33 - time_diff)
        
        self.last_request_time = datetime.now()
    
    async def get_historical_data(self, 
                                symbol: str, 
                                from_date: datetime, 
                                to_date: datetime,
                                interval: str = "day") -> List[Dict]:
        """
        Get historical OHLCV data with caching
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            from_date: Start date
            to_date: End date
            interval: Data interval (minute, day, etc.)
        """
        
        # Check cache first
        cache_key = f"historical:{symbol}:{interval}:{from_date.date()}:{to_date.date()}"
        
        if self.cache_enabled and self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    logger.debug(f"Cache hit for {symbol} historical data")
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Make API request with rate limiting
        await self.rate_limit()
        
        try:
            # Convert symbol to instrument token if needed
            instrument_token = await self.get_instrument_token(symbol)
            
            historical_data = await asyncio.to_thread(
                self.kite.historical_data,
                instrument_token,
                from_date,
                to_date,
                interval
            )
            
            # Cache the result for 24 hours
            if self.cache_enabled and self.redis_client:
                try:
                    self.redis_client.setex(
                        cache_key, 
                        86400,  # 24 hours
                        json.dumps(historical_data, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")
            
            logger.debug(f"Retrieved {len(historical_data)} records for {symbol}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Historical data fetch failed for {symbol}: {e}")
            raise
    
    async def get_quote(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get real-time quotes for multiple symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict with symbol as key and quote data as value
        """
        
        await self.rate_limit()
        
        try:
            # Convert symbols to instrument tokens
            instrument_tokens = []
            for symbol in symbols:
                token = await self.get_instrument_token(symbol)
                instrument_tokens.append(f"NSE:{symbol}")
            
            quotes = await asyncio.to_thread(self.kite.quote, instrument_tokens)
            
            # Transform response to use symbols as keys
            transformed_quotes = {}
            for key, value in quotes.items():
                symbol = key.split(':')[1] if ':' in key else key
                transformed_quotes[symbol] = value
            
            return transformed_quotes
            
        except Exception as e:
            logger.error(f"Quote fetch failed for {symbols}: {e}")
            raise
    
    async def get_instrument_token(self, symbol: str) -> int:
        """
        Get instrument token for symbol with caching
        """
        
        cache_key = f"instrument_token:{symbol}"
        
        if self.cache_enabled and self.redis_client:
            try:
                cached_token = self.redis_client.get(cache_key)
                if cached_token:
                    return int(cached_token)
            except Exception as e:
                logger.warning(f"Cache read error for instrument token: {e}")
        
        # Fetch instruments list (cached for 24 hours)
        instruments_cache_key = "instruments_list"
        instruments = None
        
        if self.cache_enabled and self.redis_client:
            try:
                cached_instruments = self.redis_client.get(instruments_cache_key)
                if cached_instruments:
                    instruments = json.loads(cached_instruments)
            except Exception as e:
                logger.warning(f"Instruments cache read error: {e}")
        
        if not instruments:
            await self.rate_limit()
            instruments = await asyncio.to_thread(self.kite.instruments, "NSE")
            
            # Cache instruments list
            if self.cache_enabled and self.redis_client:
                try:
                    self.redis_client.setex(
                        instruments_cache_key,
                        86400,  # 24 hours
                        json.dumps(instruments, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Instruments cache write error: {e}")
        
        # Find instrument token for symbol
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                token = instrument['instrument_token']
                
                # Cache the token
                if self.cache_enabled and self.redis_client:
                    try:
                        self.redis_client.setex(cache_key, 86400, str(token))
                    except Exception as e:
                        logger.warning(f"Token cache write error: {e}")
                
                return token
        
        raise ValueError(f"Instrument token not found for symbol: {symbol}")
    
    async def place_order(self, 
                         symbol: str,
                         quantity: int,
                         side: str,
                         order_type: str = "MARKET",
                         product: str = "CNC") -> Dict:
        """
        Place trading order with comprehensive error handling
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            side: BUY or SELL
            order_type: MARKET, LIMIT, etc.
            product: CNC (delivery), MIS (intraday)
        """
        
        await self.rate_limit()
        
        try:
            # Validate order parameters
            if side not in ['BUY', 'SELL']:
                raise ValueError(f"Invalid side: {side}")
            
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}")
            
            # Place order
            order_response = await asyncio.to_thread(
                self.kite.place_order,
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=side,
                quantity=quantity,
                product=product,
                order_type=order_type
            )
            
            logger.info(f"Order placed: {symbol} {side} {quantity} - Order ID: {order_response['order_id']}")
            return order_response
            
        except Exception as e:
            logger.error(f"Order placement failed: {symbol} {side} {quantity} - Error: {e}")
            raise
    
    async def get_orders(self) -> List[Dict]:
        """Get all orders for the day"""
        await self.rate_limit()
        
        try:
            orders = await asyncio.to_thread(self.kite.orders)
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            raise
    
    async def get_positions(self) -> Dict:
        """Get current positions"""
        await self.rate_limit()
        
        try:
            positions = await asyncio.to_thread(self.kite.positions)
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise


class SonarConnector:
    """
    Sonar API connector for sentiment analysis and macro data
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.sonar.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
        # Rate limiting
        self.request_count = 0
        self.request_window_start = datetime.now()
        self.max_requests_per_minute = 60
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = datetime.now()
        
        # Reset counter every minute
        if (current_time - self.request_window_start).total_seconds() >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we've exceeded rate limit
        if self.request_count >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_window_start).total_seconds()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.request_window_start = datetime.now()
        
        self.request_count += 1
    
    async def get_market_sentiment(self) -> Dict:
        """
        Get overall market sentiment indicators
        
        Returns:
            Dict with sentiment metrics like Fear & Greed Index, VIX, etc.
        """
        
        await self.rate_limit_check()
        
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/market/sentiment") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'fear_greed_index': data.get('fear_greed_index', 50),
                        'vix_level': data.get('vix_level', 20),
                        'market_breadth': data.get('market_breadth', 0.5),
                        'sentiment_score': data.get('sentiment_score', 0.5),
                        'timestamp': datetime.now()
                    }
                else:
                    logger.warning(f"Sentiment API returned status {response.status}")
                    return self._get_default_sentiment()
                    
        except Exception as e:
            logger.error(f"Failed to fetch market sentiment: {e}")
            return self._get_default_sentiment()
    
    async def get_stock_sentiment(self, symbol: str) -> Dict:
        """
        Get sentiment analysis for specific stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with stock-specific sentiment metrics
        """
        
        await self.rate_limit_check()
        
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/stocks/{symbol}/sentiment") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'symbol': symbol,
                        'news_sentiment': data.get('news_sentiment', 0.5),
                        'social_sentiment': data.get('social_sentiment', 0.5),
                        'analyst_sentiment': data.get('analyst_sentiment', 0.5),
                        'composite_sentiment': data.get('composite_sentiment', 0.5),
                        'sentiment_trend': data.get('sentiment_trend', 'NEUTRAL'),
                        'timestamp': datetime.now()
                    }
                else:
                    logger.warning(f"Stock sentiment API returned status {response.status} for {symbol}")
                    return self._get_default_stock_sentiment(symbol)
                    
        except Exception as e:
            logger.error(f"Failed to fetch stock sentiment for {symbol}: {e}")
            return self._get_default_stock_sentiment(symbol)
    
    async def get_sector_flows(self) -> Dict:
        """
        Get FII/DII sector-wise flow data
        """
        
        await self.rate_limit_check()
        
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.base_url}/flows/sectors") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'fii_flows': data.get('fii_flows', {}),
                        'dii_flows': data.get('dii_flows', {}),
                        'sector_rotation': data.get('sector_rotation', {}),
                        'timestamp': datetime.now()
                    }
                else:
                    logger.warning(f"Sector flows API returned status {response.status}")
                    return self._get_default_sector_flows()
                    
        except Exception as e:
            logger.error(f"Failed to fetch sector flows: {e}")
            return self._get_default_sector_flows()
    
    def _get_default_sentiment(self) -> Dict:
        """Default sentiment when API is unavailable"""
        return {
            'fear_greed_index': 50,
            'vix_level': 20,
            'market_breadth': 0.5,
            'sentiment_score': 0.5,
            'timestamp': datetime.now(),
            'source': 'DEFAULT'
        }
    
    def _get_default_stock_sentiment(self, symbol: str) -> Dict:
        """Default stock sentiment when API is unavailable"""
        return {
            'symbol': symbol,
            'news_sentiment': 0.5,
            'social_sentiment': 0.5,
            'analyst_sentiment': 0.5,
            'composite_sentiment': 0.5,
            'sentiment_trend': 'NEUTRAL',
            'timestamp': datetime.now(),
            'source': 'DEFAULT'
        }
    
    def _get_default_sector_flows(self) -> Dict:
        """Default sector flows when API is unavailable"""
        return {
            'fii_flows': {},
            'dii_flows': {},
            'sector_rotation': {},
            'timestamp': datetime.now(),
            'source': 'DEFAULT'
        }


class APIManager:
    """
    Centralized API management for all external data sources
    """
    
    def __init__(self, credentials: APICredentials):
        self.credentials = credentials
        self.zerodha = None
        self.sonar = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all API connections"""
        
        try:
            # Initialize Zerodha connector
            self.zerodha = ZerodhaConnector(
                self.credentials.zerodha_api_key,
                self.credentials.zerodha_access_token
            )
            await self.zerodha.initialize_cache(self.credentials.redis_url)
            
            # Initialize Sonar connector
            self.sonar = SonarConnector(
                self.credentials.sonar_api_key,
                self.credentials.sonar_base_url
            )
            await self.sonar.initialize()
            
            self.initialized = True
            logger.info("✅ All API connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ API initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup API connections"""
        if self.sonar:
            await self.sonar.close()
        
        logger.info("API connections cleaned up")
    
    def get_zerodha(self) -> ZerodhaConnector:
        """Get Zerodha connector instance"""
        if not self.initialized:
            raise RuntimeError("API Manager not initialized")
        return self.zerodha
    
    def get_sonar(self) -> SonarConnector:
        """Get Sonar connector instance"""
        if not self.initialized:
            raise RuntimeError("API Manager not initialized")
        return self.sonar


# Global API manager instance
_api_manager = None

async def get_api_manager() -> APIManager:
    """Get singleton API manager instance"""
    global _api_manager
    
    if _api_manager is None:
        from .trading_config import get_trading_config
        config = get_trading_config()
        
        credentials = APICredentials(
            zerodha_api_key=config.ZERODHA_API_KEY,
            zerodha_access_token=config.ZERODHA_ACCESS_TOKEN,
            zerodha_api_secret=config.ZERODHA_API_SECRET,
            sonar_api_key=config.SONAR_API_KEY,
            sonar_base_url=config.SONAR_BASE_URL,
            redis_url=config.REDIS_URL
        )
        
        _api_manager = APIManager(credentials)
        await _api_manager.initialize()
    
    return _api_manager
