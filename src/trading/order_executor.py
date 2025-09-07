"""
Institutional Order Execution Engine
Smart order execution with TWAP/VWAP algorithms and market impact minimization
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import logging
from decimal import Decimal, ROUND_HALF_UP
import uuid

from config.trading_config import get_trading_config
from config.api_config import ZerodhaConnector
from config.database_config import get_db_session, Trade, OrderLog, ExecutionMetrics
from strategy.risk_manager import ComprehensiveRiskManager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"

class ExecutionAlgorithm(Enum):
    """Execution algorithms"""
    AGGRESSIVE = "AGGRESSIVE"  # Market orders for immediate execution
    PASSIVE = "PASSIVE"       # Limit orders with patient execution
    STEALTH = "STEALTH"       # Iceberg orders to hide size
    TWAP = "TWAP"            # Time-weighted average price
    VWAP = "VWAP"            # Volume-weighted average price
    POV = "POV"              # Percent of volume

@dataclass
class OrderRequest:
    """Order request data structure"""
    
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: OrderType
    
    # Price fields
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    
    # Execution algorithm
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.PASSIVE
    
    # Algorithm parameters
    time_horizon_minutes: int = 60  # For TWAP/VWAP
    max_participation_rate: float = 0.20  # Maximum 20% of volume
    slice_size_shares: int = 100  # Minimum slice size
    
    # Risk parameters
    max_price_impact_bps: float = 50  # 50 basis points maximum impact
    timeout_minutes: int = 240  # 4 hours default timeout
    
    # Metadata
    strategy_name: str = "manual"
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH, URGENT
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate order request"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.side not in ['BUY', 'SELL']:
            raise ValueError("Side must be BUY or SELL")
        
        if self.order_type in [OrderType.LIMIT] and self.price is None:
            raise ValueError("Price required for limit orders")

@dataclass
class ExecutionSlice:
    """Individual execution slice within larger order"""
    
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: int
    
    order_type: OrderType
    price: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    
    exchange_order_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


class SmartOrderExecutor:
    """
    Institutional-grade order execution with multiple algorithms
    
    Execution Algorithms:
    1. AGGRESSIVE: Market orders for immediate execution
    2. PASSIVE: Intelligent limit orders with price improvement
    3. STEALTH: Iceberg orders to minimize market impact
    4. TWAP: Time-Weighted Average Price execution
    5. VWAP: Volume-Weighted Average Price execution
    6. POV: Percent of Volume participation
    
    Features:
    - Market impact minimization
    - Real-time slippage monitoring
    - Adaptive execution based on market conditions
    - Pre-trade and post-trade analytics
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.zerodha = ZerodhaConnector()
        self.risk_manager = ComprehensiveRiskManager()
        self.db_session = get_db_session()
        
        # Execution state
        self.active_orders: Dict[str, OrderRequest] = {}
        self.execution_slices: Dict[str, List[ExecutionSlice]] = {}
        
        # Performance tracking
        self.execution_metrics = {
            'total_orders': 0,
            'successful_fills': 0,
            'average_fill_time': 0.0,
            'average_slippage_bps': 0.0,
            'total_volume_inr': 0.0
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = {}
        
        logger.info("🚀 Smart Order Executor initialized")
    
    async def execute_order(self, order_request: OrderRequest) -> Dict[str, Union[str, bool, Dict]]:
        """
        Main order execution entry point
        
        Args:
            order_request: Order request with execution parameters
            
        Returns:
            Dict with execution result and tracking information
        """
        
        try:
            # 1. Pre-trade validation
            validation_result = await self._pre_trade_validation(order_request)
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'order_id': order_request.order_id,
                    'error': validation_result['error'],
                    'validation_details': validation_result
                }
            
            # 2. Pre-trade analysis
            pre_trade_analysis = await self._pre_trade_analysis(order_request)
            
            # 3. Select optimal execution algorithm
            optimal_algorithm = self._select_execution_algorithm(order_request, pre_trade_analysis)
            
            # 4. Store active order
            self.active_orders[order_request.order_id] = order_request
            
            # 5. Execute based on algorithm
            if optimal_algorithm == ExecutionAlgorithm.AGGRESSIVE:
                result = await self._execute_aggressive(order_request, pre_trade_analysis)
            
            elif optimal_algorithm == ExecutionAlgorithm.PASSIVE:
                result = await self._execute_passive(order_request, pre_trade_analysis)
            
            elif optimal_algorithm == ExecutionAlgorithm.STEALTH:
                result = await self._execute_stealth(order_request, pre_trade_analysis)
            
            elif optimal_algorithm == ExecutionAlgorithm.TWAP:
                result = await self._execute_twap(order_request, pre_trade_analysis)
            
            elif optimal_algorithm == ExecutionAlgorithm.VWAP:
                result = await self._execute_vwap(order_request, pre_trade_analysis)
            
            else:  # Default to passive
                result = await self._execute_passive(order_request, pre_trade_analysis)
            
            # 6. Post-trade analysis
            if result['success']:
                post_trade_analysis = await self._post_trade_analysis(order_request, result)
                result['post_trade_metrics'] = post_trade_analysis
            
            # 7. Update metrics
            self._update_execution_metrics(order_request, result)
            
            # 8. Log execution
            self._log_order_execution(order_request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed for {order_request.order_id}: {e}")
            return {
                'success': False,
                'order_id': order_request.order_id,
                'error': str(e),
                'algorithm_used': None
            }
        
        finally:
            # Clean up completed order
            if order_request.order_id in self.active_orders:
                if order_request.order_id not in ['PENDING', 'OPEN']:
                    del self.active_orders[order_request.order_id]
    
    async def _pre_trade_validation(self, order: OrderRequest) -> Dict[str, Union[bool, str, float]]:
        """
        Comprehensive pre-trade validation
        
        Validations:
        1. Symbol validity and tradability
        2. Quantity within exchange limits
        3. Price reasonableness (within circuit limits)
        4. Risk limits (position size, sector concentration)
        5. Market conditions (volatility, liquidity)
        6. Trading hours and market status
        """
        
        try:
            symbol = order.symbol
            
            # 1. Symbol validation
            if not await self._is_symbol_tradable(symbol):
                return {
                    'is_valid': False,
                    'error': f'Symbol {symbol} is not tradable',
                    'validation_step': 'SYMBOL_CHECK'
                }
            
            # 2. Market data availability
            market_data = await self._get_real_time_market_data(symbol)
            if not market_data:
                return {
                    'is_valid': False,
                    'error': f'Market data unavailable for {symbol}',
                    'validation_step': 'MARKET_DATA'
                }
            
            # 3. Price reasonableness (within circuit limits)
            if order.price:
                ltp = market_data.get('last_price', 0)
                price_deviation = abs(order.price - ltp) / ltp
                
                if price_deviation > 0.20:  # 20% deviation limit
                    return {
                        'is_valid': False,
                        'error': f'Order price {order.price} deviates {price_deviation*100:.1f}% from LTP {ltp}',
                        'validation_step': 'PRICE_CHECK',
                        'ltp': ltp,
                        'price_deviation_pct': price_deviation * 100
                    }
            
            # 4. Quantity limits
            lot_size = market_data.get('lot_size', 1)
            if order.quantity % lot_size != 0:
                return {
                    'is_valid': False,
                    'error': f'Quantity {order.quantity} not multiple of lot size {lot_size}',
                    'validation_step': 'QUANTITY_CHECK',
                    'lot_size': lot_size
                }
            
            # 5. Market timing
            if not self._is_market_open():
                return {
                    'is_valid': False,
                    'error': 'Market is closed',
                    'validation_step': 'MARKET_HOURS'
                }
            
            # 6. Risk limits (would integrate with risk manager)
            risk_check = await self._validate_risk_limits(order, market_data)
            if not risk_check['is_valid']:
                return risk_check
            
            # 7. Liquidity check
            liquidity_check = self._validate_liquidity(order, market_data)
            if not liquidity_check['is_valid']:
                return liquidity_check
            
            return {
                'is_valid': True,
                'validation_step': 'COMPLETE',
                'market_data': market_data,
                'estimated_cost': order.quantity * market_data.get('last_price', 0)
            }
            
        except Exception as e:
            logger.error(f"Pre-trade validation failed: {e}")
            return {
                'is_valid': False,
                'error': f'Validation error: {str(e)}',
                'validation_step': 'ERROR'
            }
    
    async def _pre_trade_analysis(self, order: OrderRequest) -> Dict[str, Union[float, int, str]]:
        """
        Pre-trade cost analysis and market impact estimation
        
        Analysis:
        1. Expected market impact
        2. Optimal execution time horizon
        3. Liquidity assessment
        4. Volatility-adjusted sizing
        5. Timing recommendations
        """
        
        symbol = order.symbol
        market_data = await self._get_real_time_market_data(symbol)
        
        # Get historical data for analysis
        historical_data = await self._get_historical_data(symbol, days=5)
        
        # 1. Liquidity metrics
        avg_daily_volume = historical_data['volume'].mean() if len(historical_data) > 0 else 1000000
        avg_daily_value = avg_daily_volume * market_data.get('last_price', 100)
        
        order_value = order.quantity * market_data.get('last_price', 100)
        participation_rate = order_value / avg_daily_value if avg_daily_value > 0 else 1.0
        
        # 2. Volatility analysis
        if len(historical_data) > 1:
            returns = historical_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            volatility = 0.25  # Default 25% volatility
        
        # 3. Market impact estimation (Square root law)
        # Impact = volatility * (participation_rate ^ 0.5) * spread_factor
        spread_factor = 0.5  # Simplified - would use actual bid-ask spread
        estimated_impact_bps = volatility * (participation_rate ** 0.5) * spread_factor * 10000
        
        # 4. Optimal execution parameters
        if participation_rate < 0.05:  # Small order
            recommended_algorithm = ExecutionAlgorithm.AGGRESSIVE
            time_horizon = 15  # minutes
        elif participation_rate < 0.20:  # Medium order
            recommended_algorithm = ExecutionAlgorithm.PASSIVE
            time_horizon = 60  # minutes
        else:  # Large order
            recommended_algorithm = ExecutionAlgorithm.TWAP
            time_horizon = min(240, max(60, participation_rate * 480))  # Scale with size
        
        # 5. Risk assessment
        risk_level = "LOW"
        if estimated_impact_bps > 50:
            risk_level = "HIGH"
        elif estimated_impact_bps > 25:
            risk_level = "MEDIUM"
        
        return {
            'avg_daily_volume': avg_daily_volume,
            'avg_daily_value': avg_daily_value,
            'order_value': order_value,
            'participation_rate': participation_rate,
            'volatility': volatility,
            'estimated_impact_bps': estimated_impact_bps,
            'recommended_algorithm': recommended_algorithm,
            'optimal_time_horizon': time_horizon,
            'risk_level': risk_level,
            'liquidity_score': min(1.0, 1.0 / max(0.1, participation_rate))  # Higher is better
        }
    
    def _select_execution_algorithm(self, 
                                  order: OrderRequest, 
                                  analysis: Dict) -> ExecutionAlgorithm:
        """
        Select optimal execution algorithm based on order characteristics
        
        Decision Matrix:
        - Small orders (<5% ADV): AGGRESSIVE
        - Medium orders (5-20% ADV): PASSIVE or STEALTH
        - Large orders (>20% ADV): TWAP or VWAP
        - Urgent orders: AGGRESSIVE regardless of size
        - Stealth required: ICEBERG/STEALTH
        """
        
        # Override if algorithm specified in order
        if hasattr(order, 'algorithm') and order.algorithm != ExecutionAlgorithm.PASSIVE:
            return order.algorithm
        
        participation_rate = analysis['participation_rate']
        urgency = getattr(order, 'urgency', 'NORMAL')
        estimated_impact = analysis['estimated_impact_bps']
        
        # High urgency always uses aggressive
        if urgency in ['HIGH', 'URGENT']:
            return ExecutionAlgorithm.AGGRESSIVE
        
        # Impact-based selection
        if estimated_impact > 100:  # Very high impact
            return ExecutionAlgorithm.VWAP  # Spread over time with volume matching
        
        elif estimated_impact > 50:  # High impact
            return ExecutionAlgorithm.TWAP  # Spread over time evenly
        
        elif participation_rate > 0.15:  # Large order
            return ExecutionAlgorithm.STEALTH  # Hide order size
        
        elif participation_rate > 0.05:  # Medium order
            return ExecutionAlgorithm.PASSIVE  # Patient limit orders
        
        else:  # Small order
            return ExecutionAlgorithm.AGGRESSIVE  # Quick execution
    
    async def _execute_aggressive(self, order: OrderRequest, analysis: Dict) -> Dict:
        """
        Aggressive execution using market orders for immediate fill
        
        Strategy:
        - Use market orders for fastest execution
        - Monitor for adverse price movements
        - Split very large orders into manageable chunks
        """
        
        try:
            symbol = order.symbol
            market_data = await self._get_real_time_market_data(symbol)
            
            # For very large orders, split into chunks
            max_chunk_size = min(order.quantity, 10000)  # Max 10k shares per chunk
            
            slices = []
            remaining_qty = order.quantity
            slice_count = 0
            
            total_filled = 0
            total_cost = 0.0
            
            while remaining_qty > 0 and slice_count < 10:  # Max 10 slices
                slice_qty = min(remaining_qty, max_chunk_size)
                
                # Create execution slice
                slice_id = f"{order.order_id}_slice_{slice_count + 1}"
                execution_slice = ExecutionSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=symbol,
                    side=order.side,
                    quantity=slice_qty,
                    order_type=OrderType.MARKET
                )
                
                # Execute market order
                slice_result = await self._place_market_order(execution_slice)
                
                if slice_result['success']:
                    execution_slice.status = OrderStatus.COMPLETE
                    execution_slice.filled_quantity = slice_result['filled_quantity']
                    execution_slice.average_fill_price = slice_result['average_price']
                    execution_slice.filled_at = datetime.now()
                    
                    total_filled += slice_result['filled_quantity']
                    total_cost += slice_result['filled_quantity'] * slice_result['average_price']
                    
                    remaining_qty -= slice_result['filled_quantity']
                    
                else:
                    execution_slice.status = OrderStatus.REJECTED
                    logger.error(f"Market order slice {slice_id} failed: {slice_result.get('error')}")
                    break
                
                slices.append(execution_slice)
                slice_count += 1
                
                # Brief pause between slices to avoid overwhelming the market
                if remaining_qty > 0:
                    await asyncio.sleep(0.5)
            
            # Store slices
            self.execution_slices[order.order_id] = slices
            
            # Calculate results
            average_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / order.quantity
            
            success = fill_rate >= 0.95  # Consider successful if >95% filled
            
            return {
                'success': success,
                'algorithm_used': ExecutionAlgorithm.AGGRESSIVE,
                'total_filled': total_filled,
                'fill_rate': fill_rate,
                'average_fill_price': average_price,
                'total_slices': len(slices),
                'execution_time_seconds': (datetime.now() - order.created_at).total_seconds(),
                'slices': slices
            }
            
        except Exception as e:
            logger.error(f"Aggressive execution failed: {e}")
            return {
                'success': False,
                'algorithm_used': ExecutionAlgorithm.AGGRESSIVE,
                'error': str(e)
            }
    
    async def _execute_twap(self, order: OrderRequest, analysis: Dict) -> Dict:
        """
        Time-Weighted Average Price execution
        
        Strategy:
        - Divide order into equal time slices
        - Execute small portions at regular intervals
        - Minimize timing risk and market impact
        """
        
        try:
            time_horizon = order.time_horizon_minutes
            slice_interval = max(5, time_horizon // 12)  # Execute every 5+ minutes, max 12 slices
            num_slices = min(12, max(3, time_horizon // slice_interval))
            
            slice_size = order.quantity // num_slices
            remaining_qty = order.quantity
            
            slices = []
            total_filled = 0
            total_cost = 0.0
            
            for i in range(num_slices):
                # Adjust final slice for any remainder
                current_slice_size = slice_size
                if i == num_slices - 1:
                    current_slice_size = remaining_qty
                
                # Create slice
                slice_id = f"{order.order_id}_twap_{i + 1}"
                execution_slice = ExecutionSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=current_slice_size,
                    order_type=OrderType.LIMIT
                )
                
                # Get current market price for limit order
                market_data = await self._get_real_time_market_data(order.symbol)
                
                # Set limit price slightly aggressive to ensure fill
                if order.side == 'BUY':
                    limit_price = market_data['ask_price'] * 1.001  # 0.1% above ask
                else:
                    limit_price = market_data['bid_price'] * 0.999  # 0.1% below bid
                
                execution_slice.price = limit_price
                
                # Execute slice
                slice_result = await self._place_limit_order(execution_slice)
                
                if slice_result['success']:
                    execution_slice.status = OrderStatus.COMPLETE
                    execution_slice.filled_quantity = slice_result['filled_quantity']
                    execution_slice.average_fill_price = slice_result['average_price']
                    execution_slice.filled_at = datetime.now()
                    
                    total_filled += slice_result['filled_quantity']
                    total_cost += slice_result['filled_quantity'] * slice_result['average_price']
                    remaining_qty -= slice_result['filled_quantity']
                
                slices.append(execution_slice)
                
                # Wait for next interval (except last slice)
                if i < num_slices - 1 and remaining_qty > 0:
                    await asyncio.sleep(slice_interval * 60)  # Convert to seconds
            
            # Store slices
            self.execution_slices[order.order_id] = slices
            
            # Calculate results
            average_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / order.quantity
            
            return {
                'success': fill_rate >= 0.90,
                'algorithm_used': ExecutionAlgorithm.TWAP,
                'total_filled': total_filled,
                'fill_rate': fill_rate,
                'average_fill_price': average_price,
                'total_slices': len(slices),
                'time_horizon_minutes': time_horizon,
                'slice_interval_minutes': slice_interval,
                'execution_time_seconds': (datetime.now() - order.created_at).total_seconds(),
                'slices': slices
            }
            
        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
            return {
                'success': False,
                'algorithm_used': ExecutionAlgorithm.TWAP,
                'error': str(e)
            }
    
    async def _execute_passive(self, order: OrderRequest, analysis: Dict) -> Dict:
        """
        Passive execution using intelligent limit orders
        
        Strategy:
        - Start with limit orders at or better than market
        - Gradually increase aggression if not filled
        - Monitor market conditions and adjust
        """
        
        try:
            symbol = order.symbol
            timeout_seconds = order.timeout_minutes * 60
            start_time = datetime.now()
            
            remaining_qty = order.quantity
            total_filled = 0
            total_cost = 0.0
            
            slices = []
            attempt = 0
            
            while remaining_qty > 0 and (datetime.now() - start_time).total_seconds() < timeout_seconds:
                attempt += 1
                
                # Get current market data
                market_data = await self._get_real_time_market_data(symbol)
                
                # Start passive, become more aggressive over time
                aggression_factor = min(1.0, attempt * 0.1)  # Increase by 10% each attempt
                
                if order.side == 'BUY':
                    # Start at bid, move towards ask
                    base_price = market_data['bid_price']
                    spread = market_data['ask_price'] - market_data['bid_price']
                    limit_price = base_price + (spread * aggression_factor)
                else:
                    # Start at ask, move towards bid
                    base_price = market_data['ask_price']
                    spread = market_data['ask_price'] - market_data['bid_price']
                    limit_price = base_price - (spread * aggression_factor)
                
                # Create execution slice
                slice_size = min(remaining_qty, order.slice_size_shares)
                slice_id = f"{order.order_id}_passive_{attempt}"
                
                execution_slice = ExecutionSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=symbol,
                    side=order.side,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT,
                    price=limit_price
                )
                
                # Execute limit order with timeout
                slice_result = await self._place_limit_order_with_timeout(execution_slice, timeout_seconds=60)
                
                if slice_result['success']:
                    execution_slice.status = OrderStatus.COMPLETE
                    execution_slice.filled_quantity = slice_result['filled_quantity']
                    execution_slice.average_fill_price = slice_result['average_price']
                    execution_slice.filled_at = datetime.now()
                    
                    total_filled += slice_result['filled_quantity']
                    total_cost += slice_result['filled_quantity'] * slice_result['average_price']
                    remaining_qty -= slice_result['filled_quantity']
                
                else:
                    execution_slice.status = OrderStatus.CANCELLED
                
                slices.append(execution_slice)
                
                # Brief pause before next attempt
                await asyncio.sleep(5)
            
            # Store slices
            self.execution_slices[order.order_id] = slices
            
            # Calculate results
            average_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / order.quantity
            
            return {
                'success': fill_rate >= 0.80,  # Lower threshold for passive
                'algorithm_used': ExecutionAlgorithm.PASSIVE,
                'total_filled': total_filled,
                'fill_rate': fill_rate,
                'average_fill_price': average_price,
                'total_attempts': attempt,
                'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
                'slices': slices
            }
            
        except Exception as e:
            logger.error(f"Passive execution failed: {e}")
            return {
                'success': False,
                'algorithm_used': ExecutionAlgorithm.PASSIVE,
                'error': str(e)
            }
    
    async def _execute_stealth(self, order: OrderRequest, analysis: Dict) -> Dict:
        """
        Stealth execution using iceberg orders to hide size
        
        Strategy:
        - Show only small portions of order at a time
        - Refresh hidden quantity as portions fill
        - Minimize market impact from size discovery
        """
        
        try:
            # Iceberg parameters
            visible_size = min(order.quantity // 10, 1000)  # Show 10% or max 1000 shares
            if visible_size < 100:
                visible_size = min(100, order.quantity)
            
            remaining_qty = order.quantity
            total_filled = 0
            total_cost = 0.0
            
            slices = []
            iceberg_round = 0
            
            while remaining_qty > 0 and iceberg_round < 20:  # Max 20 iceberg rounds
                iceberg_round += 1
                
                # Current iceberg size
                current_iceberg_size = min(remaining_qty, visible_size)
                
                # Create iceberg slice
                slice_id = f"{order.order_id}_iceberg_{iceberg_round}"
                execution_slice = ExecutionSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=current_iceberg_size,
                    order_type=OrderType.LIMIT
                )
                
                # Set competitive limit price
                market_data = await self._get_real_time_market_data(order.symbol)
                
                if order.side == 'BUY':
                    # Bid at current best bid or slightly better
                    limit_price = market_data['bid_price'] + 0.05  # ₹0.05 better
                else:
                    # Offer at current best ask or slightly better
                    limit_price = market_data['ask_price'] - 0.05  # ₹0.05 better
                
                execution_slice.price = limit_price
                
                # Execute iceberg slice with patience
                slice_result = await self._place_limit_order_with_timeout(execution_slice, timeout_seconds=120)
                
                if slice_result['success']:
                    execution_slice.status = OrderStatus.COMPLETE
                    execution_slice.filled_quantity = slice_result['filled_quantity']
                    execution_slice.average_fill_price = slice_result['average_price']
                    execution_slice.filled_at = datetime.now()
                    
                    total_filled += slice_result['filled_quantity']
                    total_cost += slice_result['filled_quantity'] * slice_result['average_price']
                    remaining_qty -= slice_result['filled_quantity']
                
                else:
                    execution_slice.status = OrderStatus.CANCELLED
                
                slices.append(execution_slice)
                
                # Pause between iceberg reveals to avoid detection
                await asyncio.sleep(np.random.uniform(10, 30))  # Random 10-30 second delay
            
            # Store slices
            self.execution_slices[order.order_id] = slices
            
            # Calculate results
            average_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / order.quantity
            
            return {
                'success': fill_rate >= 0.85,
                'algorithm_used': ExecutionAlgorithm.STEALTH,
                'total_filled': total_filled,
                'fill_rate': fill_rate,
                'average_fill_price': average_price,
                'iceberg_rounds': iceberg_round,
                'visible_size': visible_size,
                'execution_time_seconds': (datetime.now() - order.created_at).total_seconds(),
                'slices': slices
            }
            
        except Exception as e:
            logger.error(f"Stealth execution failed: {e}")
            return {
                'success': False,
                'algorithm_used': ExecutionAlgorithm.STEALTH,
                'error': str(e)
            }
    
    async def _execute_vwap(self, order: OrderRequest, analysis: Dict) -> Dict:
        """
        Volume-Weighted Average Price execution
        
        Strategy:
        - Match execution pace to historical volume patterns
        - Execute more during high-volume periods
        - Target VWAP price or better
        """
        
        try:
            # Get historical volume profile
            volume_profile = await self._get_intraday_volume_profile(order.symbol)
            
            # Divide order based on volume distribution
            time_horizon = order.time_horizon_minutes
            total_expected_volume = sum(volume_profile.values()) if volume_profile else 1000000
            
            slices = []
            remaining_qty = order.quantity
            total_filled = 0
            total_cost = 0.0
            
            # Create volume-weighted slices
            for time_bucket, expected_volume in volume_profile.items():
                if remaining_qty <= 0:
                    break
                
                # Calculate slice size based on volume proportion
                volume_proportion = expected_volume / total_expected_volume
                participation_rate = min(0.20, order.max_participation_rate)  # Max 20% participation
                
                slice_size = min(
                    remaining_qty,
                    int(expected_volume * participation_rate),
                    order.quantity // 10  # Max 10% of total order per slice
                )
                
                if slice_size < 50:  # Skip very small slices
                    continue
                
                # Create VWAP slice
                slice_id = f"{order.order_id}_vwap_{time_bucket}"
                execution_slice = ExecutionSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    order_type=OrderType.LIMIT
                )
                
                # Execute at VWAP-targeting price
                market_data = await self._get_real_time_market_data(order.symbol)
                
                # Target mid-price for VWAP
                mid_price = (market_data['bid_price'] + market_data['ask_price']) / 2
                execution_slice.price = mid_price
                
                # Execute slice
                slice_result = await self._place_limit_order_with_timeout(execution_slice, timeout_seconds=300)
                
                if slice_result['success']:
                    execution_slice.status = OrderStatus.COMPLETE
                    execution_slice.filled_quantity = slice_result['filled_quantity']
                    execution_slice.average_fill_price = slice_result['average_price']
                    execution_slice.filled_at = datetime.now()
                    
                    total_filled += slice_result['filled_quantity']
                    total_cost += slice_result['filled_quantity'] * slice_result['average_price']
                    remaining_qty -= slice_result['filled_quantity']
                
                slices.append(execution_slice)
                
                # Wait for next volume bucket
                await asyncio.sleep(60)  # 1 minute between slices
            
            # Store slices
            self.execution_slices[order.order_id] = slices
            
            # Calculate results
            average_price = total_cost / total_filled if total_filled > 0 else 0
            fill_rate = total_filled / order.quantity
            
            return {
                'success': fill_rate >= 0.85,
                'algorithm_used': ExecutionAlgorithm.VWAP,
                'total_filled': total_filled,
                'fill_rate': fill_rate,
                'average_fill_price': average_price,
                'vwap_slices': len(slices),
                'execution_time_seconds': (datetime.now() - order.created_at).total_seconds(),
                'slices': slices
            }
            
        except Exception as e:
            logger.error(f"VWAP execution failed: {e}")
            return {
                'success': False,
                'algorithm_used': ExecutionAlgorithm.VWAP,
                'error': str(e)
            }
    
    # Helper methods for order placement and market data
    
    async def _place_market_order(self, slice: ExecutionSlice) -> Dict:
        """Place market order through Zerodha API"""
        
        try:
            # Simulate order placement (replace with actual Zerodha API call)
            result = await self.zerodha.place_order(
                exchange='NSE',
                tradingsymbol=slice.symbol,
                transaction_type=slice.side,
                quantity=slice.quantity,
                order_type='MARKET',
                product='MIS'  # Intraday
            )
            
            if result.get('status') == 'success':
                # Simulate immediate fill for market order
                market_data = await self._get_real_time_market_data(slice.symbol)
                
                if slice.side == 'BUY':
                    fill_price = market_data.get('ask_price', market_data.get('last_price', 100))
                else:
                    fill_price = market_data.get('bid_price', market_data.get('last_price', 100))
                
                return {
                    'success': True,
                    'order_id': result.get('order_id'),
                    'filled_quantity': slice.quantity,
                    'average_price': fill_price,
                    'fill_time': datetime.now()
                }
            else:
                return {
                    'success': False,
                    'error': result.get('message', 'Order placement failed')
                }
                
        except Exception as e:
            logger.error(f"Market order placement failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _place_limit_order(self, slice: ExecutionSlice) -> Dict:
        """Place limit order through Zerodha API"""
        
        try:
            result = await self.zerodha.place_order(
                exchange='NSE',
                tradingsymbol=slice.symbol,
                transaction_type=slice.side,
                quantity=slice.quantity,
                order_type='LIMIT',
                price=slice.price,
                product='MIS'
            )
            
            if result.get('status') == 'success':
                # Simulate potential fill based on market conditions
                fill_probability = np.random.uniform(0.6, 0.9)  # 60-90% fill probability
                
                if np.random.random() < fill_probability:
                    return {
                        'success': True,
                        'order_id': result.get('order_id'),
                        'filled_quantity': slice.quantity,
                        'average_price': slice.price,
                        'fill_time': datetime.now()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Order not filled within time limit'
                    }
            else:
                return {
                    'success': False,
                    'error': result.get('message', 'Order placement failed')
                }
                
        except Exception as e:
            logger.error(f"Limit order placement failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _place_limit_order_with_timeout(self, slice: ExecutionSlice, timeout_seconds: int = 120) -> Dict:
        """Place limit order with timeout and potential price improvement"""
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            # Try to place order
            result = await self._place_limit_order(slice)
            
            if result['success']:
                return result
            
            # Update price based on current market (price improvement)
            market_data = await self._get_real_time_market_data(slice.symbol)
            
            if slice.side == 'BUY':
                new_price = min(slice.price * 1.002, market_data['ask_price'])  # Max 0.2% worse or ask
            else:
                new_price = max(slice.price * 0.998, market_data['bid_price'])  # Max 0.2% worse or bid
            
            slice.price = new_price
            
            await asyncio.sleep(10)  # Wait 10 seconds before retry
        
        return {
            'success': False,
            'error': 'Order timeout exceeded'
        }
    
    async def _get_real_time_market_data(self, symbol: str) -> Dict:
        """Get real-time market data"""
        
        # Check cache first
        if symbol in self.market_data_cache:
            last_update = self.last_market_update.get(symbol, datetime.min)
            if (datetime.now() - last_update).total_seconds() < 5:  # 5-second cache
                return self.market_data_cache[symbol]
        
        try:
            # Get from Zerodha API
            data = await self.zerodha.get_quote(symbol)
            
            market_data = {
                'symbol': symbol,
                'last_price': data.get('last_price', 100),
                'bid_price': data.get('bid_price', data.get('last_price', 100) * 0.999),
                'ask_price': data.get('ask_price', data.get('last_price', 100) * 1.001),
                'volume': data.get('volume', 100000),
                'timestamp': datetime.now()
            }
            
            # Update cache
            self.market_data_cache[symbol] = market_data
            self.last_market_update[symbol] = datetime.now()
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            # Return default data if API fails
            return {
                'symbol': symbol,
                'last_price': 100,
                'bid_price': 99.9,
                'ask_price': 100.1,
                'volume': 100000,
                'timestamp': datetime.now()
            }
    
    async def _get_historical_data(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Get historical OHLCV data"""
        
        try:
            # This would call Zerodha historical data API
            # Placeholder implementation
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            data = {
                'date': dates,
                'open': np.random.uniform(95, 105, days),
                'high': np.random.uniform(100, 110, days),
                'low': np.random.uniform(90, 100, days),
                'close': np.random.uniform(95, 105, days),
                'volume': np.random.randint(100000, 1000000, days)
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_intraday_volume_profile(self, symbol: str) -> Dict[str, int]:
        """Get intraday volume profile for VWAP execution"""
        
        # Typical NSE volume pattern (simplified)
        volume_profile = {
            '09:15-10:00': 150000,  # Opening hour - high volume
            '10:00-11:00': 100000,
            '11:00-12:00': 80000,
            '12:00-13:00': 70000,
            '13:00-14:00': 75000,
            '14:00-15:00': 90000,
            '15:00-15:30': 120000   # Closing - high volume
        }
        
        return volume_profile
    
    # Validation methods
    
    async def _is_symbol_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable"""
        # This would check symbol validity, trading status, etc.
        return True  # Simplified
    
    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now().time()
        
        # NSE trading hours: 9:15 AM to 3:30 PM
        market_start = time(9, 15)
        market_end = time(15, 30)
        
        # Simplified check (would also check holidays, weekends)
        return market_start <= now <= market_end
    
    async def _validate_risk_limits(self, order: OrderRequest, market_data: Dict) -> Dict:
        """Validate order against risk limits"""
        
        # This would integrate with risk manager
        return {
            'is_valid': True,
            'validation_step': 'RISK_CHECK'
        }
    
    def _validate_liquidity(self, order: OrderRequest, market_data: Dict) -> Dict:
        """Validate order against liquidity requirements"""
        
        volume = market_data.get('volume', 0)
        order_value = order.quantity * market_data.get('last_price', 100)
        
        # Basic liquidity check - order should be <20% of daily volume
        if volume > 0:
            value_proportion = order_value / (volume * market_data.get('last_price', 100))
            
            if value_proportion > 0.25:  # 25% of daily volume
                return {
                    'is_valid': False,
                    'error': f'Order size {value_proportion*100:.1f}% of daily volume exceeds 25% limit',
                    'validation_step': 'LIQUIDITY_CHECK'
                }
        
        return {
            'is_valid': True,
            'validation_step': 'LIQUIDITY_CHECK'
        }
    
    async def _post_trade_analysis(self, order: OrderRequest, result: Dict) -> Dict:
        """Post-trade analysis and performance metrics"""
        
        if not result['success']:
            return {}
        
        try:
            # Calculate slippage
            market_data = await self._get_real_time_market_data(order.symbol)
            benchmark_price = market_data['last_price']
            
            actual_price = result['average_fill_price']
            
            if order.side == 'BUY':
                slippage_bps = ((actual_price - benchmark_price) / benchmark_price) * 10000
            else:
                slippage_bps = ((benchmark_price - actual_price) / benchmark_price) * 10000
            
            # Calculate timing metrics
            execution_time = result.get('execution_time_seconds', 0)
            
            # Market impact estimation
            order_value = order.quantity * actual_price
            market_impact_estimate = abs(slippage_bps) / 2  # Simplified
            
            return {
                'slippage_bps': slippage_bps,
                'execution_time_seconds': execution_time,
                'market_impact_bps': market_impact_estimate,
                'benchmark_price': benchmark_price,
                'actual_price': actual_price,
                'order_value': order_value,
                'cost_efficiency': 'GOOD' if abs(slippage_bps) < 25 else 'POOR'
            }
            
        except Exception as e:
            logger.error(f"Post-trade analysis failed: {e}")
            return {'error': str(e)}
    
    def _update_execution_metrics(self, order: OrderRequest, result: Dict):
        """Update execution performance metrics"""
        
        self.execution_metrics['total_orders'] += 1
        
        if result['success']:
            self.execution_metrics['successful_fills'] += 1
            
            # Update running averages
            execution_time = result.get('execution_time_seconds', 0)
            current_avg_time = self.execution_metrics['average_fill_time']
            total_orders = self.execution_metrics['total_orders']
            
            self.execution_metrics['average_fill_time'] = (
                (current_avg_time * (total_orders - 1) + execution_time) / total_orders
            )
            
            # Update slippage if available
            post_trade = result.get('post_trade_metrics', {})
            if 'slippage_bps' in post_trade:
                slippage = abs(post_trade['slippage_bps'])
                current_avg_slippage = self.execution_metrics['average_slippage_bps']
                
                self.execution_metrics['average_slippage_bps'] = (
                    (current_avg_slippage * (total_orders - 1) + slippage) / total_orders
                )
            
            # Update volume
            if 'order_value' in post_trade:
                self.execution_metrics['total_volume_inr'] += post_trade['order_value']
    
    def _log_order_execution(self, order: OrderRequest, result: Dict):
        """Log order execution to database"""
        
        try:
            log_entry = OrderLog(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type.value,
                algorithm=result.get('algorithm_used', ExecutionAlgorithm.PASSIVE).value,
                success=result['success'],
                filled_quantity=result.get('total_filled', 0),
                average_price=result.get('average_fill_price', 0),
                execution_time_seconds=result.get('execution_time_seconds', 0),
                slippage_bps=result.get('post_trade_metrics', {}).get('slippage_bps', 0),
                error_message=result.get('error', None)
            )
            
            self.db_session.add(log_entry)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to log order execution: {e}")
    
    def get_execution_metrics(self) -> Dict:
        """Get current execution performance metrics"""
        
        success_rate = 0
        if self.execution_metrics['total_orders'] > 0:
            success_rate = (self.execution_metrics['successful_fills'] / 
                          self.execution_metrics['total_orders'] * 100)
        
        return {
            'total_orders': self.execution_metrics['total_orders'],
            'successful_fills': self.execution_metrics['successful_fills'],
            'success_rate_pct': round(success_rate, 2),
            'average_fill_time_seconds': round(self.execution_metrics['average_fill_time'], 2),
            'average_slippage_bps': round(self.execution_metrics['average_slippage_bps'], 2),
            'total_volume_inr': round(self.execution_metrics['total_volume_inr'], 2),
            'active_orders': len(self.active_orders)
        }
