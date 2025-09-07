"""
Comprehensive Backtesting Engine
Walk-forward optimization with realistic transaction costs and slippage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from config.trading_config import get_trading_config, BacktestConfig
from analytics.fundamental_analyzer import FundamentalAnalyzer, TechnicalAnalyzer
from analytics.quantitative_engine import QuantitativeEngine, MacroSentimentAnalyzer
from strategy.trade_filter import AdvancedTradeFilter
from strategy.position_manager import CapitalManager, PortfolioTracker
from strategy.risk_manager import ComprehensiveRiskManager

logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    
    # Performance metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Risk metrics
    var_95: float
    var_99: float
    beta: float
    alpha: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Execution metrics
    avg_holding_days: float
    avg_slippage_bps: float
    total_commission: float
    
    # Portfolio metrics
    avg_positions: float
    max_positions: int
    sector_concentration: Dict[str, float]
    
    # Time series data
    equity_curve: pd.Series
    daily_returns: pd.Series
    position_history: pd.DataFrame
    trade_history: pd.DataFrame
    
    # Walk-forward results
    oos_return: float  # Out-of-sample return
    oos_sharpe: float  # Out-of-sample Sharpe
    stability_ratio: float  # IS vs OOS performance consistency

@dataclass
class OptimizationResults:
    """Parameter optimization results"""
    
    best_params: Dict[str, Union[float, int]]
    best_score: float
    param_sensitivity: Dict[str, List[Tuple[float, float]]]
    optimization_surface: pd.DataFrame
    convergence_data: List[Dict]


class ComprehensiveBacktester:
    """
    Institutional-grade backtesting engine with:
    
    1. Walk-Forward Optimization
    2. Realistic Transaction Costs
    3. Market Impact Modeling
    4. Regime-Aware Testing
    5. Monte Carlo Validation
    6. Multi-Asset Universe Testing
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trading_config = get_trading_config()
        
        # Initialize components
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.quantitative_engine = QuantitativeEngine()
        self.macro_analyzer = MacroSentimentAnalyzer()
        self.trade_filter = AdvancedTradeFilter()
        self.capital_manager = CapitalManager(self.trading_config)
        self.risk_manager = ComprehensiveRiskManager()
        
        # Backtest state
        self.current_date = None
        self.portfolio_tracker = None
        self.universe_data = {}
        self.results_cache = {}
        
        logger.info("📊 Comprehensive Backtester initialized")
    
    def run_full_backtest(self, 
                         start_date: str,
                         end_date: str,
                         initial_capital: float = 5000000,
                         universe_symbols: List[str] = None) -> BacktestResults:
        """
        Run comprehensive backtest with all features
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital in rupees
            universe_symbols: List of symbols to test (None = Nifty universe)
            
        Returns:
            BacktestResults with comprehensive metrics
        """
        
        logger.info(f"🎯 Starting full backtest: {start_date} to {end_date}")
        
        try:
            # 1. Prepare data
            universe_data = self._prepare_universe_data(start_date, end_date, universe_symbols)
            
            if not universe_data:
                raise ValueError("No data available for backtest period")
            
            # 2. Initialize portfolio tracker
            portfolio_tracker = PortfolioTracker(
                initial_capital=initial_capital,
                max_positions=self.trading_config.MAX_POSITIONS,
                max_sector_allocation=self.trading_config.MAX_SECTOR_ALLOCATION
            )
            
            # 3. Run simulation
            simulation_results = self._run_simulation(
                universe_data, 
                portfolio_tracker, 
                start_date, 
                end_date
            )
            
            # 4. Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(simulation_results)
            
            # 5. Generate trade statistics
            trade_stats = self._calculate_trade_statistics(simulation_results['trades'])
            
            # 6. Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(simulation_results['returns'])
            
            # 7. Portfolio analysis
            portfolio_metrics = self._analyze_portfolio_metrics(simulation_results['positions'])
            
            # 8. Execution analysis
            execution_metrics = self._analyze_execution_metrics(simulation_results['trades'])
            
            # 9. Compile results
            results = BacktestResults(
                # Performance
                total_return=performance_metrics['total_return'],
                annual_return=performance_metrics['annual_return'],
                volatility=performance_metrics['volatility'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                max_drawdown=performance_metrics['max_drawdown'],
                calmar_ratio=performance_metrics['calmar_ratio'],
                
                # Risk
                var_95=risk_metrics['var_95'],
                var_99=risk_metrics['var_99'],
                beta=risk_metrics.get('beta', 1.0),
                alpha=risk_metrics.get('alpha', 0.0),
                
                # Trades
                total_trades=trade_stats['total_trades'],
                winning_trades=trade_stats['winning_trades'],
                losing_trades=trade_stats['losing_trades'],
                win_rate=trade_stats['win_rate'],
                avg_win=trade_stats['avg_win'],
                avg_loss=trade_stats['avg_loss'],
                profit_factor=trade_stats['profit_factor'],
                
                # Execution
                avg_holding_days=execution_metrics['avg_holding_days'],
                avg_slippage_bps=execution_metrics['avg_slippage_bps'],
                total_commission=execution_metrics['total_commission'],
                
                # Portfolio
                avg_positions=portfolio_metrics['avg_positions'],
                max_positions=portfolio_metrics['max_positions'],
                sector_concentration=portfolio_metrics['sector_concentration'],
                
                # Time series
                equity_curve=simulation_results['equity_curve'],
                daily_returns=simulation_results['returns'],
                position_history=simulation_results['positions'],
                trade_history=simulation_results['trades'],
                
                # Walk-forward (will be set by walk-forward method)
                oos_return=0.0,
                oos_sharpe=0.0,
                stability_ratio=0.0
            )
            
            logger.info(f"✅ Backtest completed: {performance_metrics['annual_return']:.1f}% annual return, "
                       f"{performance_metrics['sharpe_ratio']:.2f} Sharpe, "
                       f"{performance_metrics['max_drawdown']:.1f}% max drawdown")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def walk_forward_optimization(self,
                                start_date: str,
                                end_date: str,
                                param_ranges: Dict[str, List],
                                initial_capital: float = 5000000,
                                train_months: int = 12,
                                test_months: int = 3) -> Tuple[BacktestResults, OptimizationResults]:
        """
        Walk-forward optimization with realistic out-of-sample testing
        
        Args:
            start_date: Optimization start date
            end_date: Optimization end date
            param_ranges: Parameter ranges to optimize
            initial_capital: Starting capital
            train_months: Training period length
            test_months: Testing period length
            
        Returns:
            Tuple of (best backtest results, optimization results)
        """
        
        logger.info(f"🔄 Starting walk-forward optimization: {train_months}M train, {test_months}M test")
        
        try:
            # 1. Create walk-forward windows
            windows = self._create_walk_forward_windows(
                start_date, end_date, train_months, test_months
            )
            
            if len(windows) < 2:
                raise ValueError("Insufficient data for walk-forward optimization")
            
            # 2. Optimize parameters for each window
            window_results = []
            best_params_by_window = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logger.info(f"Window {i+1}/{len(windows)}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
                
                # Optimize on training data
                window_optimization = self._optimize_parameters(
                    train_start, train_end, param_ranges, initial_capital
                )
                
                best_params = window_optimization.best_params
                best_params_by_window.append(best_params)
                
                # Test on out-of-sample data
                oos_results = self._run_backtest_with_params(
                    test_start, test_end, best_params, initial_capital
                )
                
                window_results.append({
                    'window': i + 1,
                    'train_period': f"{train_start} to {train_end}",
                    'test_period': f"{test_start} to {test_end}",
                    'is_return': window_optimization.best_score,
                    'oos_return': oos_results.annual_return,
                    'is_sharpe': 0.0,  # Would be calculated from optimization
                    'oos_sharpe': oos_results.sharpe_ratio,
                    'best_params': best_params
                })
            
            # 3. Analyze parameter stability
            param_stability = self._analyze_parameter_stability(best_params_by_window)
            
            # 4. Calculate ensemble parameters
            ensemble_params = self._calculate_ensemble_parameters(best_params_by_window, param_stability)
            
            # 5. Run final backtest with ensemble parameters
            final_results = self._run_backtest_with_params(
                start_date, end_date, ensemble_params, initial_capital
            )
            
            # 6. Calculate walk-forward metrics
            oos_returns = [w['oos_return'] for w in window_results]
            is_returns = [w['is_return'] for w in window_results]
            
            avg_oos_return = np.mean(oos_returns)
            avg_oos_sharpe = np.mean([w['oos_sharpe'] for w in window_results])
            
            # Stability ratio: OOS performance / IS performance
            stability_ratio = avg_oos_return / np.mean(is_returns) if np.mean(is_returns) > 0 else 0
            
            # Update final results with walk-forward metrics
            final_results.oos_return = avg_oos_return
            final_results.oos_sharpe = avg_oos_sharpe
            final_results.stability_ratio = stability_ratio
            
            # 7. Compile optimization results
            optimization_results = OptimizationResults(
                best_params=ensemble_params,
                best_score=final_results.annual_return,
                param_sensitivity=param_stability,
                optimization_surface=pd.DataFrame(window_results),
                convergence_data=window_results
            )
            
            logger.info(f"✅ Walk-forward optimization completed: "
                       f"Final return {final_results.annual_return:.1f}%, "
                       f"Stability ratio {stability_ratio:.2f}")
            
            return final_results, optimization_results
            
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {e}")
            raise
    
    def monte_carlo_validation(self,
                              base_results: BacktestResults,
                              num_simulations: int = 1000,
                              confidence_level: float = 0.95) -> Dict[str, Union[float, List]]:
        """
        Monte Carlo validation of backtest results
        
        Generates confidence intervals for key metrics by:
        1. Bootstrap resampling of returns
        2. Random start date variations
        3. Parameter uncertainty modeling
        
        Args:
            base_results: Base backtest results
            num_simulations: Number of Monte Carlo runs
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict with confidence intervals and validation metrics
        """
        
        logger.info(f"🎲 Running Monte Carlo validation: {num_simulations} simulations")
        
        try:
            base_returns = base_results.daily_returns
            
            # Storage for simulation results
            sim_results = {
                'annual_returns': [],
                'sharpe_ratios': [],
                'max_drawdowns': [],
                'total_returns': []
            }
            
            for sim in range(num_simulations):
                if sim % 100 == 0:
                    logger.info(f"Simulation {sim}/{num_simulations}")
                
                # Method 1: Bootstrap resampling
                if sim < num_simulations // 3:
                    simulated_returns = self._bootstrap_returns(base_returns)
                
                # Method 2: Block bootstrap (preserves serial correlation)
                elif sim < 2 * num_simulations // 3:
                    simulated_returns = self._block_bootstrap_returns(base_returns, block_size=20)
                
                # Method 3: Random start dates
                else:
                    simulated_returns = self._random_start_simulation(base_results)
                
                # Calculate metrics for simulation
                sim_metrics = self._calculate_performance_metrics_from_returns(simulated_returns)
                
                sim_results['annual_returns'].append(sim_metrics['annual_return'])
                sim_results['sharpe_ratios'].append(sim_metrics['sharpe_ratio'])
                sim_results['max_drawdowns'].append(sim_metrics['max_drawdown'])
                sim_results['total_returns'].append(sim_metrics['total_return'])
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            confidence_intervals = {}
            for metric, values in sim_results.items():
                confidence_intervals[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'lower': np.percentile(values, lower_percentile),
                    'upper': np.percentile(values, upper_percentile),
                    'base_value': getattr(base_results, metric.replace('_', '') if metric.endswith('s') else metric)
                }
            
            # Risk of ruin calculation
            negative_returns = [r for r in sim_results['total_returns'] if r < 0]
            risk_of_ruin = len(negative_returns) / num_simulations
            
            # Probability of meeting targets
            target_annual_return = 15.0  # 15% target
            prob_meet_target = len([r for r in sim_results['annual_returns'] if r >= target_annual_return]) / num_simulations
            
            validation_results = {
                'confidence_intervals': confidence_intervals,
                'risk_of_ruin': risk_of_ruin,
                'prob_meet_target': prob_meet_target,
                'target_annual_return': target_annual_return,
                'num_simulations': num_simulations,
                'confidence_level': confidence_level,
                'validation_summary': {
                    'robust_performance': all(
                        ci['base_value'] >= ci['lower'] for ci in confidence_intervals.values()
                    ),
                    'expected_annual_return': confidence_intervals['annual_returns']['mean'],
                    'worst_case_drawdown': confidence_intervals['max_drawdowns']['upper']
                }
            }
            
            logger.info(f"✅ Monte Carlo validation completed: "
                       f"Risk of ruin {risk_of_ruin:.1%}, "
                       f"Target probability {prob_meet_target:.1%}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Monte Carlo validation failed: {e}")
            raise
    
    def regime_analysis(self,
                       results: BacktestResults,
                       market_index_data: pd.Series = None) -> Dict[str, Dict]:
        """
        Analyze strategy performance across different market regimes
        
        Regimes:
        1. Bull Market (strong uptrend)
        2. Bear Market (strong downtrend) 
        3. Sideways Market (low volatility)
        4. High Volatility (crisis periods)
        
        Args:
            results: Backtest results
            market_index_data: Market index returns for regime identification
            
        Returns:
            Dict with performance by regime
        """
        
        logger.info("📈 Analyzing performance across market regimes")
        
        try:
            strategy_returns = results.daily_returns
            
            # Use Nifty 50 if no benchmark provided
            if market_index_data is None:
                market_index_data = self._get_nifty_data(
                    strategy_returns.index[0], 
                    strategy_returns.index[-1]
                )
            
            # Define regimes using market conditions
            regimes = self._identify_market_regimes(market_index_data)
            
            # Analyze performance by regime
            regime_performance = {}
            
            for regime_name, regime_mask in regimes.items():
                # Get strategy returns for this regime
                regime_returns = strategy_returns[regime_mask]
                
                if len(regime_returns) == 0:
                    continue
                
                # Calculate regime-specific metrics
                regime_metrics = self._calculate_performance_metrics_from_returns(regime_returns)
                
                # Additional regime analysis
                market_returns = market_index_data[regime_mask]
                
                # Beta calculation
                if len(regime_returns) > 10 and len(market_returns) > 10:
                    aligned_data = pd.DataFrame({
                        'strategy': regime_returns,
                        'market': market_returns
                    }).dropna()
                    
                    if len(aligned_data) > 10:
                        covariance = aligned_data.cov().iloc[0, 1]
                        market_var = aligned_data['market'].var()
                        beta = covariance / market_var if market_var > 0 else 1.0
                        
                        alpha = regime_metrics['annual_return'] - beta * market_returns.mean() * 252
                    else:
                        beta = 1.0
                        alpha = 0.0
                else:
                    beta = 1.0
                    alpha = 0.0
                
                regime_performance[regime_name] = {
                    'period_count': len(regime_returns),
                    'period_days': len(regime_returns),
                    'total_return': regime_metrics['total_return'],
                    'annual_return': regime_metrics['annual_return'],
                    'volatility': regime_metrics['volatility'],
                    'sharpe_ratio': regime_metrics['sharpe_ratio'],
                    'max_drawdown': regime_metrics['max_drawdown'],
                    'beta': beta,
                    'alpha': alpha,
                    'win_rate': len(regime_returns[regime_returns > 0]) / len(regime_returns),
                    'avg_return': regime_returns.mean(),
                    'worst_day': regime_returns.min(),
                    'best_day': regime_returns.max()
                }
            
            # Calculate regime consistency
            regime_sharpe_ratios = [perf['sharpe_ratio'] for perf in regime_performance.values()]
            regime_consistency = 1 - (np.std(regime_sharpe_ratios) / np.mean(regime_sharpe_ratios)) if regime_sharpe_ratios else 0
            
            # Overall regime analysis
            regime_summary = {
                'regime_performance': regime_performance,
                'regime_consistency': regime_consistency,
                'best_regime': max(regime_performance.keys(), 
                                 key=lambda x: regime_performance[x]['sharpe_ratio']) if regime_performance else None,
                'worst_regime': min(regime_performance.keys(), 
                                  key=lambda x: regime_performance[x]['sharpe_ratio']) if regime_performance else None,
                'regime_diversification_benefit': self._calculate_regime_diversification(regime_performance)
            }
            
            logger.info(f"✅ Regime analysis completed: "
                       f"Best regime {regime_summary.get('best_regime', 'Unknown')}, "
                       f"Consistency {regime_consistency:.2f}")
            
            return regime_summary
            
        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return {}
    
    # Core simulation methods
    
    def _run_simulation(self,
                       universe_data: Dict[str, pd.DataFrame],
                       portfolio_tracker: PortfolioTracker,
                       start_date: str,
                       end_date: str) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Run the main backtest simulation
        """
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Storage for results
        equity_values = []
        daily_returns = []
        position_records = []
        trade_records = []
        
        # Simulation state
        last_equity = portfolio_tracker.initial_capital
        
        for current_date in dates:
            self.current_date = current_date
            
            try:
                # 1. Update market data for current date
                self._update_market_data(current_date, universe_data)
                
                # 2. Update portfolio with current prices
                portfolio_tracker.update_positions(current_date)
                
                # 3. Check for exits (risk management)
                exit_decisions = self._check_exit_conditions(portfolio_tracker, current_date)
                
                # 4. Execute exits
                for exit_decision in exit_decisions:
                    trade_record = self._execute_exit(portfolio_tracker, exit_decision, current_date)
                    if trade_record:
                        trade_records.append(trade_record)
                
                # 5. Generate new trade signals (only on certain days to reduce overfitting)
                if self._should_generate_signals(current_date):
                    trade_signals = self._generate_trade_signals(current_date, universe_data)
                    
                    # 6. Filter and prioritize signals
                    filtered_signals = self.trade_filter.filter_top_trades(trade_signals)
                    
                    # 7. Size positions
                    sized_positions = self._size_positions(filtered_signals, portfolio_tracker)
                    
                    # 8. Execute entries
                    for position in sized_positions:
                        trade_record = self._execute_entry(portfolio_tracker, position, current_date)
                        if trade_record:
                            trade_records.append(trade_record)
                
                # 9. Record portfolio state
                current_equity = portfolio_tracker.total_value
                daily_return = (current_equity - last_equity) / last_equity if last_equity > 0 else 0
                
                equity_values.append(current_equity)
                daily_returns.append(daily_return)
                
                # Record positions
                position_record = {
                    'date': current_date,
                    'total_value': current_equity,
                    'cash': portfolio_tracker.cash,
                    'invested_value': portfolio_tracker.invested_value,
                    'num_positions': len(portfolio_tracker.positions),
                    'daily_pnl': current_equity - last_equity,
                    'daily_return': daily_return
                }
                position_records.append(position_record)
                
                last_equity = current_equity
                
            except Exception as e:
                logger.warning(f"Simulation error on {current_date}: {e}")
                continue
        
        # Convert to pandas objects
        equity_curve = pd.Series(equity_values, index=dates[:len(equity_values)])
        returns_series = pd.Series(daily_returns, index=dates[:len(daily_returns)])
        positions_df = pd.DataFrame(position_records)
        trades_df = pd.DataFrame(trade_records)
        
        return {
            'equity_curve': equity_curve,
            'returns': returns_series,
            'positions': positions_df,
            'trades': trades_df
        }
    
    def _generate_trade_signals(self, current_date: datetime, universe_data: Dict) -> List[Dict]:
        """Generate trade signals for current date"""
        
        signals = []
        
        # Get available symbols for current date
        available_symbols = []
        for symbol, data in universe_data.items():
            if current_date in data.index:
                available_symbols.append(symbol)
        
        if len(available_symbols) < 10:  # Need minimum universe
            return signals
        
        # Generate signals using trade filter
        try:
            # This would integrate with the trade filter's composite scoring
            for symbol in available_symbols:
                # Simplified signal generation
                signal_data = {
                    'symbol': symbol,
                    'date': current_date,
                    'composite_score': np.random.uniform(0.3, 0.9),  # Placeholder
                    'fundamental_score': np.random.uniform(0.3, 0.9),
                    'technical_score': np.random.uniform(0.3, 0.9),
                    'quantitative_score': np.random.uniform(0.3, 0.9),
                    'macro_score': np.random.uniform(0.3, 0.9),
                    'price': universe_data[symbol].loc[current_date, 'close'],
                    'conviction_level': np.random.uniform(0.5, 1.0)
                }
                
                # Only include high-quality signals
                if signal_data['composite_score'] >= 0.70:
                    signals.append(signal_data)
            
        except Exception as e:
            logger.warning(f"Signal generation failed for {current_date}: {e}")
        
        return signals
    
    def _should_generate_signals(self, current_date: datetime) -> bool:
        """Determine if we should generate signals on this date"""
        
        # Generate signals only on certain days to avoid overfitting
        # For example: Monday, Wednesday, Friday
        weekday = current_date.weekday()
        return weekday in [0, 2, 4]  # Monday=0, Wednesday=2, Friday=4
    
    def _size_positions(self, signals: List[Dict], portfolio_tracker: PortfolioTracker) -> List[Dict]:
        """Size positions based on signals and risk management"""
        
        sized_positions = []
        
        try:
            current_portfolio_state = {
                'positions': portfolio_tracker.positions,
                'cash': portfolio_tracker.cash,
                'total_value': portfolio_tracker.total_value
            }
            
            for signal in signals:
                # Check if we can add this position
                if len(portfolio_tracker.positions) >= self.trading_config.MAX_POSITIONS:
                    break
                
                # Skip if already holding
                if signal['symbol'] in portfolio_tracker.positions:
                    continue
                
                # Calculate position size
                position_size = self.capital_manager.calculate_position_size(
                    symbol=signal['symbol'],
                    composite_score=signal['composite_score'],
                    conviction_level=signal['conviction_level'],
                    volatility_atr=0.025,  # Simplified - would use actual ATR
                    current_portfolio=current_portfolio_state
                )
                
                if position_size['validation']['within_position_limits']:
                    sized_positions.append({
                        'symbol': signal['symbol'],
                        'side': 'BUY',  # Simplified - assume long-only
                        'position_size_inr': position_size['position_size_inr'],
                        'price': signal['price'],
                        'composite_score': signal['composite_score'],
                        'conviction_level': signal['conviction_level'],
                        'entry_date': signal['date']
                    })
            
        except Exception as e:
            logger.warning(f"Position sizing failed: {e}")
        
        return sized_positions
    
    def _execute_entry(self, portfolio_tracker: PortfolioTracker, position: Dict, current_date: datetime) -> Optional[Dict]:
        """Execute position entry"""
        
        try:
            symbol = position['symbol']
            position_size_inr = position['position_size_inr']
            entry_price = position['price']
            
            # Apply slippage and commission
            slippage_bps = self._calculate_slippage(symbol, position_size_inr, 'BUY')
            commission = self._calculate_commission(position_size_inr)
            
            # Adjust entry price for slippage
            adjusted_price = entry_price * (1 + slippage_bps / 10000)
            
            # Calculate quantity
            quantity = int(position_size_inr / adjusted_price)
            actual_cost = quantity * adjusted_price + commission
            
            # Check if we have enough cash
            if actual_cost > portfolio_tracker.cash:
                return None
            
            # Add position to portfolio
            portfolio_tracker.add_position(
                symbol=symbol,
                quantity=quantity,
                entry_price=adjusted_price,
                entry_date=current_date,
                strategy_name='composite_scoring'
            )
            
            # Create trade record
            trade_record = {
                'trade_id': f"{symbol}_{current_date.strftime('%Y%m%d')}_ENTRY",
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'entry_price': adjusted_price,
                'entry_date': current_date,
                'exit_price': None,
                'exit_date': None,
                'pnl': 0,
                'commission': commission,
                'slippage_bps': slippage_bps,
                'composite_score': position['composite_score'],
                'conviction_level': position['conviction_level'],
                'status': 'OPEN'
            }
            
            return trade_record
            
        except Exception as e:
            logger.warning(f"Entry execution failed for {position['symbol']}: {e}")
            return None
    
    def _execute_exit(self, portfolio_tracker: PortfolioTracker, exit_decision: Dict, current_date: datetime) -> Optional[Dict]:
        """Execute position exit"""
        
        try:
            symbol = exit_decision['symbol']
            exit_reason = exit_decision['reason']
            current_price = exit_decision['current_price']
            
            # Get position
            if symbol not in portfolio_tracker.positions:
                return None
            
            position = portfolio_tracker.positions[symbol]
            
            # Apply slippage and commission
            position_size_inr = position['quantity'] * current_price
            slippage_bps = self._calculate_slippage(symbol, position_size_inr, 'SELL')
            commission = self._calculate_commission(position_size_inr)
            
            # Adjust exit price for slippage
            adjusted_price = current_price * (1 - slippage_bps / 10000)
            
            # Calculate P&L
            gross_pnl = (adjusted_price - position['entry_price']) * position['quantity']
            net_pnl = gross_pnl - commission
            
            # Remove position from portfolio
            portfolio_tracker.remove_position(symbol)
            
            # Create trade record
            holding_days = (current_date - position['entry_date']).days
            
            trade_record = {
                'trade_id': f"{symbol}_{position['entry_date'].strftime('%Y%m%d')}_EXIT",
                'symbol': symbol,
                'side': 'SELL',
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'entry_date': position['entry_date'],
                'exit_price': adjusted_price,
                'exit_date': current_date,
                'pnl': net_pnl,
                'pnl_pct': (net_pnl / (position['entry_price'] * position['quantity'])) * 100,
                'commission': commission,
                'slippage_bps': slippage_bps,
                'holding_days': holding_days,
                'exit_reason': exit_reason,
                'composite_score': position.get('composite_score', 0),
                'conviction_level': position.get('conviction_level', 0),
                'status': 'CLOSED'
            }
            
            return trade_record
            
        except Exception as e:
            logger.warning(f"Exit execution failed for {exit_decision['symbol']}: {e}")
            return None
    
    def _check_exit_conditions(self, portfolio_tracker: PortfolioTracker, current_date: datetime) -> List[Dict]:
        """Check exit conditions for all positions"""
        
        exit_decisions = []
        
        for symbol, position in portfolio_tracker.positions.items():
            try:
                # Get current price
                current_price = self._get_current_price(symbol, current_date)
                if current_price is None:
                    continue
                
                # Check various exit conditions
                
                # 1. Time-based exit (30 days)
                holding_days = (current_date - position['entry_date']).days
                if holding_days >= self.trading_config.MAX_HOLDING_DAYS:
                    exit_decisions.append({
                        'symbol': symbol,
                        'reason': 'TIME_EXIT',
                        'current_price': current_price
                    })
                    continue
                
                # 2. Stop loss (simplified - would use ATR-based)
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct <= -0.08:  # 8% stop loss
                    exit_decisions.append({
                        'symbol': symbol,
                        'reason': 'STOP_LOSS',
                        'current_price': current_price
                    })
                    continue
                
                # 3. Profit taking (simplified)
                if pnl_pct >= 0.25:  # 25% profit taking
                    exit_decisions.append({
                        'symbol': symbol,
                        'reason': 'PROFIT_TAKING',
                        'current_price': current_price
                    })
                    continue
                
                # 4. Fundamental deterioration (simplified)
                if np.random.random() < 0.02:  # 2% daily chance
                    exit_decisions.append({
                        'symbol': symbol,
                        'reason': 'FUNDAMENTAL_DETERIORATION',
                        'current_price': current_price
                    })
                    continue
                
            except Exception as e:
                logger.warning(f"Exit condition check failed for {symbol}: {e}")
                continue
        
        return exit_decisions
    
    # Performance calculation methods
    
    def _calculate_performance_metrics(self, simulation_results: Dict) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        equity_curve = simulation_results['equity_curve']
        returns = simulation_results['returns']
        
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0
            }
        
        # Total return
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # Annualized return
        days = len(returns)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else volatility
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_performance_metrics_from_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics from returns series"""
        
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        days = len(returns)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    # Helper methods (placeholder implementations)
    
    def _prepare_universe_data(self, start_date: str, end_date: str, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Prepare universe data for backtesting"""
        
        # This would load actual market data
        # Placeholder implementation with synthetic data
        
        if symbols is None:
            symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'LT', 'ASIANPAINT']
        
        universe_data = {}
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        for symbol in symbols:
            # Generate synthetic OHLCV data
            base_price = np.random.uniform(100, 2000)
            price_series = []
            
            for i in range(len(dates)):
                if i == 0:
                    price = base_price
                else:
                    return_val = np.random.normal(0.0005, 0.02)  # Slight positive drift with 2% volatility
                    price = price_series[-1] * (1 + return_val)
                price_series.append(price)
            
            # Create OHLCV data
            data = pd.DataFrame({
                'open': price_series,
                'high': [p * np.random.uniform(1.0, 1.03) for p in price_series],
                'low': [p * np.random.uniform(0.97, 1.0) for p in price_series],
                'close': price_series,
                'volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
            
            universe_data[symbol] = data
        
        return universe_data
    
    def _update_market_data(self, current_date: datetime, universe_data: Dict):
        """Update market data for current date"""
        # This would update the current market data cache
        pass
    
    def _get_current_price(self, symbol: str, current_date: datetime) -> Optional[float]:
        """Get current price for symbol"""
        # This would get the current price from market data
        return np.random.uniform(100, 200)  # Placeholder
    
    def _calculate_slippage(self, symbol: str, position_size_inr: float, side: str) -> float:
        """Calculate realistic slippage in basis points"""
        
        # Simple slippage model based on position size
        base_slippage = 5  # 5 bps base
        
        # Size impact (square root relationship)
        size_factor = (position_size_inr / 1000000) ** 0.5  # Per million rupees
        size_slippage = size_factor * 10  # Up to 10 bps for large orders
        
        total_slippage = base_slippage + size_slippage
        
        return min(total_slippage, 50)  # Cap at 50 bps
    
    def _calculate_commission(self, position_size_inr: float) -> float:
        """Calculate brokerage commission"""
        
        # Zerodha-like commission structure
        commission_rate = 0.0003  # 0.03% or ₹20 per trade, whichever is lower
        max_commission = 20.0
        
        commission = position_size_inr * commission_rate
        
        return min(commission, max_commission)
    
    # Additional helper methods would go here...
    
    def _calculate_trade_statistics(self, trades_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Calculate trade statistics"""
        
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return {
                'total_trades': len(trades_df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] <= 0]
        
        total_trades = len(closed_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        if len(returns) == 0:
            return {
                'var_95': 0,
                'var_99': 0,
                'beta': 1.0,
                'alpha': 0.0
            }
        
        # Value at Risk
        var_95 = returns.quantile(0.05)  # 5th percentile
        var_99 = returns.quantile(0.01)  # 1st percentile
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'beta': 1.0,  # Would calculate vs benchmark
            'alpha': 0.0   # Would calculate vs benchmark
        }
    
    def _analyze_portfolio_metrics(self, positions_df: pd.DataFrame) -> Dict[str, Union[float, int, Dict]]:
        """Analyze portfolio metrics"""
        
        if len(positions_df) == 0:
            return {
                'avg_positions': 0,
                'max_positions': 0,
                'sector_concentration': {}
            }
        
        avg_positions = positions_df['num_positions'].mean()
        max_positions = positions_df['num_positions'].max()
        
        return {
            'avg_positions': avg_positions,
            'max_positions': max_positions,
            'sector_concentration': {}  # Would calculate actual sector concentration
        }
    
    def _analyze_execution_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze execution metrics"""
        
        if len(trades_df) == 0:
            return {
                'avg_holding_days': 0,
                'avg_slippage_bps': 0,
                'total_commission': 0
            }
        
        closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return {
                'avg_holding_days': 0,
                'avg_slippage_bps': 0,
                'total_commission': trades_df['commission'].sum()
            }
        
        avg_holding_days = closed_trades['holding_days'].mean()
        avg_slippage_bps = closed_trades['slippage_bps'].mean()
        total_commission = trades_df['commission'].sum()
        
        return {
            'avg_holding_days': avg_holding_days,
            'avg_slippage_bps': avg_slippage_bps,
            'total_commission': total_commission
        }
    
    # Additional methods for walk-forward optimization, Monte Carlo, etc. would be implemented here...
    # For brevity, I'm including placeholder stubs
    
    def _create_walk_forward_windows(self, start_date: str, end_date: str, train_months: int, test_months: int) -> List[Tuple]:
        """Create walk-forward optimization windows"""
        # Implementation would create sliding windows
        return [('2023-01-01', '2023-12-31', '2024-01-01', '2024-03-31')]  # Placeholder
    
    def _optimize_parameters(self, start_date: str, end_date: str, param_ranges: Dict, capital: float) -> OptimizationResults:
        """Optimize parameters for given period"""
        # Implementation would use optimization algorithms
        return OptimizationResults(
            best_params={'param1': 0.5},
            best_score=0.15,
            param_sensitivity={},
            optimization_surface=pd.DataFrame(),
            convergence_data=[]
        )
    
    def _run_backtest_with_params(self, start_date: str, end_date: str, params: Dict, capital: float) -> BacktestResults:
        """Run backtest with specific parameters"""
        # Implementation would run backtest with given parameters
        return self.run_full_backtest(start_date, end_date, capital)
    
    def _bootstrap_returns(self, returns: pd.Series) -> pd.Series:
        """Bootstrap resample returns"""
        return returns.sample(n=len(returns), replace=True)
    
    def _block_bootstrap_returns(self, returns: pd.Series, block_size: int) -> pd.Series:
        """Block bootstrap to preserve serial correlation"""
        # Implementation would do block bootstrap
        return returns.sample(n=len(returns), replace=True)
    
    def _random_start_simulation(self, results: BacktestResults) -> pd.Series:
        """Random start date simulation"""
        returns = results.daily_returns
        start_idx = np.random.randint(0, max(1, len(returns) - 252))
        end_idx = min(start_idx + 252, len(returns))
        return returns.iloc[start_idx:end_idx]
    
    def _analyze_parameter_stability(self, params_by_window: List[Dict]) -> Dict:
        """Analyze parameter stability across windows"""
        return {}  # Placeholder
    
    def _calculate_ensemble_parameters(self, params_by_window: List[Dict], stability: Dict) -> Dict:
        """Calculate ensemble parameters"""
        return {'param1': 0.5}  # Placeholder
    
    def _identify_market_regimes(self, market_data: pd.Series) -> Dict[str, pd.Series]:
        """Identify market regimes"""
        # Simplified regime identification
        rolling_return = market_data.rolling(20).mean()
        rolling_vol = market_data.rolling(20).std()
        
        bull_mask = rolling_return > 0.001  # Positive trend
        bear_mask = rolling_return < -0.001  # Negative trend
        high_vol_mask = rolling_vol > rolling_vol.quantile(0.8)
        
        return {
            'bull_market': bull_mask,
            'bear_market': bear_mask,
            'high_volatility': high_vol_mask,
            'sideways': ~(bull_mask | bear_mask)
        }
    
    def _get_nifty_data(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get Nifty 50 benchmark data"""
        # Placeholder - would get actual Nifty data
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        returns = np.random.normal(0.0005, 0.015, len(dates))  # Simulate market returns
        return pd.Series(returns, index=dates)
    
    def _calculate_regime_diversification(self, regime_performance: Dict) -> float:
        """Calculate regime diversification benefit"""
        if not regime_performance:
            return 0.0
        
        returns = [perf['annual_return'] for perf in regime_performance.values()]
        return 1 - (np.std(returns) / np.mean(returns)) if returns and np.mean(returns) > 0 else 0
