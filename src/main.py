"""
Main Application Entry Point
Nifty Universe Institutional Trading System
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.trading_config import get_trading_config, setup_logging
from strategy.strategy_executor import InstitutionalTradingEngine
from backtesting.backtest_engine import ComprehensiveBacktester

logger = logging.getLogger(__name__)

class TradingApplication:
    """
    Main application controller for the institutional trading system
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.trading_engine = None
        self.backtester = None
        
        # Setup logging
        setup_logging()
        
        logger.info("🏛️ Nifty Universe Trading System - Institutional Grade")
        logger.info(f"💰 Total Capital: ₹{self.config.TOTAL_CAPITAL:,.0f}")
        logger.info(f"📊 Max Capital Deployment: {self.config.MAX_CAPITAL_UTILIZATION*100:.0f}%")
        logger.info(f"🎯 Max Positions: {self.config.MAX_POSITIONS}")
        logger.info(f"⚠️  Daily Loss Limit: {abs(self.config.DAILY_LOSS_LIMIT)*100:.0f}%")
        logger.info(f"🛡️  Max Drawdown Limit: {abs(self.config.MAX_DRAWDOWN_LIMIT)*100:.0f}%")
    
    async def run_live_trading(self):
        """
        Run live trading session
        """
        
        logger.info("🚀 Starting Live Trading Mode")
        
        try:
            # Initialize trading engine
            self.trading_engine = InstitutionalTradingEngine()
            
            # Start trading session
            session_result = await self.trading_engine.start_trading_session()
            
            if not session_result['success']:
                logger.error(f"Failed to start trading session: {session_result['message']}")
                return
            
            logger.info(f"✅ Trading session started: {session_result['session_id']}")
            logger.info(f"📊 Portfolio Value: ₹{session_result['portfolio_value']:,.0f}")
            logger.info(f"💵 Cash Available: ₹{session_result['cash_available']:,.0f}")
            logger.info(f"📈 Current Positions: {session_result['positions_count']}")
            
            # Main trading loop
            cycle_count = 0
            
            while True:
                try:
                    # Check if we should continue trading
                    status = self.trading_engine.get_trading_status()
                    
                    if not status['session_active'] or status['emergency_stop_active']:
                        logger.warning("Trading session inactive or emergency stop active")
                        break
                    
                    if not status['market_open']:
                        logger.info("Market closed - waiting for next session")
                        await asyncio.sleep(300)  # Check every 5 minutes
                        continue
                    
                    # Run trading cycle
                    cycle_count += 1
                    logger.info(f"🔄 Starting trading cycle #{cycle_count}")
                    
                    cycle_result = await self.trading_engine.run_trading_cycle()
                    
                    if cycle_result['cycle_completed']:
                        logger.info(f"✅ Cycle #{cycle_count} completed in {cycle_result['cycle_duration_seconds']:.1f}s")
                        logger.info(f"📡 Signals: {cycle_result['signals_generated']} → {cycle_result['filtered_signals']} filtered")
                        logger.info(f"⚡ Executions: {cycle_result['successful_executions']}/{cycle_result['orders_placed']}")
                        logger.info(f"💰 Portfolio: ₹{cycle_result['portfolio_value']:,.0f} (₹{cycle_result['daily_pnl']:+,.0f} today)")
                        
                        if cycle_result['risk_alerts'] > 0:
                            logger.warning(f"⚠️  Risk Alerts: {cycle_result['risk_alerts']}")
                    
                    else:
                        logger.warning(f"⚠️  Cycle #{cycle_count} incomplete: {cycle_result.get('message', 'Unknown error')}")
                        
                        if cycle_result.get('emergency_stop'):
                            logger.critical("🚨 Emergency stop triggered - halting trading")
                            break
                    
                    # Wait before next cycle (5 minutes default)
                    await asyncio.sleep(300)
                    
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal - stopping trading")
                    break
                    
                except Exception as e:
                    logger.error(f"Trading cycle error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    continue
            
            # Stop trading session
            stop_result = await self.trading_engine.stop_trading_session()
            
            if stop_result['success']:
                logger.info("✅ Trading session stopped successfully")
                logger.info(f"📊 Session Report: {stop_result.get('session_report', {})}")
            else:
                logger.error(f"Failed to stop trading session: {stop_result.get('message', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Live trading failed: {e}")
            raise
    
    async def run_backtest(self, 
                          start_date: str = "2023-01-01",
                          end_date: str = "2024-01-01",
                          initial_capital: float = None):
        """
        Run comprehensive backtest
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital (defaults to config)
        """
        
        logger.info(f"📊 Starting Backtest Mode: {start_date} to {end_date}")
        
        try:
            # Initialize backtester
            self.backtester = ComprehensiveBacktester()
            
            # Set initial capital
            if initial_capital is None:
                initial_capital = self.config.TOTAL_CAPITAL
            
            logger.info(f"💰 Backtest Capital: ₹{initial_capital:,.0f}")
            
            # Run backtest
            results = self.backtester.run_full_backtest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            
            # Display results
            self._display_backtest_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def run_walk_forward_optimization(self,
                                          start_date: str = "2022-01-01",
                                          end_date: str = "2024-01-01",
                                          train_months: int = 12,
                                          test_months: int = 3):
        """
        Run walk-forward optimization
        """
        
        logger.info(f"🔄 Starting Walk-Forward Optimization: {start_date} to {end_date}")
        logger.info(f"📅 Training: {train_months} months, Testing: {test_months} months")
        
        try:
            # Initialize backtester
            self.backtester = ComprehensiveBacktester()
            
            # Define parameter ranges for optimization
            param_ranges = {
                'composite_score_threshold': [0.65, 0.70, 0.75, 0.80],
                'fundamental_weight': [0.20, 0.25, 0.30],
                'technical_weight': [0.25, 0.30, 0.35],
                'quantitative_weight': [0.20, 0.25, 0.30],
                'macro_weight': [0.15, 0.20, 0.25],
                'max_position_size': [0.06, 0.08, 0.10],
                'atr_stop_multiplier': [2.0, 2.5, 3.0]
            }
            
            # Run walk-forward optimization
            results, optimization = await self.backtester.walk_forward_optimization(
                start_date=start_date,
                end_date=end_date,
                param_ranges=param_ranges,
                initial_capital=self.config.TOTAL_CAPITAL,
                train_months=train_months,
                test_months=test_months
            )
            
            # Display results
            self._display_optimization_results(results, optimization)
            
            return results, optimization
            
        except Exception as e:
            logger.error(f"Walk-forward optimization failed: {e}")
            raise
    
    def _display_backtest_results(self, results):
        """Display comprehensive backtest results"""
        
        logger.info("="*80)
        logger.info("📊 BACKTEST RESULTS SUMMARY")
        logger.info("="*80)
        
        # Performance metrics
        logger.info(f"📈 Total Return: {results.total_return*100:+.2f}%")
        logger.info(f"📅 Annual Return: {results.annual_return*100:+.2f}%")
        logger.info(f"📊 Volatility: {results.volatility*100:.2f}%")
        logger.info(f"⭐ Sharpe Ratio: {results.sharpe_ratio:.3f}")
        logger.info(f"📉 Max Drawdown: {results.max_drawdown*100:.2f}%")
        logger.info(f"🎯 Calmar Ratio: {results.calmar_ratio:.3f}")
        
        # Trade statistics
        logger.info(f"🔢 Total Trades: {results.total_trades}")
        logger.info(f"✅ Win Rate: {results.win_rate*100:.1f}%")
        logger.info(f"💰 Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"📊 Avg Win: ₹{results.avg_win:,.0f}")
        logger.info(f"📉 Avg Loss: ₹{results.avg_loss:,.0f}")
        
        # Portfolio metrics
        logger.info(f"📈 Avg Positions: {results.avg_positions:.1f}")
        logger.info(f"📅 Avg Holding: {results.avg_holding_days:.1f} days")
        logger.info(f"💸 Total Commission: ₹{results.total_commission:,.0f}")
        logger.info(f"⚡ Avg Slippage: {results.avg_slippage_bps:.1f} bps")
        
        # Risk metrics
        logger.info(f"⚠️  VaR 95%: {results.var_95*100:.2f}%")
        logger.info(f"🚨 VaR 99%: {results.var_99*100:.2f}%")
        
        # Walk-forward metrics (if available)
        if results.oos_return != 0:
            logger.info(f"🔄 Out-of-Sample Return: {results.oos_return:.2f}%")
            logger.info(f"⚖️  Stability Ratio: {results.stability_ratio:.3f}")
        
        logger.info("="*80)
    
    def _display_optimization_results(self, results, optimization):
        """Display optimization results"""
        
        logger.info("="*80)
        logger.info("🔄 WALK-FORWARD OPTIMIZATION RESULTS")
        logger.info("="*80)
        
        # Best parameters
        logger.info("🎯 OPTIMAL PARAMETERS:")
        for param, value in optimization.best_params.items():
            logger.info(f"   {param}: {value}")
        
        logger.info(f"⭐ Best Score: {optimization.best_score:.2f}%")
        logger.info(f"🔄 Out-of-Sample Return: {results.oos_return:.2f}%")
        logger.info(f"📊 Out-of-Sample Sharpe: {results.oos_sharpe:.3f}")
        logger.info(f"⚖️  Stability Ratio: {results.stability_ratio:.3f}")
        
        # Overall performance
        logger.info(f"📈 Final Annual Return: {results.annual_return*100:.2f}%")
        logger.info(f"⭐ Final Sharpe Ratio: {results.sharpe_ratio:.3f}")
        logger.info(f"📉 Final Max Drawdown: {results.max_drawdown*100:.2f}%")
        
        logger.info("="*80)


async def main():
    """Main application entry point"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nifty Universe Trading System')
    parser.add_argument('--mode', choices=['live', 'backtest', 'optimize'], 
                       default='backtest', help='Trading mode')
    parser.add_argument('--start-date', default='2023-01-01', 
                       help='Start date for backtest/optimization (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01',
                       help='End date for backtest/optimization (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=None,
                       help='Initial capital (defaults to config)')
    parser.add_argument('--train-months', type=int, default=12,
                       help='Training months for optimization')
    parser.add_argument('--test-months', type=int, default=3,
                       help='Testing months for optimization')
    
    args = parser.parse_args()
    
    # Initialize application
    app = TradingApplication()
    
    try:
        if args.mode == 'live':
            await app.run_live_trading()
        
        elif args.mode == 'backtest':
            await app.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.capital
            )
        
        elif args.mode == 'optimize':
            await app.run_walk_forward_optimization(
                start_date=args.start_date,
                end_date=args.end_date,
                train_months=args.train_months,
                test_months=args.test_months
            )
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
    
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main())
