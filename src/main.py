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

logger = logging.getLogger(__name__)

class TradingApplication:
    """
    Main application controller for the institutional trading system
    """
    
    def __init__(self):
        self.config = get_trading_config()
        self.logger = setup_logging(self.config.LOG_LEVEL)
        
    async def initialize_system(self):
        """Initialize all system components"""
        try:
            self.logger.info("🚀 Initializing Nifty Universe Trading System...")
            
            # Validate configuration
            self.config.validate_config()
            
            # Display system configuration
            self.display_system_info()
            
            self.logger.info("✅ System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def display_system_info(self):
        """Display comprehensive system information"""
        
        self.logger.info("=" * 80)
        self.logger.info("🏛️  NIFTY UNIVERSE INSTITUTIONAL TRADING SYSTEM")
        self.logger.info("=" * 80)
        
        # Capital Management Info
        self.logger.info("💰 CAPITAL MANAGEMENT:")
        self.logger.info(f"   Total Capital: ₹{self.config.TOTAL_CAPITAL:,.0f}")
        self.logger.info(f"   Deployable Capital: ₹{self.config.deployable_capital:,.0f} (50%)")
        self.logger.info(f"   Cash Reserve: ₹{self.config.TOTAL_CAPITAL - self.config.deployable_capital:,.0f} (50%)")
        
        # Position Sizing Info
        self.logger.info("📊 POSITION SIZING:")
        self.logger.info(f"   Max Positions: {self.config.MAX_POSITIONS}")
        self.logger.info(f"   Min Positions: {self.config.MIN_POSITIONS}")
        self.logger.info(f"   Position Size Range: ₹{self.config.min_position_value:,.0f} - ₹{self.config.max_position_value:,.0f}")
        
        # Risk Management Info
        self.logger.info("🛡️  RISK MANAGEMENT:")
        self.logger.info(f"   Daily Loss Limit: {abs(self.config.DAILY_LOSS_LIMIT)*100:.1f}%")
        self.logger.info(f"   Max Drawdown Limit: {abs(self.config.MAX_DRAWDOWN_LIMIT)*100:.1f}%")
        self.logger.info(f"   Max Sector Allocation: {self.config.MAX_SECTOR_ALLOCATION*100:.1f}%")
        
        # Scoring Weights
        self.logger.info("⚖️  SCORING WEIGHTS:")
        self.logger.info(f"   Fundamental: {self.config.FUNDAMENTAL_WEIGHT*100:.0f}%")
        self.logger.info(f"   Technical: {self.config.TECHNICAL_WEIGHT*100:.0f}%")
        self.logger.info(f"   Quantitative: {self.config.QUANTITATIVE_WEIGHT*100:.0f}%")
        self.logger.info(f"   Macro: {self.config.MACRO_WEIGHT*100:.0f}%")
        
        # API Status
        self.logger.info("🔌 API CONFIGURATION:")
        self.logger.info(f"   Zerodha API: {'✅ Configured' if self.config.ZERODHA_API_KEY else '❌ Not Configured'}")
        self.logger.info(f"   Sonar API: {'✅ Configured' if self.config.SONAR_API_KEY else '❌ Not Configured'}")
        
        self.logger.info("=" * 80)
    
    async def run_system_validation(self):
        """Run comprehensive system validation"""
        
        self.logger.info("🔍 Running system validation...")
        
        validation_results = {
            'config_validation': False,
            'database_connection': False,
            'api_connectivity': False,
            'risk_limits': False
        }
        
        # 1. Configuration Validation
        try:
            self.config.validate_config()
            validation_results['config_validation'] = True
            self.logger.info("✅ Configuration validation passed")
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
        
        # 2. Database Connection Test
        try:
            from config.database_config import get_database_manager
            db_manager = get_database_manager()
            validation_results['database_connection'] = True
            self.logger.info("✅ Database connection successful")
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
        
        # 3. Risk Limits Validation
        try:
            risk_limits = self.config.get_risk_limits()
            validation_results['risk_limits'] = True
            self.logger.info("✅ Risk limits validation passed")
            for limit_name, limit_value in risk_limits.items():
                self.logger.info(f"   {limit_name}: ₹{limit_value:,.0f}")
        except Exception as e:
            self.logger.error(f"❌ Risk limits validation failed: {e}")
        
        # Summary
        passed_validations = sum(validation_results.values())
        total_validations = len(validation_results)
        
        self.logger.info(f"📋 Validation Summary: {passed_validations}/{total_validations} checks passed")
        
        if passed_validations == total_validations:
            self.logger.info("🎉 All validations passed! System ready for operation.")
            return True
        else:
            self.logger.warning("⚠️  Some validations failed. Please review configuration.")
            return False
    
    async def run_demo_mode(self):
        """Run system in demonstration mode"""
        
        self.logger.info("🎭 Running in demonstration mode...")
        
        # Simulate universe filtering
        self.logger.info("📈 Simulating trade filtering...")
        await asyncio.sleep(2)
        
        demo_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        demo_scores = [0.85, 0.78, 0.82, 0.76, 0.79]
        
        self.logger.info("🎯 Top Scoring Opportunities (Demo):")
        for stock, score in zip(demo_stocks, demo_scores):
            position_size = self.config.deployable_capital * 0.06  # 6% allocation
            self.logger.info(f"   {stock}: Score {score:.2f} | Position: ₹{position_size:,.0f}")
        
        # Simulate position sizing
        self.logger.info("💼 Position Sizing Analysis:")
        total_allocation = sum(self.config.deployable_capital * 0.06 for _ in demo_stocks)
        utilization = (total_allocation / self.config.TOTAL_CAPITAL) * 100
        
        self.logger.info(f"   Total Allocation: ₹{total_allocation:,.0f}")
        self.logger.info(f"   Capital Utilization: {utilization:.1f}%")
        self.logger.info(f"   Remaining Cash: ₹{self.config.TOTAL_CAPITAL - total_allocation:,.0f}")
        
        # Risk analysis
        self.logger.info("🛡️  Risk Analysis:")
        max_daily_loss = self.config.TOTAL_CAPITAL * abs(self.config.DAILY_LOSS_LIMIT)
        self.logger.info(f"   Daily Loss Limit: ₹{max_daily_loss:,.0f}")
        self.logger.info(f"   Stop Loss per position: ~2.5x ATR")
        
        self.logger.info("✅ Demo simulation completed successfully!")

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Nifty Universe Institutional Trading System')
    parser.add_argument('--mode', choices=['validate', 'demo', 'backtest'], 
                       default='validate', help='Operating mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize application
    app = TradingApplication()
    
    # Set log level
    app.logger.setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize system
        if not await app.initialize_system():
            sys.exit(1)
        
        # Run based on mode
        if args.mode == 'validate':
            success = await app.run_system_validation()
            if not success:
                sys.exit(1)
                
        elif args.mode == 'demo':
            await app.run_demo_mode()
            
        elif args.mode == 'backtest':
            app.logger.info("📊 Backtesting mode not yet implemented")
            app.logger.info("   Coming soon: Comprehensive backtesting framework")
        
        app.logger.info("🏁 Application completed successfully")
        
    except KeyboardInterrupt:
        app.logger.info("⏹️  Application interrupted by user")
    except Exception as e:
        app.logger.error(f"💥 Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
    
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
