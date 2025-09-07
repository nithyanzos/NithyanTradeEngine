# Nifty Universe Trading System
## Institutional-Grade Algorithmic Trading Platform

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Trading](https://img.shields.io/badge/trading-algorithmic-red.svg)

A sophisticated algorithmic trading system designed for the Indian Nifty universe (large, mid, small cap stocks) with institutional-grade risk management and capital preservation principles.

## 🎯 Core Philosophy

### Capital Management (Non-Negotiable)
- **NEVER exceed 50% capital deployment** - Always maintain 50% cash reserve
- **Position limits**: 3-8% of deployable capital per position (NOT total capital)
- **Maximum 15 positions**, minimum 10 positions at any time
- **Sector limits**: Maximum 30% allocation to any single sector
- **Daily loss limit**: Stop all trading if portfolio loses >2% in single day
- **Maximum drawdown**: Emergency stop if portfolio drawdown exceeds 8%

### Trade Selection Philosophy
- **Quality over quantity**: Filter ~300 Nifty stocks to top 10-15 opportunities only
- **Composite scoring**: 25% fundamental + 30% technical + 25% quantitative + 20% macro
- **Minimum score threshold**: Only trade opportunities with composite score ≥ 0.70
- **Risk-adjusted sizing**: Position size based on volatility (ATR), conviction, and correlation

## 🏗️ System Architecture

```
nifty-trading-system/
├── src/
│   ├── config/                 # Configuration and API connections
│   │   ├── trading_config.py   # Core trading parameters
│   │   ├── api_config.py       # Zerodha/Sonar API integration
│   │   └── database_config.py  # Database models and connections
│   ├── analytics/              # Multi-factor analysis engines
│   │   ├── fundamental_analyzer.py    # Financial statement analysis
│   │   ├── technical_analyzer.py     # Technical indicators & patterns
│   │   ├── quantitative_engine.py    # Momentum/Quality/Value factors
│   │   └── macro_analyzer.py         # Sentiment & macro analysis
│   ├── strategy/               # Core trading strategy
│   │   ├── trade_filter.py     # Multi-factor signal filtering
│   │   ├── position_manager.py # Conservative position sizing
│   │   ├── risk_manager.py     # Multi-layered risk management
│   │   └── strategy_executor.py # Main strategy orchestrator
│   ├── trading/                # Order execution and portfolio management
│   │   ├── order_executor.py   # Smart order execution algorithms
│   │   └── portfolio_tracker.py # Real-time portfolio tracking
│   ├── backtesting/            # Backtesting and optimization
│   │   ├── backtest_engine.py  # Walk-forward optimization
│   │   └── performance_analyzer.py # Performance analytics
│   └── main.py                 # Application entry point
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL (for production) or SQLite (for development)
- Redis (for caching)
- TA-Lib technical analysis library

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd NithyanTradeEngine
```

2. **Run the automated setup**
```bash
./run.sh backtest 2023-01-01 2024-01-01
```

3. **Manual setup (alternative)**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run backtest
cd src
python main.py --mode backtest --start-date 2023-01-01 --end-date 2024-01-01
```

## 🎮 Usage Modes

### 1. Backtesting Mode
Test the strategy on historical data with realistic transaction costs:

```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2024-01-01 --capital 5000000
```

### 2. Walk-Forward Optimization
Optimize parameters with out-of-sample validation:

```bash
python main.py --mode optimize --start-date 2022-01-01 --end-date 2024-01-01 --train-months 12 --test-months 3
```

### 3. Live Trading Mode
Run live trading (requires API credentials):

```bash
python main.py --mode live
```

## 📊 Strategy Components

### Multi-Factor Scoring System

#### 1. Fundamental Analysis (25% Weight)
- **Profitability**: ROE > 15%, Operating margin > 10%, ROIC > 12%
- **Financial Health**: Debt/Equity < 0.5, Interest coverage > 5x, Current ratio > 1.2
- **Growth**: Revenue CAGR > 10%, EPS growth consistency, Book value growth
- **Governance**: Promoter holding > 50%, Institutional growth

#### 2. Technical Analysis (30% Weight)
- **Trend**: Moving average systems, trend strength indicators
- **Momentum**: RSI, MACD, Rate of Change
- **Volume**: Volume-price analysis, accumulation/distribution
- **Support/Resistance**: Key level identification and breakouts

#### 3. Quantitative Factors (25% Weight)
- **Momentum**: 3-month, 6-month, 12-month price momentum
- **Quality**: ROE stability, Earnings predictability, Balance sheet strength
- **Value**: P/E relative to growth, P/B vs ROE, EV/EBITDA analysis
- **Low Volatility**: Risk-adjusted returns, volatility ranking

#### 4. Macro Sentiment Analysis (20% Weight)
- **News Sentiment**: AI-powered news analysis using Sonar API
- **Sector Rotation**: Relative sector performance analysis
- **Market Regime**: Bull/bear/sideways market identification
- **Economic Indicators**: Interest rates, inflation, GDP growth impact

### Risk Management Layers

#### 1. Position-Level Risk
- **ATR-based stops**: Initial stop at 2.5x ATR from entry
- **Trailing stops**: 1.5x ATR trailing for profitable positions
- **Time-based exits**: Maximum 30-day holding period
- **Fundamental deterioration**: Exit if score drops >15%

#### 2. Portfolio-Level Risk
- **Capital utilization**: Hard 50% limit on deployed capital
- **Position concentration**: Maximum 8% per position, 30% per sector
- **Correlation limits**: Avoid highly correlated positions
- **Daily loss limits**: Stop trading if portfolio loses >2% in one day

#### 3. System-Level Risk
- **Maximum drawdown**: Emergency liquidation if drawdown exceeds 8%
- **Market regime detection**: Reduce exposure in volatile conditions
- **Liquidity management**: Maintain minimum cash for margin calls
- **API failure handling**: Graceful degradation and retry mechanisms

## 📈 Performance Targets

### Return Objectives
- **Annual Return**: 18-25% target (net of all costs)
- **Sharpe Ratio**: >1.8 target (risk-adjusted returns)
- **Maximum Drawdown**: <8% (hard stop)
- **Win Rate**: 55-65% target

### Risk Constraints
- **Daily VaR**: 95% confidence, 1-day horizon
- **Capital at Risk**: Never exceed 50% of total capital
- **Sector Concentration**: Maximum 30% in any sector
- **Position Turnover**: <200% annually (minimize transaction costs)

## 🛠️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Trading Configuration
TOTAL_CAPITAL=5000000
MAX_CAPITAL_UTILIZATION=0.50
DAILY_LOSS_LIMIT=-0.02
MAX_DRAWDOWN_LIMIT=-0.08

# API Credentials
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token
SONAR_API_KEY=your_sonar_key

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/trading
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
```

### Key Parameters
The system enforces these non-negotiable constraints:

```python
# Capital Management (NEVER change these)
MAX_CAPITAL_UTILIZATION = 0.50      # 50% maximum deployment
CASH_RESERVE_RATIO = 0.50            # Always maintain 50% cash
MAX_POSITION_SIZE = 0.08             # 8% of deployable capital per position
DAILY_LOSS_LIMIT = -0.02             # 2% daily loss triggers full stop
MAX_DRAWDOWN_LIMIT = -0.08           # 8% drawdown triggers emergency stop

# Position Management
MAX_POSITIONS = 15                   # Maximum 15 positions
MIN_POSITIONS = 10                   # Minimum 10 positions for diversification
MAX_SECTOR_ALLOCATION = 0.30         # 30% maximum per sector
COMPOSITE_SCORE_THRESHOLD = 0.70     # Minimum score to trade
```

## 🧪 Testing and Validation

### Backtesting Features
- **Walk-forward optimization**: Prevents overfitting with out-of-sample testing
- **Realistic transaction costs**: Includes slippage, commission, market impact
- **Regime analysis**: Performance across bull/bear/sideways markets
- **Monte Carlo validation**: Confidence intervals for key metrics

### Test Coverage
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run backtests
pytest tests/backtests/
```

## 📋 API Requirements

### Zerodha Kite Connect
- **Data**: Real-time quotes, historical data, market depth
- **Trading**: Order placement, modification, cancellation
- **Portfolio**: Position tracking, P&L monitoring
- **Rate Limits**: 3 requests per second (built-in throttling)

### Sonar API (Optional)
- **News Sentiment**: AI-powered news analysis
- **Social Sentiment**: Social media sentiment tracking
- **Alternative Data**: Satellite data, economic indicators

## 📊 Monitoring and Alerts

### Real-Time Monitoring
- **Portfolio P&L**: Live tracking of daily/total P&L
- **Risk Metrics**: VaR, drawdown, concentration monitoring
- **Order Status**: Execution monitoring, slippage tracking
- **System Health**: API connectivity, database status

### Alert System
- **Risk Alerts**: Position limits, sector concentration, daily losses
- **Market Alerts**: High volatility, gap openings, circuit breakers
- **System Alerts**: API failures, database issues, execution errors
- **Performance Alerts**: Drawdown warnings, target achievement

## 🔐 Security and Compliance

### Data Security
- **API Key Management**: Secure storage and rotation of API credentials
- **Database Encryption**: Encrypted storage of sensitive trading data
- **Access Control**: Role-based access to different system components
- **Audit Logging**: Complete audit trail of all trading decisions

### Risk Controls
- **Pre-trade Validation**: Multiple validation layers before order placement
- **Real-time Monitoring**: Continuous risk monitoring during market hours
- **Circuit Breakers**: Automatic stops for various risk scenarios
- **Manual Override**: Emergency controls for manual intervention

## 📚 Documentation

### Code Documentation
- **Docstrings**: Comprehensive documentation for all functions and classes
- **Type Hints**: Full type annotations for better code reliability
- **Examples**: Usage examples for all major components
- **Architecture**: Detailed system architecture documentation

### Trading Documentation
- **Strategy Logic**: Detailed explanation of all trading rules
- **Risk Management**: Comprehensive risk management procedures
- **Performance Analysis**: Methods for analyzing strategy performance
- **Optimization**: Guidelines for parameter optimization

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8, use Black formatter
2. **Testing**: Write tests for all new features
3. **Documentation**: Update documentation for any changes
4. **Risk Management**: Never compromise on risk management principles

### Risk Management Rules (Non-Negotiable)
1. **Never exceed 50% capital deployment**
2. **Always maintain position size limits**
3. **Implement all stop-loss mechanisms**
4. **Preserve capital above all else**

## ⚠️ Disclaimer

This trading system is for educational and research purposes. Real trading involves substantial risk of loss. The authors and contributors are not responsible for any trading losses incurred using this system.

### Important Warnings
- **Past performance does not guarantee future results**
- **Trading involves risk of substantial loss**
- **Test thoroughly before deploying real capital**
- **Maintain proper risk management at all times**
- **Consider regulatory requirements in your jurisdiction**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Indian Stock Exchanges**: NSE, BSE for market data
- **Zerodha**: For providing robust trading APIs
- **Open Source Community**: For excellent Python libraries
- **Academic Research**: Quantitative finance research papers

---

**Built with ❤️ for the Indian capital markets**

*"In trading, capital preservation is not just a strategy—it's survival."*