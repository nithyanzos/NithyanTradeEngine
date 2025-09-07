#!/bin/bash

# Nifty Universe Trading System - Startup Script

echo "🏛️  Nifty Universe Trading System - Institutional Grade"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python Version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
echo "⚙️  Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create logs directory
mkdir -p logs

# Check command line arguments
MODE=${1:-backtest}
START_DATE=${2:-2023-01-01}
END_DATE=${3:-2024-01-01}

echo "🚀 Starting trading system in $MODE mode..."
echo "📅 Period: $START_DATE to $END_DATE"

# Run the application
cd src
python main.py --mode $MODE --start-date $START_DATE --end-date $END_DATE

echo "✅ Trading system execution completed"
