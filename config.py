"""
Configuration settings for the Quant Alpha system.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT_DIR / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')  # Default to us-east-1
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'quant-alpha-data')

# Trading Configuration
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech
    "JPM", "BAC", "GS", "MS", "WFC",          # Finance
    "XOM", "CVX", "COP", "SLB", "EOG",        # Energy
    "JNJ", "PFE", "MRK", "ABBV", "LLY",       # Healthcare
    "SPY"                                     # Benchmark
]

# Factor Parameters
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
VOLATILITY_WINDOW = 21
BETA_WINDOW = 252
RSI_WINDOW = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SMA_FAST = 20
SMA_SLOW = 50
MOMENTUM_WINDOW = 20

# Regime Detection Parameters
VIX_HIGH_THRESHOLD = 25.0
ROLLING_STD_HIGH_THRESHOLD = 0.02
TREND_WINDOW = 60
TREND_THRESHOLD = 0.05
VOL_WINDOW = 21
VOL_LOOKBACK = 252
VOL_PERCENTILE_LOW = 0.33
VOL_PERCENTILE_HIGH = 0.67
SMOOTH_WINDOW = 5

# Backtest Parameters
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.001  # 0.1% per trade
REBALANCE_FREQ = "5D"  # Rebalance every 5 days
TOP_N_STOCKS = 5
MAX_POSITION_SIZE = 0.2
STOP_LOSS_PCT = 0.5
CASH_BUFFER_PCT = 0.1
WEEKLY_REBALANCE = True

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Dashboard Configuration
DASHBOARD_PORT = 8501
DASHBOARD_THEME = 'light'

# Default date range for analysis
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)  # 1 year of data

# Free Tier Limits
FREE_TIER_LIMITS = {
    's3': {
        'storage_gb': 5,
        'requests': 20000
    },
    'lambda': {
        'invocations': 1000000,
        'compute_seconds': 400000
    },
    'api_gateway': {
        'requests': 1000000,
        'data_transfer_gb': 1
    },
    'ec2': {
        'hours_per_month': 750,
        'storage_gb': 30
    }
}

# Logging configuration
LOG_FILE = RESULTS_DIR / f'quant_alpha_{datetime.now().strftime("%Y%m%d")}.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO' 