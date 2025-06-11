# Quant Alpha AWS

A robust, multi-factor backtesting system for quantitative trading strategies with AWS integration.

## Overview

This system implements a multi-factor alpha strategy generator and backtesting engine. It processes historical market data, computes various alpha factors, combines them into a composite signal, and simulates trading based on that signal.

## Features

- **Data Fetching**: Historical market data from Yahoo Finance
- **Alpha Signal Generation**: Multiple factors including RSI, SMA crossover, volatility, momentum, liquidity, gap, price percentile, and rate of change (ROC)
- **Backtesting Engine**: Flexible backtesting with slippage, commissions, and position sizing
- **Strategy Optimization**: Bayesian optimization of strategy parameters to maximize return and Sharpe ratio
- **AWS Integration**: S3 integration for storing data and results
- **Visualization**: Performance metrics and position allocation over time
- **Enhanced Features**:
  - **Market Regime Detection**: Automatically adapts to different market conditions
  - **Machine Learning Signal Combining**: Uses ridge regression and LightGBM to combine signals
  - **Dynamic Position Constraints**: Sector-based risk management and volatility targeting
  - **Performance Monitoring**: Real-time performance tracking with automatic de-risking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quant-alpha-aws.git
cd quant-alpha-aws
```

2. Create a Python environment (recommended Python 3.9+):
```bash
conda create -n quant-alpha python=3.9
conda activate quant-alpha
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Set up environment variables for AWS (optional):
```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_S3_BUCKET=your-bucket-name
```
   Alternatively, create a `.env` file with these variables.

## Usage

### Basic Backtesting

Run a basic backtest with default parameters:

```bash
python main.py --start_date 2023-01-01 --end_date 2023-12-31 --plot
```

### Enhanced Backtesting

Run a backtest with enhanced features:

```bash
python run_enhanced_backtest.py --data ./data/raw/market_data.csv --config ./configs/enhanced_backtest_config.json --start-date 2023-01-01 --end-date 2023-12-31 --plot
```

### Strategy Optimization

Optimize the strategy parameters:

```bash
python main.py --optimize --start_date 2023-01-01 --end_date 2023-12-31 --n_calls 20 --plot
```

This will perform Bayesian optimization to find the best combination of:
- Factor parameters (e.g., RSI window, SMA periods)
- Factor weights for the composite signal
- Portfolio construction parameters (e.g., rebalance frequency, position sizing)

### Running with Optimized Parameters

After optimization, use the best parameters for a backtest:

```bash
python main.py --start_date 2023-01-01 --end_date 2023-12-31 --use_best_params --best_params_file ./results/optimization_YYYYMMDD_HHMMSS/optimization_results.json --plot
```

### Command Line Arguments

- `--start_date`: Start date for data and backtest (YYYY-MM-DD)
- `--end_date`: End date for data and backtest (YYYY-MM-DD)
- `--optimize`: Run strategy optimization
- `--n_calls`: Number of optimization iterations (with --optimize)
- `--n_jobs`: Number of parallel jobs for optimization (-1 for all cores)
- `--cv_folds`: Number of cross-validation folds for optimization
- `--use_best_params`: Use best parameters from previous optimization
- `--best_params_file`: Path to best parameters file
- `--plot`: Generate and save plots
- `--save_to_s3`: Save results to S3 bucket

### Enhanced Backtest Arguments

- `--data`: Path to market data CSV
- `--signals`: Path to signals data CSV
- `--benchmark`: Path to benchmark data CSV
- `--config`: Path to configuration JSON
- `--output`: Output directory
- `--s3-bucket`: S3 bucket for storing results
- `--start-date`: Start date for backtest (YYYY-MM-DD)
- `--end-date`: End date for backtest (YYYY-MM-DD)
- `--plot`: Generate performance plots

## Directory Structure

- `aws/`: AWS integration utilities
- `backtest/`: Backtesting engine
- `data/`: Market data fetching and processing
- `factors/`: Alpha signal generation
- `utils/`: Utility functions and optimization
  - `market_regime.py`: Market regime detection
  - `signal_combiner.py`: Advanced signal combination
  - `position_constraints.py`: Position risk management
  - `performance_monitor.py`: Performance tracking
- `results/`: Backtest and optimization results
- `dashboard/`: Visualization dashboard (work in progress)
- `configs/`: Configuration files for enhanced backtests

## Example Workflow

1. **Data Collection**:
   ```
   python main.py --start_date 2023-01-01 --end_date 2023-12-31
   ```

2. **Strategy Optimization**:
   ```
   python main.py --optimize --start_date 2023-01-01 --end_date 2023-12-31 --n_calls 30 --cv_folds 3 --plot
   ```

3. **Production Backtesting**:
   ```
   python main.py --start_date 2023-01-01 --end_date 2023-12-31 --use_best_params --best_params_file ./results/optimization_YYYYMMDD_HHMMSS/optimization_results.json --plot
   ```

4. **Enhanced Backtesting**:
   ```
   python run_enhanced_backtest.py --data ./data/raw/market_data.csv --config ./configs/enhanced_backtest_config.json --start-date 2023-01-01 --end-date 2023-12-31 --plot
   ```

## Implemented Alpha Factors

- **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and change of price movements
- **SMA Crossover**: Signal based on fast and slow moving average crossovers
- **Volatility**: Inverse volatility signal (favors low volatility stocks)
- **Momentum**: Simple price momentum over a specified lookback period
- **Liquidity**: Dollar volume-based liquidity measure
- **Gap**: Overnight price gap signal (mean-reversion approach)
- **Price Percentile**: Current price relative to 52-week range (contrarian)
- **Rate of Change (ROC)**: Price rate of change over a specified lookback period

## Enhanced Features

### Market Regime Detection

The system detects and adapts to different market regimes:

- **Trend Classification**: Bullish, Bearish, or Sideways
- **Volatility Classification**: Low, Medium, or High
- **Regime-Specific Weights**: Optimal factor weights for each market condition
- **Regime Change Detection**: Automatically adapts to changing market conditions

### Advanced Signal Combining

Multiple methods for combining alpha factors:

- **Ridge Regression**: Learned factor weights based on historical performance
- **LightGBM**: Tree-based model for capturing non-linear factor relationships
- **Factor Rotation**: Monthly rotation based on trailing information ratios
- **Regime-Based Weighting**: Different factor weights for different market regimes

### Position Constraints & Risk Management

Advanced position sizing and risk controls:

- **Sector Exposure Limits**: Prevents overconcentration in specific sectors
- **Volatility Targeting**: Maintains consistent portfolio risk profile
- **Flexible Position Sizing**: Equal-weighted, inverse volatility, risk parity methods
- **Minimum Position Constraints**: Avoids tiny, inefficient allocations

### Performance Monitoring

Real-time performance tracking:

- **Rolling Performance Metrics**: Tracks metrics across multiple timeframes
- **Drawdown-Based De-risking**: Reduces exposure during significant drawdowns
- **Volatility Spike Response**: Adjusts positioning during volatility events
- **Alert System**: Notifies when performance metrics breach thresholds

## Optimization Parameters

The strategy optimizer tunes parameters in three categories:

1. **Signal Generation**:
   - Lookback windows for various factors
   - Thresholds for signals

2. **Factor Weighting**:
   - Relative weights for each factor in the composite signal

3. **Portfolio Construction**:
   - Rebalancing frequency
   - Number of positions (top N)
   - Position sizing method
   - Max position size
   - Cash buffer

## AWS Free Tier Usage

This system is designed to work within AWS Free Tier limits:
- Data storage in S3 (< 5GB)
- Minimal EC2 usage if deployed

## Performance Metrics

The backtester calculates and reports:
- Total Return
- Annualized Return
- Sharpe Ratio
- Max Drawdown
- Volatility
- Win Rate
- Profit Factor
- Return/Max Drawdown Ratio
- Regime-specific performance

## License

MIT License

## Project Structure

```
quant-alpha-aws/
├── data/
│   └── market_data.py      # Market data fetching and processing
├── factors/
│   └── alpha_signals.py    # Alpha signal generation
├── backtest/
│   ├── engine.py           # Basic backtesting engine
│   └── enhanced_engine.py  # Enhanced backtesting engine
├── utils/
│   ├── market_regime.py    # Market regime detection
│   ├── signal_combiner.py  # Advanced signal combination
│   ├── position_constraints.py # Position risk management
│   └── performance_monitor.py # Performance tracking
├── aws/
│   ├── api_handler.py      # FastAPI handler for AWS Lambda
│   └── utils.py            # AWS utilities
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── configs/
│   └── enhanced_backtest_config.json # Enhanced backtest configuration
├── config.py               # Configuration settings
├── main.py                 # Main entry point
├── run_enhanced_backtest.py # Enhanced backtest runner
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- `yfinance` for market data
- `bt` for backtesting
- `streamlit` for dashboard
- `fastapi` for API
- `boto3` for AWS integration
- `scikit-learn` for machine learning components
- `LightGBM` for gradient boosting 