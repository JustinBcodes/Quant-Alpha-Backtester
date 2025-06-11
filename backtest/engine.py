"""
Simple, robust backtesting engine that doesn't rely on external libraries.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import INITIAL_CAPITAL, COMMISSION, REBALANCE_FREQ

# Import new modules
try:
    from utils.market_regime import MarketRegimeDetector
    from utils.signal_combiner import SignalCombiner
    from utils.position_constraints import ConstraintsManager
    from utils.performance_monitor import PerformanceMonitor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    import warnings
    warnings.warn("Enhanced features not available. Using basic backtest engine.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Simple and robust backtesting engine using pandas."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = COMMISSION,
        rebalance_freq: str = REBALANCE_FREQ,
        top_n: int = 5,
        max_position_size: float = 0.2,  # Max 20% in any one position
        stop_loss_pct: float = 0.5,      # Stop trading if we lose 50%
        cash_buffer_pct: float = 0.1,    # Keep 10% in cash
        weekly_rebalance: bool = True,   # Default to weekly rebalancing
        slippage: float = 0.001,         # 0.1% slippage per trade
        position_sizing_method: str = 'equal',  # Options: 'equal', 'inverse_volatility', 'rank', 'score_weighted'
        decay_threshold: float = 0.0     # Signal decay threshold for rebalancing
    ):
        """
        Initialize the BacktestEngine.
        
        Args:
            data: DataFrame with multi-index (ticker, date) and OHLCV columns
            signals: DataFrame with signal scores for each ticker
            initial_capital: Initial capital for backtest
            commission: Commission rate per trade
            rebalance_freq: Rebalancing frequency in days
            top_n: Number of top stocks to hold
            max_position_size: Maximum position size as a fraction of portfolio
            stop_loss_pct: Stop trading if portfolio value falls below this % of initial capital
            cash_buffer_pct: Percentage of portfolio to keep in cash
            weekly_rebalance: If True, rebalance weekly regardless of rebalance_freq
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("Signals must be a pandas DataFrame")
            
        # Make sure the data has the expected structure
        if 'ticker' not in data.index.names:
            raise ValueError("Data must have a multi-index with 'ticker' as one of the levels")
        
        # Ensure signal DataFrame has the required 'final_score' column
        if 'final_score' not in signals.columns:
            raise ValueError("Signals DataFrame must have a 'final_score' column")
        
        # Store initial parameters
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.top_n = top_n
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.cash_buffer_pct = cash_buffer_pct
        self.weekly_rebalance = weekly_rebalance
        self.slippage = slippage
        self.position_sizing_method = position_sizing_method
        self.decay_threshold = decay_threshold
        
        # Parse rebalance frequency
        if isinstance(rebalance_freq, str) and 'D' in rebalance_freq:
            self.rebalance_freq = int(rebalance_freq.replace('D', ''))
        else:
            self.rebalance_freq = int(rebalance_freq)
            
        # Ensure rebalance frequency is not too aggressive
        if self.weekly_rebalance:
            self.rebalance_freq = 5  # Approximately weekly (5 trading days)
            
        self.results = {}
        
        # Log initialization
        logger.info(f"BacktestEngine initialized with {len(self.data.index.get_level_values('ticker').unique())} tickers")
        logger.info(f"Data shape: {self.data.shape}, Signals shape: {self.signals.shape}")
        logger.info(f"Initial capital: ${self.initial_capital}, Commission: {self.commission * 100}%")
        logger.info(f"Strategy params: Top N: {self.top_n}, Max Position: {self.max_position_size*100}%, Stop Loss: {self.stop_loss_pct*100}%")
        logger.info(f"Rebalance frequency: {self.rebalance_freq} days, Cash buffer: {self.cash_buffer_pct*100}%")
    
    def prepare_data(self) -> Tuple[pd.DataFrame, List[datetime]]:
        """
        Prepare data for backtesting by aligning dates and tickers.
        
        Returns:
            Tuple of (pivot prices DataFrame, sorted dates list)
        """
        logger.info("Preparing data for backtesting...")
        
        try:
            # Get unique dates from data
            dates = sorted(set(self.data.index.get_level_values('Date')))
            
            # Pivot data to get a DataFrame with dates as index and tickers as columns
            price_pivot = self.data['Close'].unstack('ticker')
            
            # Sort index
            price_pivot = price_pivot.sort_index()
            
            logger.info(f"Prepared price data with {len(price_pivot)} dates and {len(price_pivot.columns)} tickers")
            return price_pivot, dates
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def prepare_signals(self) -> pd.DataFrame:
        """
        Prepare signals for backtesting.
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        logger.info("Preparing signals for backtesting...")
        
        try:
            # Extract final_score column and pivot
            signals_pivot = self.signals['final_score'].unstack('ticker')
            
            # Sort index
            signals_pivot = signals_pivot.sort_index()
            
            logger.info(f"Prepared signals with shape {signals_pivot.shape}")
            return signals_pivot
            
        except Exception as e:
            logger.error(f"Error preparing signals: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the backtest and compute performance metrics.
        
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("Running backtest...")
        
        try:
            # Prepare data
            prices, all_dates = self.prepare_data()
            signals = self.prepare_signals()
            
            # Initialize portfolio
            portfolio_value = pd.Series(index=prices.index, dtype=float)
            portfolio_value.iloc[0] = self.initial_capital
            
            # Initialize holdings (as float to avoid dtype issues)
            holdings = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            
            # Initialize cash
            cash = pd.Series(index=prices.index, dtype=float)
            cash.iloc[0] = self.initial_capital
            
            # Track trades for analysis
            trades = []
            
            # Get rebalance days but ensure first day is included
            rebalance_days = list(range(0, len(prices), self.rebalance_freq))
            if 0 not in rebalance_days:
                rebalance_days.insert(0, 0)
            
            # Run backtest
            logger.info(f"Starting backtest with {len(rebalance_days)} rebalance points")
            
            for i in range(1, len(prices)):
                current_date = prices.index[i]
                prev_date = prices.index[i-1]
                
                # Update holdings based on price changes
                for ticker in holdings.columns:
                    if ticker in prices.columns and not pd.isna(prices.iloc[i-1][ticker]) and not pd.isna(prices.iloc[i][ticker]):
                        price_change = prices.iloc[i][ticker] / prices.iloc[i-1][ticker]
                        # Use .loc to avoid FutureWarning about chained assignment
                        holdings.loc[prices.index[i], ticker] = holdings.iloc[i-1][ticker] * price_change
                
                # Copy cash forward
                cash.iloc[i] = cash.iloc[i-1]
                
                # Calculate current portfolio value
                current_portfolio_value = cash.iloc[i] + holdings.iloc[i].sum()
                
                # Check stop loss - if we've lost more than the stop loss percentage, stop trading
                if current_portfolio_value < self.initial_capital * (1 - self.stop_loss_pct):
                    logger.warning(f"Stop loss triggered on {current_date}. Portfolio value: ${current_portfolio_value:.2f}")
                    # Continue tracking portfolio value but don't make new trades
                    portfolio_value.iloc[i] = current_portfolio_value
                    continue
                
                # Rebalance if it's a rebalance day
                if i in rebalance_days:
                    # Get signals for current date (or closest available date)
                    if current_date in signals.index:
                        current_signals = signals.loc[current_date]
                    else:
                        # Find closest earlier date with signals
                        signal_dates = signals.index[signals.index <= current_date]
                        if len(signal_dates) > 0:
                            closest_date = signal_dates[-1]
                            current_signals = signals.loc[closest_date]
                        else:
                            logger.warning(f"No signals available for or before {current_date}. Skipping rebalance.")
                            # Still need to update the portfolio value
                            portfolio_value.iloc[i] = current_portfolio_value
                            continue
                    
                    # Remove NaN values
                    current_signals = current_signals.dropna()
                    
                    if current_signals.empty:
                        logger.warning(f"No valid signals for {current_date}. Skipping rebalance.")
                        portfolio_value.iloc[i] = current_portfolio_value
                        continue
                    
                    # Normalize signals - make all signals positive and scale to sum to 1
                    # First, we will rank the signals to avoid extreme values
                    ranked_signals = current_signals.rank(pct=True)  # Percentile ranking
                    
                    # Get top N tickers based on ranked signals
                    top_tickers = ranked_signals.nlargest(self.top_n).index.tolist()
                    
                    if not top_tickers:
                        logger.warning(f"No top tickers selected for {current_date}. Skipping rebalance.")
                        portfolio_value.iloc[i] = current_portfolio_value
                        continue
                    
                    # Create normalized weights for the top tickers
                    weights = self._calculate_position_sizes(current_signals, top_tickers, current_portfolio_value)
                    
                    logger.info(f"Rebalancing on {current_date} with top tickers: {', '.join(top_tickers)}")
                    
                    # Calculate target values
                    target_values = weights * current_portfolio_value
                    
                    # Sell all current holdings that are not in top tickers or have reduced weight
                    for ticker in holdings.columns:
                        current_holding_value = holdings.iloc[i][ticker]
                        target_value = target_values.get(ticker, 0)
                        
                        if current_holding_value > 0:
                            if ticker not in top_tickers or current_holding_value > target_value:
                                # Calculate amount to sell
                                sell_value = current_holding_value if ticker not in top_tickers else (current_holding_value - target_value)
                                
                                # Apply commission
                                cash.iloc[i] += self._apply_slippage(sell_value, 'SELL') * (1 - self.commission)
                                
                                # Record trade
                                trades.append({
                                    'date': current_date,
                                    'ticker': ticker,
                                    'action': 'SELL',
                                    'price': prices.iloc[i][ticker] if ticker in prices.columns and not pd.isna(prices.iloc[i][ticker]) else 0,
                                    'quantity': sell_value / prices.iloc[i][ticker] if ticker in prices.columns and not pd.isna(prices.iloc[i][ticker]) else 0,
                                    'value': sell_value,
                                    'commission': sell_value * self.commission
                                })
                                
                                # Update holdings
                                if ticker not in top_tickers:
                                    holdings.loc[prices.index[i], ticker] = 0.0
                                else:
                                    holdings.loc[prices.index[i], ticker] -= sell_value
                    
                    # Recalculate cash after selling
                    available_cash = cash.iloc[i]
                    
                    # Buy or adjust positions
                    for ticker in top_tickers:
                        if ticker in prices.columns and not pd.isna(prices.iloc[i][ticker]):
                            current_value = holdings.iloc[i][ticker]
                            target_value = target_values.get(ticker, 0)
                            
                            if current_value < target_value:
                                # Calculate amount to buy
                                buy_value = min(target_value - current_value, available_cash)
                                
                                if buy_value > 0:
                                    # Apply commission
                                    actual_buy_value = self._apply_slippage(buy_value, 'BUY') * (1 - self.commission)
                                    
                                    # Update holdings
                                    holdings.loc[prices.index[i], ticker] += actual_buy_value
                                    
                                    # Update cash
                                    cash.iloc[i] -= buy_value
                                    available_cash -= buy_value
                                    
                                    # Record trade
                                    trades.append({
                                        'date': current_date,
                                        'ticker': ticker,
                                        'action': 'BUY',
                                        'price': prices.iloc[i][ticker],
                                        'quantity': actual_buy_value / prices.iloc[i][ticker],
                                        'value': buy_value,
                                        'commission': buy_value * self.commission
                                    })
                
                # Calculate portfolio value for this day
                portfolio_value.iloc[i] = cash.iloc[i] + holdings.iloc[i].sum()
                
                # Another stop-loss check - if our portfolio value is very low, stop trading
                if portfolio_value.iloc[i] < 0.1 * self.initial_capital:
                    logger.warning(f"Critical portfolio value on {current_date}: ${portfolio_value.iloc[i]:.2f}")
                    logger.warning("Stopping backtesting due to extreme losses")
                    # Fill remaining dates with the last value
                    if i < len(portfolio_value) - 1:
                        portfolio_value.iloc[i+1:] = portfolio_value.iloc[i]
                        break
            
            # Calculate daily returns
            returns = portfolio_value.pct_change().fillna(0)
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            
            # Add profit/loss to trades
            if not trades_df.empty and len(trades_df) > 1:
                try:
                    # Group by ticker to track P&L per position
                    trades_df['profit'] = 0.0
                    
                    # Calculate profit for SELL trades
                    for ticker in trades_df['ticker'].unique():
                        ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()
                        buys = ticker_trades[ticker_trades['action'] == 'BUY'].copy()
                        sells = ticker_trades[ticker_trades['action'] == 'SELL'].copy()
                        
                        if not buys.empty and not sells.empty:
                            # Simple FIFO approach to calculate profit
                            # This is a simplification - a real system would track each lot
                            avg_buy_price = buys['value'].sum() / buys['quantity'].sum() if buys['quantity'].sum() > 0 else 0
                            
                            for idx, sell in sells.iterrows():
                                if avg_buy_price > 0:
                                    profit = sell['value'] - (sell['quantity'] * avg_buy_price)
                                    trades_df.loc[idx, 'profit'] = profit
                except Exception as e:
                    logger.warning(f"Error calculating trade profits: {str(e)}")
            
            # Calculate metrics
            metrics = self._compute_metrics(portfolio_value, trades_df)
            
            # Store results
            self.results = {
                'positions': portfolio_value,
                'holdings': holdings,
                'cash': cash,
                'trades': trades_df,
                'metrics': metrics,
                'returns': returns
            }
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _calculate_position_sizes(self, current_signals: pd.Series, top_tickers: List[str], current_portfolio_value: float) -> pd.Series:
        """
        Calculate position sizes based on the selected method.
        
        Args:
            current_signals: Series with signal scores
            top_tickers: List of top tickers to allocate to
            current_portfolio_value: Current portfolio value
            
        Returns:
            Series with position weights
        """
        weights = pd.Series(0.0, index=current_signals.index)
        
        # Only proceed with tickers that have prices
        valid_tickers = [t for t in top_tickers if t in self.data.index.get_level_values('ticker')]
        
        if not valid_tickers:
            return weights
        
        if self.position_sizing_method == 'equal':
            # Equal weight for each ticker
            weight_per_ticker = (1 - self.cash_buffer_pct) / len(valid_tickers)
            for ticker in valid_tickers:
                weights.loc[ticker] = weight_per_ticker
                
        elif self.position_sizing_method == 'inverse_volatility':
            # Inverse volatility weighting
            volatilities = {}
            
            for ticker in valid_tickers:
                # Get ticker data
                ticker_data = self.data.loc[ticker]
                
                # Calculate volatility (20-day)
                returns = ticker_data['Close'].pct_change().dropna()
                if len(returns) >= 20:
                    vol = returns.rolling(window=20).std().dropna().iloc[-1]
                    volatilities[ticker] = vol if vol > 0 else 1e-6  # Avoid division by zero
                else:
                    volatilities[ticker] = 1.0  # Default if not enough data
            
            # Calculate inverse volatility
            inv_vols = {t: 1/v for t, v in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            
            # Set weights
            if total_inv_vol > 0:
                for ticker, inv_vol in inv_vols.items():
                    weights.loc[ticker] = inv_vol / total_inv_vol * (1 - self.cash_buffer_pct)
                    
        elif self.position_sizing_method == 'rank':
            # Rank-based weighting - higher ranks get higher weights
            # Get ranks for top tickers
            ranks = {}
            
            for i, ticker in enumerate(valid_tickers):
                ranks[ticker] = len(valid_tickers) - i  # Reverse rank (top ticker gets highest rank)
            
            # Calculate total rank
            total_rank = sum(ranks.values())
            
            # Set weights
            if total_rank > 0:
                for ticker, rank in ranks.items():
                    weights.loc[ticker] = rank / total_rank * (1 - self.cash_buffer_pct)
                    
        elif self.position_sizing_method == 'score_weighted':
            # Signal score-weighted allocation
            scores = {}
            
            for ticker in valid_tickers:
                scores[ticker] = float(current_signals[ticker])
            
            # Calculate total score
            total_score = sum(scores.values())
            
            # Set weights
            if total_score > 0:
                for ticker, score in scores.items():
                    weights.loc[ticker] = score / total_score * (1 - self.cash_buffer_pct)
        else:
            # Default to equal weight if method is unknown
            weight_per_ticker = (1 - self.cash_buffer_pct) / len(valid_tickers)
            for ticker in valid_tickers:
                weights.loc[ticker] = weight_per_ticker
        
        # Cap max position size
        for ticker in weights.index:
            if weights[ticker] > self.max_position_size:
                excess = weights[ticker] - self.max_position_size
                weights.loc[ticker] = self.max_position_size
                # Redistribute excess to other positions (proportionally)
                other_tickers = [t for t in valid_tickers if t != ticker and weights[t] < self.max_position_size]
                if other_tickers:
                    # Proportional redistribution
                    total_other_weight = sum(weights[t] for t in other_tickers)
                    if total_other_weight > 0:
                        for other_ticker in other_tickers:
                            additional = excess * (weights[other_ticker] / total_other_weight)
                            new_weight = weights[other_ticker] + additional
                            # Check if this would exceed max position size
                            if new_weight > self.max_position_size:
                                weights.loc[other_ticker] = self.max_position_size
                            else:
                                weights.loc[other_ticker] = new_weight
                else:
                    # No other positions to redistribute to, so add to cash
                    pass
        
        return weights
    
    def _apply_slippage(self, trade_value: float, action: str) -> float:
        """
        Apply slippage to a trade.
        
        Args:
            trade_value: Trade value before slippage
            action: 'BUY' or 'SELL'
            
        Returns:
            Trade value after slippage
        """
        if action == 'BUY':
            # Pay more when buying
            return trade_value * (1 + self.slippage)
        else:
            # Receive less when selling
            return trade_value * (1 - self.slippage)
    
    def _compute_metrics(self, portfolio_value: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        Args:
            portfolio_value: Series with portfolio values
            trades: DataFrame with trade information
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            metrics = {}
            
            # Basic return metrics
            if len(portfolio_value) > 1:
                metrics['total_return'] = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1) * 100
                returns_series = portfolio_value.pct_change().dropna()
                
                if len(returns_series) > 0:
                    metrics['annual_volatility'] = returns_series.std() * np.sqrt(252) * 100
                    avg_return = returns_series.mean()
                    metrics['sharpe_ratio'] = (avg_return / returns_series.std()) * np.sqrt(252) if returns_series.std() > 0 else 0
                    
                    # Compute max drawdown
                    cum_returns = (1 + returns_series).cumprod()
                    peak = cum_returns.cummax()
                    drawdown_series = (cum_returns / peak - 1) * 100
                    metrics['max_drawdown'] = drawdown_series.min() if not drawdown_series.empty else 0
                    
                    # Compute annual return
                    days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
                    if days > 0:
                        years = days / 365.25
                        metrics['annual_return'] = ((1 + metrics['total_return'] / 100) ** (1 / years) - 1) * 100
                    else:
                        metrics['annual_return'] = 0
                else:
                    metrics['annual_volatility'] = 0
                    metrics['sharpe_ratio'] = 0
                    metrics['max_drawdown'] = 0
                    metrics['annual_return'] = 0
            else:
                logger.warning("Not enough data points to compute return metrics")
                metrics['total_return'] = 0
                metrics['annual_volatility'] = 0
                metrics['sharpe_ratio'] = 0
                metrics['max_drawdown'] = 0
                metrics['annual_return'] = 0
            
            # Trade metrics
            if not trades.empty:
                metrics['total_trades'] = len(trades)
                
                buy_trades = trades[trades['action'] == 'BUY']
                sell_trades = trades[trades['action'] == 'SELL']
                
                metrics['total_buys'] = len(buy_trades)
                metrics['total_sells'] = len(sell_trades)
                metrics['total_commission'] = trades['commission'].sum()
                
                # Win rate (if possible to calculate)
                if 'profit' in trades.columns:
                    profitable_trades = trades[trades['profit'] > 0]
                    metrics['win_rate'] = len(profitable_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
                    metrics['avg_profit'] = profitable_trades['profit'].mean() if not profitable_trades.empty else 0
                    metrics['avg_loss'] = trades[trades['profit'] < 0]['profit'].mean() if len(trades[trades['profit'] < 0]) > 0 else 0
                else:
                    metrics['win_rate'] = 0
                    metrics['avg_profit'] = 0
                    metrics['avg_loss'] = 0
            else:
                metrics['total_trades'] = 0
                metrics['total_buys'] = 0
                metrics['total_sells'] = 0
                metrics['total_commission'] = 0
                metrics['win_rate'] = 0
                metrics['avg_profit'] = 0
                metrics['avg_loss'] = 0
            
            # Exposure metrics
            avg_cash_pct = (self.results.get('cash', pd.Series()) / portfolio_value).mean() * 100 if 'cash' in self.results else 0
            metrics['avg_cash_pct'] = avg_cash_pct
            metrics['avg_exposure'] = 100 - avg_cash_pct
            
            # Return over risk metrics
            if metrics['max_drawdown'] != 0:
                metrics['return_over_max_drawdown'] = metrics['total_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['return_over_max_drawdown'] = 0
                
            # Calculate 30-day rolling Sharpe ratio at the end
            if len(returns_series) > 30:
                rolling_sharpe = returns_series.rolling(30).mean() / returns_series.rolling(30).std() * np.sqrt(252)
                metrics['final_30d_sharpe'] = rolling_sharpe.iloc[-1] if not pd.isna(rolling_sharpe.iloc[-1]) else 0
            else:
                metrics['final_30d_sharpe'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return basic metrics to prevent failures
            return {
                'total_return': 0,
                'annual_return': 0,
                'annual_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'total_buys': 0,
                'total_sells': 0,
                'total_commission': 0,
                'avg_cash_pct': 0,
                'avg_exposure': 0
            }
    
    def plot_results(self) -> Figure:
        """
        Plot backtest results.
        
        Returns:
            matplotlib Figure object
        """
        if not self.results:
            raise ValueError("Run backtest first using run_backtest()")
        
        try:
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 1, 1])
            
            # Plot equity curve
            ax1 = axes[0]
            self.results['positions'].plot(ax=ax1, label='Portfolio Value')
            
            # Add benchmark (if available)
            if 'SPY' in self.data.index.get_level_values('ticker'):
                spy_prices = self.data.loc['SPY', 'Close']
                benchmark = spy_prices / spy_prices.iloc[0] * self.initial_capital
                benchmark = benchmark.reindex(self.results['positions'].index)
                benchmark.plot(ax=ax1, label='SPY Benchmark', alpha=0.7, linestyle='--')
                
            ax1.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.3, label='Initial Capital')
            ax1.set_title('Strategy Performance')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            ax2 = axes[1]
            if len(self.results['positions']) > 1:
                returns = self.results['positions'].pct_change().fillna(0)
                cum_returns = (1 + returns).cumprod()
                peak = cum_returns.cummax()
                drawdown = (cum_returns / peak - 1) * 100
                
                drawdown.plot(ax=ax2, color='red')
                ax2.set_title('Drawdown')
                ax2.set_ylabel('Drawdown (%)')
                ax2.grid(True)
                
                # Add stop loss line
                ax2.axhline(y=-self.stop_loss_pct * 100, color='black', linestyle='--', alpha=0.5, label='Stop Loss')
                ax2.legend()
            else:
                ax2.set_title('Drawdown (not enough data)')
            
            # Plot position allocation
            ax3 = axes[2]
            if 'holdings' in self.results:
                holdings = self.results['holdings'].copy()
                # Convert to percentage
                total_value = holdings.sum(axis=1) + self.results['cash']
                for col in holdings.columns:
                    holdings[col] = holdings[col] / total_value * 100
                
                # Add cash as a column
                holdings['Cash'] = self.results['cash'] / total_value * 100
                
                # Keep only columns with non-zero allocation
                non_zero_cols = holdings.columns[holdings.max() > 1]  # More than 1% allocation at some point
                if len(non_zero_cols) > 0:
                    holdings[non_zero_cols].plot.area(ax=ax3, stacked=True, alpha=0.7)
                    ax3.set_title('Position Allocation')
                    ax3.set_ylabel('Allocation (%)')
                    ax3.grid(True)
                    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax3.set_title('No significant positions')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a blank figure if there's an error
            fig = plt.figure(figsize=(12, 8))
            plt.figtext(0.5, 0.5, f"Error plotting results: {str(e)}", 
                       ha='center', va='center', fontsize=12)
            return fig
    
    def save_results(self, filename: str) -> None:
        """
        Save backtest results to file.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            raise ValueError("Run backtest first using run_backtest()")
        
        try:
            # Save metrics
            pd.Series(self.results['metrics']).to_csv(f"{filename}_metrics.csv")
            
            # Save positions
            self.results['positions'].to_csv(f"{filename}_positions.csv")
            
            # Save trades
            if 'trades' in self.results and not self.results['trades'].empty:
                self.results['trades'].to_csv(f"{filename}_trades.csv")
            
            # Save holdings
            if 'holdings' in self.results:
                self.results['holdings'].to_csv(f"{filename}_holdings.csv")
            
            # Save plot
            fig = self.plot_results()
            fig.savefig(f"{filename}_plot.png")
            plt.close(fig)
            
            logger.info(f"Saved results to {filename}_*")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Example usage
    from data.market_data import MarketDataFetcher
    from factors.alpha_signals import AlphaSignalGenerator
    
    # Fetch data
    fetcher = MarketDataFetcher()
    data = fetcher.load_historical_data()
    
    # Generate signals
    generator = AlphaSignalGenerator(data)
    signals = generator.compute_all_signals()
    
    # Run backtest
    engine = BacktestEngine(data, signals)
    results = engine.run_backtest()
    
    # Print metrics
    print("\nBacktest Metrics:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.2f}")
    
    # Plot results
    engine.plot_results()
    plt.show() 