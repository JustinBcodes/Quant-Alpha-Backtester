"""
Enhanced strategy module.

This module integrates all enhanced features:
1. Market regime detection
2. Signal combining with ML models
3. Position constraints and risk management
4. Performance monitoring and de-risking
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.market_regime import MarketRegimeDetector
from utils.signal_combiner import SignalCombiner
from utils.position_constraints import ConstraintsManager
from utils.performance_monitor import PerformanceMonitor

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedStrategy:
    """
    Enhanced strategy that integrates advanced features.
    """
    
    def __init__(
        self,
        use_market_regime: bool = True,
        signal_combiner_method: str = 'ridge',
        position_sizing_method: str = 'inverse_vol',
        target_volatility: float = 0.15,  # 15% annualized
        max_stock_weight: float = 0.20,
        max_sector_weight: float = 0.40,
        enable_performance_monitoring: bool = True,
        model_path: Optional[str] = None,
        sector_map: Optional[Dict[str, str]] = None,
        vix_high_threshold: Optional[float] = None,
        rolling_std_high_threshold: Optional[float] = None
    ):
        """
        Initialize the EnhancedStrategy.
        
        Args:
            use_market_regime: Whether to use market regime detection
            signal_combiner_method: Method for signal combining ('equal', 'trailing', 'ridge', 'lightgbm', 'regime')
            position_sizing_method: Method for position sizing ('equal', 'inverse_vol', 'risk_parity', 'signal_weighted')
            target_volatility: Target portfolio volatility
            max_stock_weight: Maximum weight per stock
            max_sector_weight: Maximum weight per sector
            enable_performance_monitoring: Whether to enable performance monitoring
            model_path: Path to save/load ML models
            sector_map: Dictionary mapping tickers to sectors
            vix_high_threshold: VIX threshold for high volatility regime
            rolling_std_high_threshold: Rolling std threshold for high volatility
        """
        self.use_market_regime = use_market_regime
        self.signal_combiner_method = signal_combiner_method
        self.position_sizing_method = position_sizing_method
        self.target_volatility = target_volatility
        self.max_stock_weight = max_stock_weight
        self.max_sector_weight = max_sector_weight
        self.enable_performance_monitoring = enable_performance_monitoring
        self.model_path = model_path
        self.sector_map = sector_map or {}
        
        # Import config values for regime detection thresholds
        from config import VIX_HIGH_THRESHOLD, ROLLING_STD_HIGH_THRESHOLD
        self.vix_high_threshold = vix_high_threshold if vix_high_threshold is not None else VIX_HIGH_THRESHOLD
        self.rolling_std_high_threshold = rolling_std_high_threshold if rolling_std_high_threshold is not None else ROLLING_STD_HIGH_THRESHOLD
        
        # Initialize components
        self._initialize_components()
        
        # Cache
        self.current_regime = None
        self.factor_weights = None
        self.current_positions = None
        self.cash_weight = 0.0
        self.performance_summary = None
        self.current_vix = None
        self.current_rolling_std = None
        self.use_mean_reversion = False
        
        logger.info(
            f"EnhancedStrategy initialized with regime={use_market_regime}, "
            f"signal_method={signal_combiner_method}, position_method={position_sizing_method}, "
            f"vix_threshold={vix_high_threshold}, rolling_std_threshold={rolling_std_high_threshold}"
        )
    
    def _initialize_components(self):
        """Initialize strategy components."""
        # Market regime detector
        if self.use_market_regime:
            from config import (TREND_WINDOW, TREND_THRESHOLD, VOL_WINDOW, VOL_LOOKBACK,
                              VOL_PERCENTILE_LOW, VOL_PERCENTILE_HIGH, SMOOTH_WINDOW)
            
            self.regime_detector = MarketRegimeDetector(
                trend_window=TREND_WINDOW,
                trend_threshold=TREND_THRESHOLD,
                vol_window=VOL_WINDOW,
                vol_lookback=VOL_LOOKBACK,
                vol_percentile_low=VOL_PERCENTILE_LOW,
                vol_percentile_high=VOL_PERCENTILE_HIGH,
                smooth_window=SMOOTH_WINDOW,
                vix_high_threshold=self.vix_high_threshold,
                rolling_std_high_threshold=self.rolling_std_high_threshold
            )
        else:
            self.regime_detector = None
        
        # Signal combiner
        self.signal_combiner = SignalCombiner(
            method=self.signal_combiner_method,
            lookback_window=126,
            refit_frequency=21,
            performance_metric='information_ratio',
            min_history=63,
            regularization=1.0,
            model_path=self.model_path
        )
        
        # Position constraints manager
        self.constraints_manager = ConstraintsManager(
            target_volatility=self.target_volatility,
            max_stock_weight=self.max_stock_weight,
            max_sector_weight=self.max_sector_weight,
            min_stocks=5,
            max_stocks=20,
            position_sizing=self.position_sizing_method,
            vol_lookback=63,
            vol_max_adjustment=2.0,
            sector_neutral=False,
            min_position_weight=0.01
        )
        
        # Performance monitor
        if self.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(
                lookback_windows=[5, 21, 63, 126, 252],
                drawdown_threshold=-0.10,
                vol_threshold=0.20,
                sharpe_threshold=0.0,
                correlation_threshold=0.7,
                performance_threshold=-0.05,
                metric_dampening=0.5,
                alerts_enabled=True
            )
        else:
            self.performance_monitor = None
    
    def detect_market_regime(
        self, 
        benchmark_prices: pd.Series,
        vix_data: Optional[pd.Series] = None
    ) -> Dict[str, str]:
        """
        Detect current market regime using benchmark prices and VIX.
        
        Args:
            benchmark_prices: Series with benchmark prices
            vix_data: Optional Series with VIX data
            
        Returns:
            Dictionary with regime information
        """
        if not self.use_market_regime or self.regime_detector is None:
            return {
                'trend': 'SIDEWAYS',
                'volatility': 'MEDIUM',
                'combined': 'sideways_medium_vol',
                'use_mean_reversion': False
            }
        
        # Update current rolling std
        returns = benchmark_prices.pct_change().dropna()
        self.current_rolling_std = returns.rolling(window=21).std().iloc[-1] * np.sqrt(252)
        
        # Update current VIX if available
        if vix_data is not None and not vix_data.empty:
            self.current_vix = vix_data.iloc[-1]
        
        # Detect regime
        self.current_regime = self.regime_detector.get_current_regime(benchmark_prices, vix_data)
        
        # Set the mean reversion flag
        self.use_mean_reversion = self.current_regime['use_mean_reversion']
        
        return self.current_regime
    
    def should_use_mean_reversion(self) -> bool:
        """
        Determine if the current market regime favors mean reversion strategies.
        
        Returns:
            Boolean indicating whether to use mean reversion strategies
        """
        if self.regime_detector is not None:
            return self.regime_detector.should_use_mean_reversion(
                vix_value=self.current_vix,
                rolling_std=self.current_rolling_std
            )
        return False
    
    def get_factor_weights(
        self,
        current_regime: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Get factor weights based on current regime.
        
        Args:
            current_regime: Optional current market regime
            
        Returns:
            Dictionary with factor weights
        """
        regime = current_regime or self.current_regime
        
        if self.use_market_regime and self.regime_detector is not None and regime is not None:
            # Get weights based on whether to use mean reversion or momentum
            if regime.get('use_mean_reversion', False):
                weights = self.regime_detector.get_signal_weights_by_regime(True)
                logger.info("Using mean reversion factor weights")
            else:
                weights = self.regime_detector.get_signal_weights_by_regime(False)
                logger.info("Using momentum factor weights")
        else:
            # Default equal weights
            weights = {
                'momentum': 0.125,
                'rsi': 0.125,
                'sma': 0.125,
                'volatility': 0.125,
                'liquidity': 0.125,
                'gap': 0.125,
                'price_percentile': 0.125,
                'roc': 0.125
            }
        
        self.factor_weights = weights
        return weights
    
    def combine_signals(
        self,
        factor_data: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        factor_weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Combine factor signals using the signal combiner.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: Optional DataFrame with forward returns
            current_date: Current date for signal combination
            factor_weights: Optional factor weights to use
            
        Returns:
            Series with combined signals
        """
        weights = factor_weights or self.factor_weights
        
        # Combine signals
        combined_signal = self.signal_combiner.combine_signals(
            factor_data=factor_data,
            returns=returns,
            current_date=current_date,
            regime_weights=weights
        )
        
        return combined_signal
    
    def calculate_positions(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        current_date: pd.Timestamp,
        market_caps: Optional[pd.Series] = None,
        current_drawdown: Optional[float] = None
    ) -> Tuple[pd.Series, float]:
        """
        Calculate positions based on signals and constraints.
        
        Args:
            signals: Series with combined signals
            returns: DataFrame with historical returns
            current_date: Current date for calculation
            market_caps: Optional series with market capitalizations
            current_drawdown: Optional current drawdown for de-risking
            
        Returns:
            Tuple of (position weights, cash weight)
        """
        positions, cash_weight = self.constraints_manager.apply_all_constraints(
            signals=signals,
            returns=returns,
            current_date=current_date,
            market_caps=market_caps,
            sector_map=self.sector_map,
            current_drawdown=current_drawdown
        )
        
        self.current_positions = positions
        self.cash_weight = cash_weight
        
        return positions, cash_weight
    
    def monitor_performance(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Monitor portfolio performance and check for de-risking triggers.
        
        Args:
            portfolio_values: Series with portfolio values
            benchmark_values: Optional series with benchmark values
            
        Returns:
            Dictionary with performance summary
        """
        if not self.enable_performance_monitoring or self.performance_monitor is None:
            return {}
        
        # Update metrics
        self.performance_monitor.update_metrics(portfolio_values, benchmark_values)
        
        # Check alerts
        alerts = self.performance_monitor.check_alerts()
        
        # Get summary
        summary = self.performance_monitor.get_summary_report()
        self.performance_summary = summary
        
        return summary
    
    def should_derisk(self) -> Tuple[bool, float, str]:
        """
        Check if portfolio de-risking should be triggered.
        
        Returns:
            Tuple of (should_derisk, derisk_factor, reason)
        """
        if not self.enable_performance_monitoring or self.performance_monitor is None:
            return False, 1.0, ""
        
        should_derisk, reason = self.performance_monitor.should_derisk()
        
        if should_derisk:
            derisk_factor = self.performance_monitor.get_derisk_factor()
            return True, derisk_factor, reason
        
        return False, 1.0, ""
    
    def get_monthly_factor_rotation(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        current_date: pd.Timestamp,
        window: int = 21,
        top_n: int = 3
    ) -> Dict[str, float]:
        """
        Get monthly factor rotation weights based on trailing information ratio.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            current_date: Current date for rotation
            window: Window for performance calculation
            top_n: Number of top factors to select
            
        Returns:
            Dictionary with rotated factor weights
        """
        return self.signal_combiner.calculate_monthly_factor_rotation(
            factor_data=factor_data,
            returns=returns,
            current_date=current_date,
            window=window,
            top_n=top_n
        )
    
    def get_sector_allocations(self) -> Dict[str, float]:
        """
        Get current sector allocations.
        
        Returns:
            Dictionary with sector allocations
        """
        if self.current_positions is None or self.sector_map is None:
            return {}
        
        return self.constraints_manager.get_sector_allocations(
            weights=self.current_positions,
            sector_map=self.sector_map
        )
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.Series] = None,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        rebalance_freq: int = 21,  # Monthly
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run enhanced backtest with all features.
        
        Args:
            data: DataFrame with market data
            signals: DataFrame with factor signals
            benchmark_data: Optional DataFrame with benchmark data
            vix_data: Optional Series with VIX index data
            initial_capital: Initial capital
            commission: Commission rate
            rebalance_freq: Rebalance frequency in days
            start_date: Start date for backtest
            end_date: End date for backtest
            save_path: Path to save results
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting enhanced backtest...")
        
        # Extract price data
        if isinstance(data.index, pd.MultiIndex):
            # Pivot data if it's in a multi-index format
            prices = data['Close'].unstack('ticker')
        else:
            prices = data
        
        # Extract benchmark data if available
        benchmark_prices = None
        if benchmark_data is not None:
            if 'SPY' in benchmark_data.columns:
                benchmark_prices = benchmark_data['SPY']
            elif 'Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['Close']
        
        # Filter dates if specified
        if start_date is not None:
            prices = prices.loc[start_date:]
            if signals is not None:
                signals = signals.loc[signals.index >= start_date]
            if benchmark_prices is not None:
                benchmark_prices = benchmark_prices.loc[benchmark_prices.index >= start_date]
            if vix_data is not None:
                vix_data = vix_data.loc[vix_data.index >= start_date]
                
        if end_date is not None:
            prices = prices.loc[:end_date]
            if signals is not None:
                signals = signals.loc[signals.index <= end_date]
            if benchmark_prices is not None:
                benchmark_prices = benchmark_prices.loc[benchmark_prices.index <= end_date]
            if vix_data is not None:
                vix_data = vix_data.loc[vix_data.index <= end_date]
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Initialize portfolio tracking
        dates = prices.index
        portfolio_value = pd.Series(index=dates, dtype=float)
        portfolio_value.iloc[0] = initial_capital
        
        holdings = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        cash = pd.Series(index=dates, dtype=float)
        cash.iloc[0] = initial_capital
        
        trades = []
        regimes = []
        factor_weights_history = []
        position_weights_history = []
        cash_weights_history = []
        regime_signals_history = []
        
        # Calculate rebalance days
        rebalance_days = list(range(0, len(dates), rebalance_freq))
        if 0 not in rebalance_days:
            rebalance_days.insert(0, 0)
        
        # Factor data in the right format for signal combining
        if signals is not None:
            if 'final_score' in signals.columns:
                # Signals already has final_score, separate out individual factors
                factor_data = signals.drop(columns=['final_score'])
            else:
                # Assume all columns are factor signals
                factor_data = signals
        else:
            factor_data = None
        
        # Main backtest loop
        logger.info(f"Running backtest with {len(rebalance_days)} rebalance points")
        
        for i in range(1, len(dates)):
            current_date = dates[i]
            prev_date = dates[i-1]
            
            # Update holdings based on price changes
            for ticker in holdings.columns:
                if ticker in prices.columns and not pd.isna(prices.iloc[i-1][ticker]) and not pd.isna(prices.iloc[i][ticker]):
                    price_change = prices.iloc[i][ticker] / prices.iloc[i-1][ticker]
                    holdings.loc[current_date, ticker] = holdings.loc[prev_date, ticker] * price_change
            
            # Copy cash forward
            cash.iloc[i] = cash.iloc[i-1]
            
            # Calculate current portfolio value
            current_portfolio_value = cash.iloc[i] + holdings.iloc[i].sum()
            portfolio_value.iloc[i] = current_portfolio_value
            
            # Rebalance if it's a rebalance day
            if i in rebalance_days:
                # Step 1: Detect market regime
                if benchmark_prices is not None:
                    # Get VIX data for the current date if available
                    current_vix_data = None
                    if vix_data is not None:
                        current_vix_data = vix_data.loc[:current_date]
                    
                    # Detect regime with VIX data if available
                    regime = self.detect_market_regime(
                        benchmark_prices.loc[:current_date], 
                        current_vix_data
                    )
                    
                    # Calculate rolling standard deviation for regime detection
                    benchmark_returns = benchmark_prices.pct_change().dropna()
                    rolling_std = benchmark_returns.rolling(window=21).std().iloc[-1] * np.sqrt(252)
                    
                    # Store regime data
                    regime_info = {
                        'date': current_date,
                        'regime': regime,
                        'vix': None if vix_data is None else vix_data.loc[current_date] if current_date in vix_data.index else None,
                        'rolling_std': rolling_std,
                        'use_mean_reversion': regime.get('use_mean_reversion', False)
                    }
                    regimes.append(regime_info)
                    
                    logger.info(f"Date: {current_date}, Regime: {regime['combined']}, Mean Reversion: {regime['use_mean_reversion']}")
                
                # Step 2: Get factor weights based on regime
                factor_weights = self.get_factor_weights()
                factor_weights_history.append({'date': current_date, 'weights': factor_weights})
                
                # Step 3: Combine signals
                if factor_data is not None and current_date in factor_data.index:
                    # Check if we should use mean reversion or momentum strategies
                    use_mean_reversion = self.should_use_mean_reversion()
                    
                    # Record the signal strategy used
                    regime_signals_history.append({
                        'date': current_date, 
                        'use_mean_reversion': use_mean_reversion,
                        'vix': self.current_vix,
                        'rolling_std': self.current_rolling_std
                    })
                    
                    # Combine signals to get scores for each ticker
                    combined_signal = self.combine_signals(
                        factor_data.loc[:current_date],
                        returns.loc[:current_date],
                        current_date,
                        factor_weights
                    )
                    
                    # Step 4: Calculate positions with constraints
                    if self.enable_performance_monitoring and self.performance_monitor is not None:
                        # Calculate current drawdown
                        drawdown = self.performance_monitor.calculate_drawdown(portfolio_value.loc[:current_date])
                        current_drawdown = drawdown.iloc[-1]
                    else:
                        current_drawdown = None
                    
                    positions, cash_weight = self.calculate_positions(
                        combined_signal.loc[current_date],
                        returns.loc[:current_date],
                        current_date,
                        None,  # market_caps
                        current_drawdown
                    )
                    
                    position_weights_history.append({'date': current_date, 'positions': positions})
                    cash_weights_history.append({'date': current_date, 'cash_weight': cash_weight})
                    
                    # Step 5: Execute trades to match target positions
                    # Calculate target position values
                    target_value = current_portfolio_value * (1 - cash_weight)
                    target_positions = {}
                    
                    for ticker, weight in positions.items():
                        if weight > 0:
                            target_positions[ticker] = target_value * weight
                    
                    # Calculate trades
                    for ticker in prices.columns:
                        current_value = holdings.loc[current_date, ticker]
                        target_value = target_positions.get(ticker, 0.0)
                        
                        if abs(current_value - target_value) > 0.01:  # Small threshold to avoid tiny trades
                            # Calculate trade
                            trade_value = target_value - current_value
                            
                            # Apply commission
                            commission_amount = abs(trade_value) * commission
                            
                            if trade_value > 0:
                                # Buy
                                if cash.iloc[i] >= trade_value + commission_amount:
                                    # Record trade
                                    trades.append({
                                        'date': current_date,
                                        'ticker': ticker,
                                        'action': 'BUY',
                                        'value': trade_value,
                                        'commission': commission_amount
                                    })
                                    
                                    # Update holdings and cash
                                    holdings.loc[current_date, ticker] += trade_value
                                    cash.iloc[i] -= (trade_value + commission_amount)
                            else:
                                # Sell
                                # Record trade
                                trades.append({
                                    'date': current_date,
                                    'ticker': ticker,
                                    'action': 'SELL',
                                    'value': -trade_value,
                                    'commission': commission_amount
                                })
                                
                                # Update holdings and cash
                                holdings.loc[current_date, ticker] += trade_value
                                cash.iloc[i] -= trade_value
                                cash.iloc[i] -= commission_amount
                    
                    # Step 6: Monitor performance
                    if self.enable_performance_monitoring and self.performance_monitor is not None:
                        if benchmark_prices is not None:
                            perf_summary = self.monitor_performance(
                                portfolio_value.loc[:current_date],
                                benchmark_prices.loc[:current_date]
                            )
                        else:
                            perf_summary = self.monitor_performance(
                                portfolio_value.loc[:current_date]
                            )
                        
                        # Check for de-risking triggers
                        should_derisk, derisk_factor, reason = self.should_derisk()
                        
                        if should_derisk:
                            logger.warning(
                                f"De-risking triggered on {current_date}: {reason}. "
                                f"Factor: {derisk_factor:.2f}"
                            )
        
        # Calculate final metrics
        final_metrics = self._calculate_metrics(portfolio_value, holdings, cash, trades)
        
        # Prepare results
        results = {
            'portfolio_value': portfolio_value,
            'holdings': holdings,
            'cash': cash,
            'trades': trades,
            'metrics': final_metrics,
            'regimes': regimes,
            'factor_weights': factor_weights_history,
            'position_weights': position_weights_history,
            'cash_weights': cash_weights_history,
            'regime_signals': regime_signals_history
        }
        
        # Save results if path is provided
        if save_path:
            self._save_results(results, save_path)
        
        logger.info(f"Enhanced backtest completed with final value: ${portfolio_value.iloc[-1]:.2f}")
        
        return results
    
    def _calculate_metrics(
        self,
        portfolio_value: pd.Series,
        holdings: pd.DataFrame,
        cash: pd.Series,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_value: Series with portfolio values
            holdings: DataFrame with holdings
            cash: Series with cash values
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with metrics
        """
        # Convert trades to DataFrame
        if trades:
            trades_df = pd.DataFrame(trades)
        else:
            trades_df = pd.DataFrame(columns=['date', 'ticker', 'action', 'value', 'commission'])
        
        # Calculate returns
        returns = portfolio_value.pct_change().fillna(0)
        
        # Calculate metrics
        total_days = len(portfolio_value)
        trading_days_per_year = 252
        
        # Total return
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        
        # Annualized return
        years = total_days / trading_days_per_year
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
        # Volatility
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(trading_days_per_year) * 100
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Return over max drawdown
        return_over_max_drawdown = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Trading metrics
        total_trades = len(trades_df)
        total_commission = trades_df['commission'].sum() if 'commission' in trades_df else 0
        
        # Win rate
        if 'action' in trades_df and total_trades > 0:
            sells = trades_df[trades_df['action'] == 'SELL']
            buys = trades_df[trades_df['action'] == 'BUY']
            
            if len(sells) > 0:
                profitable_trades = sells[sells['value'] > 0]
                win_rate = len(profitable_trades) / len(sells) * 100
                
                if len(profitable_trades) > 0:
                    avg_profit = profitable_trades['value'].mean()
                else:
                    avg_profit = 0
                
                if len(sells) - len(profitable_trades) > 0:
                    losing_trades = sells[sells['value'] <= 0]
                    avg_loss = losing_trades['value'].mean()
                else:
                    avg_loss = 0
            else:
                win_rate = 0
                avg_profit = 0
                avg_loss = 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
        
        # Exposure
        avg_cash_pct = (cash / portfolio_value).mean() * 100
        avg_exposure = 100 - avg_cash_pct
        
        # Calculate 30-day Sharpe ratio at the end
        if len(returns) >= 30:
            final_30d_returns = returns.iloc[-30:]
            final_30d_volatility = final_30d_returns.std() * np.sqrt(trading_days_per_year)
            final_30d_sharpe = final_30d_returns.mean() * trading_days_per_year / final_30d_volatility if final_30d_volatility > 0 else 0
        else:
            final_30d_sharpe = 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'return_over_max_drawdown': return_over_max_drawdown,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_cash_pct': avg_cash_pct,
            'avg_exposure': avg_exposure,
            'final_30d_sharpe': final_30d_sharpe
        }
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any], save_path: str):
        """
        Save backtest results to disk.
        
        Args:
            results: Dictionary with results
            save_path: Path to save results
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save portfolio value
            results['portfolio_value'].to_csv(os.path.join(save_path, 'portfolio_value.csv'))
            
            # Save holdings
            results['holdings'].to_csv(os.path.join(save_path, 'holdings.csv'))
            
            # Save cash
            results['cash'].to_csv(os.path.join(save_path, 'cash.csv'))
            
            # Save trades
            if results['trades']:
                pd.DataFrame(results['trades']).to_csv(os.path.join(save_path, 'trades.csv'), index=False)
            
            # Save metrics
            pd.DataFrame([results['metrics']]).to_csv(os.path.join(save_path, 'metrics.csv'))
            
            # Save regimes if available
            if results['regimes']:
                pd.DataFrame(results['regimes']).to_csv(os.path.join(save_path, 'regimes.csv'), index=False)
            
            # Save factor weights if available
            if results['factor_weights']:
                # Convert to DataFrame
                weights_data = []
                for entry in results['factor_weights']:
                    row = {'date': entry['date']}
                    row.update(entry['weights'])
                    weights_data.append(row)
                    
                pd.DataFrame(weights_data).to_csv(os.path.join(save_path, 'factor_weights.csv'), index=False)
            
            # Save position weights if available
            if results['position_weights']:
                # For each position weight entry, save a separate file
                position_dir = os.path.join(save_path, 'position_weights')
                os.makedirs(position_dir, exist_ok=True)
                
                for entry in results['position_weights']:
                    date_str = entry['date'].strftime('%Y%m%d')
                    pd.Series(entry['positions']).to_csv(os.path.join(position_dir, f'positions_{date_str}.csv'))
            
            # Save cash weights if available
            if results['cash_weights']:
                cash_weights = pd.DataFrame([
                    {'date': entry['date'], 'cash_weight': entry['cash_weight']}
                    for entry in results['cash_weights']
                ])
                cash_weights.to_csv(os.path.join(save_path, 'cash_weights.csv'), index=False)
            
            # Save regime signals if available
            if results['regime_signals']:
                pd.DataFrame(results['regime_signals']).to_csv(os.path.join(save_path, 'regime_signals.csv'), index=False)
            
            logger.info(f"Results saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc()) 