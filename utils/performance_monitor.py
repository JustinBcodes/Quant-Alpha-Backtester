"""
Performance monitoring module.

This module implements:
1. Rolling performance metrics
2. Drawdown monitoring
3. Performance-based de-risking triggers
4. Alerts and reporting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2

class PerformanceMonitor:
    """
    Monitors portfolio performance and triggers de-risking.
    """
    
    def __init__(
        self,
        lookback_windows: List[int] = [5, 21, 63, 126, 252],
        drawdown_threshold: float = -0.10,  # -10% drawdown
        vol_threshold: float = 0.20,  # 20% annualized volatility
        sharpe_threshold: float = 0.0,  # 0 Sharpe ratio
        correlation_threshold: float = 0.7,  # 0.7 correlation
        performance_threshold: float = -0.05,  # -5% rolling performance
        metric_dampening: float = 0.5,  # Exponential smoothing factor
        alerts_enabled: bool = True
    ):
        """
        Initialize the PerformanceMonitor.
        
        Args:
            lookback_windows: List of lookback windows for metrics
            drawdown_threshold: Threshold for drawdown alerts
            vol_threshold: Threshold for volatility alerts
            sharpe_threshold: Threshold for Sharpe ratio alerts
            correlation_threshold: Threshold for correlation alerts
            performance_threshold: Threshold for performance alerts
            metric_dampening: Factor for metric smoothing
            alerts_enabled: Whether to enable alerts
        """
        self.lookback_windows = lookback_windows
        self.drawdown_threshold = drawdown_threshold
        self.vol_threshold = vol_threshold
        self.sharpe_threshold = sharpe_threshold
        self.correlation_threshold = correlation_threshold
        self.performance_threshold = performance_threshold
        self.metric_dampening = metric_dampening
        self.alerts_enabled = alerts_enabled
        
        # Initialize metric storage
        self.metrics_history = {}
        self.current_metrics = {}
        self.alerts = []
        
        # Smoothed metrics
        self.smoothed_metrics = {}
        
        logger.info(
            f"PerformanceMonitor initialized with drawdown_threshold={drawdown_threshold:.1%}, "
            f"vol_threshold={vol_threshold:.1%}"
        )
    
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate returns from portfolio values.
        
        Args:
            portfolio_values: Series with portfolio values
            
        Returns:
            Series with returns
        """
        return portfolio_values.pct_change().fillna(0)
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate drawdown from portfolio values.
        
        Args:
            portfolio_values: Series with portfolio values
            
        Returns:
            Series with drawdown values
        """
        # Calculate rolling maximum
        rolling_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values / rolling_max) - 1.0
        
        return drawdown
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Series with returns
            window: Lookback window
            annualize: Whether to annualize the volatility
            
        Returns:
            Series with volatility values
        """
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        window: int,
        risk_free_rate: float = 0.0
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Series with returns
            window: Lookback window
            risk_free_rate: Risk-free rate
            
        Returns:
            Series with Sharpe ratio values
        """
        # Calculate rolling mean return
        mean_return = returns.rolling(window=window).mean() * 252
        
        # Calculate rolling volatility
        volatility = self.calculate_volatility(returns, window)
        
        # Calculate Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / volatility
        
        return sharpe
    
    def calculate_rolling_correlation(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Calculate rolling correlation with benchmark.
        
        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns
            window: Lookback window
            
        Returns:
            Series with correlation values
        """
        # Calculate rolling correlation
        correlation = returns.rolling(window=window).corr(benchmark_returns)
        
        return correlation
    
    def calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Calculate rolling beta with benchmark.
        
        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns
            window: Lookback window
            
        Returns:
            Series with beta values
        """
        # Calculate rolling covariance
        covariance = returns.rolling(window=window).cov(benchmark_returns)
        
        # Calculate rolling variance of benchmark
        benchmark_variance = benchmark_returns.rolling(window=window).var()
        
        # Calculate beta
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Calculate rolling information ratio.
        
        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns
            window: Lookback window
            
        Returns:
            Series with information ratio values
        """
        # Calculate excess returns
        excess_returns = returns - benchmark_returns
        
        # Calculate tracking error
        tracking_error = excess_returns.rolling(window=window).std() * np.sqrt(252)
        
        # Calculate annualized alpha
        alpha = excess_returns.rolling(window=window).mean() * 252
        
        # Calculate information ratio
        ir = alpha / tracking_error
        
        return ir
    
    def calculate_rolling_metrics(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None
    ) -> Dict[str, Dict[int, pd.Series]]:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: Series with portfolio values
            benchmark_values: Optional series with benchmark values
            
        Returns:
            Dictionary with rolling metrics
        """
        # Calculate returns
        returns = self.calculate_returns(portfolio_values)
        
        # Calculate benchmark returns if available
        benchmark_returns = None
        if benchmark_values is not None:
            benchmark_returns = self.calculate_returns(benchmark_values)
        
        # Calculate drawdown
        drawdown = self.calculate_drawdown(portfolio_values)
        
        # Initialize metrics dictionary
        metrics = {
            'drawdown': {'full': drawdown},
            'returns': {'full': returns},
            'volatility': {},
            'sharpe_ratio': {},
            'cumulative_return': {}
        }
        
        # Add benchmark metrics if available
        if benchmark_returns is not None:
            metrics['correlation'] = {}
            metrics['beta'] = {}
            metrics['information_ratio'] = {}
            metrics['excess_return'] = {}
        
        # Calculate rolling metrics for each window
        for window in self.lookback_windows:
            # Volatility
            metrics['volatility'][window] = self.calculate_volatility(returns, window)
            
            # Sharpe ratio
            metrics['sharpe_ratio'][window] = self.calculate_sharpe_ratio(returns, window)
            
            # Cumulative return
            metrics['cumulative_return'][window] = (
                (1 + returns).rolling(window=window).apply(lambda x: np.prod(x)) - 1
            )
            
            # Benchmark-relative metrics
            if benchmark_returns is not None:
                # Correlation
                metrics['correlation'][window] = self.calculate_rolling_correlation(
                    returns, benchmark_returns, window
                )
                
                # Beta
                metrics['beta'][window] = self.calculate_beta(
                    returns, benchmark_returns, window
                )
                
                # Information ratio
                metrics['information_ratio'][window] = self.calculate_information_ratio(
                    returns, benchmark_returns, window
                )
                
                # Excess return
                metrics['excess_return'][window] = (
                    (1 + returns).rolling(window=window).apply(lambda x: np.prod(x)) -
                    (1 + benchmark_returns).rolling(window=window).apply(lambda x: np.prod(x))
                )
        
        return metrics
    
    def update_metrics(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Update performance metrics with new data.
        
        Args:
            portfolio_values: Series with portfolio values
            benchmark_values: Optional series with benchmark values
            
        Returns:
            Dictionary with current metrics
        """
        # Calculate rolling metrics
        metrics = self.calculate_rolling_metrics(portfolio_values, benchmark_values)
        
        # Store metrics history
        self.metrics_history = metrics
        
        # Extract current metrics for the latest date
        current_date = portfolio_values.index[-1]
        current_metrics = {}
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                if 'full' in metric_data:
                    current_metrics[metric_name] = metric_data['full'].loc[current_date]
                else:
                    for window, values in metric_data.items():
                        metric_key = f"{metric_name}_{window}d"
                        if len(values) > 0 and current_date in values.index:
                            current_metrics[metric_key] = values.loc[current_date]
        
        # Apply smoothing to current metrics
        for key, value in current_metrics.items():
            if key in self.smoothed_metrics:
                # Exponential smoothing
                self.smoothed_metrics[key] = (
                    self.metric_dampening * value +
                    (1 - self.metric_dampening) * self.smoothed_metrics[key]
                )
            else:
                self.smoothed_metrics[key] = value
        
        # Store current metrics
        self.current_metrics = current_metrics
        
        return current_metrics
    
    def check_alerts(self) -> List[Tuple[AlertLevel, str]]:
        """
        Check for alert conditions based on current metrics.
        
        Returns:
            List of (alert_level, message) tuples
        """
        if not self.alerts_enabled:
            return []
        
        alerts = []
        
        # Check drawdown
        if 'drawdown' in self.current_metrics and self.current_metrics['drawdown'] < self.drawdown_threshold:
            message = f"Drawdown threshold exceeded: {self.current_metrics['drawdown']:.2%}"
            alerts.append((AlertLevel.CRITICAL, message))
            logger.warning(message)
        
        # Check volatility for medium-term window (21d or closest)
        vol_key = self._find_closest_window_key('volatility', 21)
        if vol_key in self.current_metrics and self.current_metrics[vol_key] > self.vol_threshold:
            message = f"Volatility threshold exceeded: {self.current_metrics[vol_key]:.2%}"
            alerts.append((AlertLevel.WARNING, message))
            logger.warning(message)
        
        # Check Sharpe ratio for medium-term window (63d or closest)
        sharpe_key = self._find_closest_window_key('sharpe_ratio', 63)
        if sharpe_key in self.current_metrics and self.current_metrics[sharpe_key] < self.sharpe_threshold:
            message = f"Sharpe ratio below threshold: {self.current_metrics[sharpe_key]:.2f}"
            alerts.append((AlertLevel.WARNING, message))
            logger.warning(message)
        
        # Check correlation if available
        corr_key = self._find_closest_window_key('correlation', 63)
        if corr_key in self.current_metrics and abs(self.current_metrics[corr_key]) > self.correlation_threshold:
            message = f"Correlation above threshold: {self.current_metrics[corr_key]:.2f}"
            alerts.append((AlertLevel.INFO, message))
            logger.info(message)
        
        # Check performance for short-term window (5d or closest)
        perf_key = self._find_closest_window_key('cumulative_return', 5)
        if perf_key in self.current_metrics and self.current_metrics[perf_key] < self.performance_threshold:
            message = f"Short-term performance below threshold: {self.current_metrics[perf_key]:.2%}"
            alerts.append((AlertLevel.WARNING, message))
            logger.warning(message)
        
        # Store alerts
        self.alerts = alerts
        
        return alerts
    
    def _find_closest_window_key(self, metric_name: str, target_window: int) -> str:
        """
        Find the closest window key for a given metric.
        
        Args:
            metric_name: Name of the metric
            target_window: Target window size
            
        Returns:
            Key for the closest window
        """
        keys = [key for key in self.current_metrics.keys() if key.startswith(f"{metric_name}_")]
        
        if not keys:
            return f"{metric_name}_{target_window}d"
        
        closest_key = keys[0]
        closest_diff = abs(int(closest_key.split('_')[1].replace('d', '')) - target_window)
        
        for key in keys:
            window = int(key.split('_')[1].replace('d', ''))
            diff = abs(window - target_window)
            if diff < closest_diff:
                closest_key = key
                closest_diff = diff
        
        return closest_key
    
    def should_derisk(self) -> Tuple[bool, str]:
        """
        Check if portfolio de-risking should be triggered.
        
        Returns:
            Tuple of (should_derisk, reason)
        """
        # Check critical alerts first
        for level, message in self.alerts:
            if level == AlertLevel.CRITICAL:
                return True, message
        
        # Check combination of warning alerts
        warning_count = sum(1 for level, _ in self.alerts if level == AlertLevel.WARNING)
        if warning_count >= 2:
            return True, "Multiple warning indicators triggered"
        
        # Check for severe drawdown
        if 'drawdown' in self.current_metrics and self.current_metrics['drawdown'] < self.drawdown_threshold * 1.5:
            return True, f"Severe drawdown: {self.current_metrics['drawdown']:.2%}"
        
        # Check for extreme volatility
        vol_key = self._find_closest_window_key('volatility', 21)
        if vol_key in self.current_metrics and self.current_metrics[vol_key] > self.vol_threshold * 1.5:
            return True, f"Extreme volatility: {self.current_metrics[vol_key]:.2%}"
        
        return False, ""
    
    def get_derisk_factor(self) -> float:
        """
        Calculate the de-risking factor based on current metrics.
        
        Returns:
            De-risking factor between 0 and 1 (0 = full cash, 1 = no de-risking)
        """
        should_derisk, _ = self.should_derisk()
        
        if not should_derisk:
            return 1.0
        
        # Calculate de-risking factor based on drawdown
        if 'drawdown' in self.current_metrics:
            drawdown = self.current_metrics['drawdown']
            if drawdown < self.drawdown_threshold:
                # Scale from 1.0 at threshold to 0.0 at 2*threshold
                drawdown_factor = 1.0 - min(1.0, (drawdown - self.drawdown_threshold) / self.drawdown_threshold)
                return max(0.0, min(1.0, drawdown_factor))
        
        # Default moderate de-risking
        return 0.5
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of current metrics.
        
        Returns:
            Dictionary with summary metrics
        """
        report = {
            'timestamp': datetime.now(),
            'metrics': self.current_metrics,
            'smoothed_metrics': self.smoothed_metrics,
            'alerts': [(level.name, message) for level, message in self.alerts],
            'should_derisk': self.should_derisk()[0],
            'derisk_factor': self.get_derisk_factor()
        }
        
        return report 