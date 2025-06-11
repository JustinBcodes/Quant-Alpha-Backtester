"""
Market regime detection module.

This module implements market regime detection using:
1. Trend classification (bullish, bearish, sideways)
2. Volatility regime classification (low, medium, high)
3. Combined market regime states
4. VIX-based volatility detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class TrendRegime(Enum):
    """Trend regime enumeration."""
    BULLISH = 1
    SIDEWAYS = 0
    BEARISH = -1

class VolatilityRegime(Enum):
    """Volatility regime enumeration."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class MarketRegimeDetector:
    """
    Detects market regimes based on trend and volatility indicators.
    """
    
    def __init__(
        self,
        trend_window: int = 60,
        trend_threshold: float = 0.05,
        vol_window: int = 21,
        vol_lookback: int = 252,
        vol_percentile_low: float = 0.33,
        vol_percentile_high: float = 0.67,
        smooth_window: int = 5,
        vix_high_threshold: float = 25.0,
        rolling_std_high_threshold: float = 0.02
    ):
        """
        Initialize the MarketRegimeDetector.
        
        Args:
            trend_window: Rolling window for trend detection
            trend_threshold: Threshold for trend classification
            vol_window: Rolling window for volatility calculation
            vol_lookback: Lookback period for volatility percentile calculation
            vol_percentile_low: Percentile threshold for low volatility
            vol_percentile_high: Percentile threshold for high volatility
            smooth_window: Window for regime smoothing
            vix_high_threshold: VIX threshold for high volatility regime
            rolling_std_high_threshold: Rolling std threshold for high volatility
        """
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.vol_percentile_low = vol_percentile_low
        self.vol_percentile_high = vol_percentile_high
        self.smooth_window = smooth_window
        self.vix_high_threshold = vix_high_threshold
        self.rolling_std_high_threshold = rolling_std_high_threshold
        
        # Cache for regimes
        self.trend_regimes = None
        self.vol_regimes = None
        self.combined_regimes = None
        self.use_mean_reversion = None
        
        logger.info(
            f"MarketRegimeDetector initialized with trend_window={trend_window}, "
            f"vol_window={vol_window}, vol_lookback={vol_lookback}, "
            f"vix_high_threshold={vix_high_threshold}, "
            f"rolling_std_high_threshold={rolling_std_high_threshold}"
        )
    
    def detect_trend_regime(self, prices: pd.Series) -> pd.Series:
        """
        Detect trend regime using price momentum and moving averages.
        
        Args:
            prices: Series of price data for a market index
            
        Returns:
            Series with trend regime values
        """
        try:
            # Calculate short-term and long-term momentum
            returns = prices.pct_change()
            momentum = prices / prices.shift(self.trend_window) - 1
            
            # Calculate moving averages
            sma_short = prices.rolling(window=self.trend_window//3).mean()
            sma_long = prices.rolling(window=self.trend_window).mean()
            
            # Trend indicator: combines momentum and moving average crossover
            trend_indicator = pd.Series(index=prices.index, dtype=float)
            
            # Bullish: price momentum positive and short MA > long MA
            bullish = (momentum > self.trend_threshold) & (sma_short > sma_long)
            trend_indicator[bullish] = TrendRegime.BULLISH.value
            
            # Bearish: price momentum negative and short MA < long MA
            bearish = (momentum < -self.trend_threshold) & (sma_short < sma_long)
            trend_indicator[bearish] = TrendRegime.BEARISH.value
            
            # Sideways: all other conditions
            sideways = ~(bullish | bearish)
            trend_indicator[sideways] = TrendRegime.SIDEWAYS.value
            
            # Apply smoothing to reduce regime flipping
            if self.smooth_window > 1:
                # Mode smoothing (most common regime in the window)
                trend_indicator = trend_indicator.rolling(
                    window=self.smooth_window
                ).apply(lambda x: pd.Series(x).mode()[0])
            
            # Cache the result
            self.trend_regimes = trend_indicator
            
            return trend_indicator
            
        except Exception as e:
            logger.error(f"Error detecting trend regime: {str(e)}")
            return pd.Series(index=prices.index, data=TrendRegime.SIDEWAYS.value)
    
    def detect_volatility_regime(
        self, 
        prices: pd.Series, 
        vix_data: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Detect volatility regime using rolling standard deviation and VIX when available.
        
        Args:
            prices: Series of price data for a market index
            vix_data: Optional VIX index data
            
        Returns:
            Series with volatility regime values
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate rolling volatility (annualized)
            rolling_vol = returns.rolling(window=self.vol_window).std() * np.sqrt(252)
            
            # Initialize volatility regimes
            vol_regimes = pd.Series(index=rolling_vol.index, dtype=int)
            
            # Use VIX data if available
            if vix_data is not None and not vix_data.empty:
                # Align VIX data with price data
                aligned_vix = vix_data.reindex(rolling_vol.index, method='ffill')
                
                # High volatility when VIX > threshold
                high_vol_vix = aligned_vix > self.vix_high_threshold
                vol_regimes[high_vol_vix] = VolatilityRegime.HIGH.value
                
                # For remaining points, use rolling volatility
                remaining_points = ~high_vol_vix
                
                # High volatility when rolling std > threshold
                high_vol_std = rolling_vol > self.rolling_std_high_threshold
                vol_regimes[remaining_points & high_vol_std] = VolatilityRegime.HIGH.value
                
                # Process remaining points with percentile method
                remaining_percentile = remaining_points & ~high_vol_std
                
                for i in rolling_vol[remaining_percentile].index:
                    # Find position in the timeseries
                    pos = rolling_vol.index.get_loc(i)
                    
                    # Skip if not enough history
                    if pos < self.vol_lookback:
                        vol_regimes.loc[i] = VolatilityRegime.MEDIUM.value
                        continue
                    
                    # Get the lookback window
                    lookback_start = pos - self.vol_lookback
                    lookback_vol = rolling_vol.iloc[lookback_start:pos]
                    
                    # Current volatility
                    current_vol = rolling_vol.loc[i]
                    
                    # Calculate percentiles from the lookback window
                    low_threshold = lookback_vol.quantile(self.vol_percentile_low)
                    
                    # Classify remaining regimes as LOW or MEDIUM
                    if current_vol <= low_threshold:
                        vol_regimes.loc[i] = VolatilityRegime.LOW.value
                    else:
                        vol_regimes.loc[i] = VolatilityRegime.MEDIUM.value
            else:
                # No VIX data, use only rolling volatility
                for i in range(self.vol_lookback, len(rolling_vol)):
                    # Get the lookback window
                    lookback_start = i - self.vol_lookback
                    lookback_vol = rolling_vol.iloc[lookback_start:i]
                    
                    # Current volatility
                    current_vol = rolling_vol.iloc[i]
                    
                    # Calculate percentiles from the lookback window
                    low_threshold = lookback_vol.quantile(self.vol_percentile_low)
                    high_threshold = lookback_vol.quantile(self.vol_percentile_high)
                    
                    # Classify regime
                    if current_vol <= low_threshold:
                        vol_regimes.iloc[i] = VolatilityRegime.LOW.value
                    elif current_vol >= high_threshold:
                        vol_regimes.iloc[i] = VolatilityRegime.HIGH.value
                    else:
                        vol_regimes.iloc[i] = VolatilityRegime.MEDIUM.value
            
            # Fill NaN values with MEDIUM regime
            vol_regimes = vol_regimes.fillna(VolatilityRegime.MEDIUM.value)
            
            # Apply smoothing to reduce regime flipping
            if self.smooth_window > 1:
                # Mode smoothing (most common regime in the window)
                vol_regimes = vol_regimes.rolling(
                    window=self.smooth_window
                ).apply(lambda x: pd.Series(x).mode()[0])
                # Fill NaN values again after smoothing
                vol_regimes = vol_regimes.fillna(VolatilityRegime.MEDIUM.value)
            
            # Cache the result
            self.vol_regimes = vol_regimes
            
            return vol_regimes
            
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {str(e)}")
            return pd.Series(index=prices.index, data=VolatilityRegime.MEDIUM.value)
    
    def detect_combined_regime(
        self, 
        prices: pd.Series, 
        vix_data: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Detect combined market regime using both trend and volatility.
        
        Args:
            prices: Series of price data for a market index
            vix_data: Optional VIX index data
            
        Returns:
            DataFrame with trend, volatility, and combined regime values
        """
        # Detect individual regimes
        trend_regime = self.detect_trend_regime(prices)
        vol_regime = self.detect_volatility_regime(prices, vix_data)
        
        # Combine regimes into a single DataFrame
        combined = pd.DataFrame({
            'trend_regime': trend_regime,
            'volatility_regime': vol_regime
        })
        
        # Create a combined regime indicator
        # Format: <trend>_<volatility> (e.g., "bullish_low_vol")
        def combine_regimes(row):
            try:
                # Handle potential NaN values
                if pd.isna(row['trend_regime']) or pd.isna(row['volatility_regime']):
                    return "sideways_medium_vol"  # Default regime
                
                trend = TrendRegime(int(row['trend_regime'])).name.lower()
                vol = VolatilityRegime(int(row['volatility_regime'])).name.lower()
                return f"{trend}_{vol}_vol"
            except (ValueError, TypeError):
                return "sideways_medium_vol"  # Default regime
                
        combined['combined_regime'] = combined.apply(combine_regimes, axis=1)
        
        # Determine whether to use mean reversion strategy
        combined['use_mean_reversion'] = (
            (combined['volatility_regime'] == VolatilityRegime.HIGH.value) | 
            (combined['trend_regime'] == TrendRegime.SIDEWAYS.value)
        )
        
        # Handle NaN values in use_mean_reversion column
        combined['use_mean_reversion'] = combined['use_mean_reversion'].fillna(False)
        
        # Cache the result
        self.combined_regimes = combined
        self.use_mean_reversion = combined['use_mean_reversion']
        
        return combined
    
    def get_current_regime(
        self, 
        prices: pd.Series, 
        vix_data: Optional[pd.Series] = None
    ) -> Dict[str, str]:
        """
        Get the current market regime.
        
        Args:
            prices: Series of price data for a market index
            vix_data: Optional VIX index data
            
        Returns:
            Dictionary with current trend, volatility, and combined regimes
        """
        # Detect regimes if not already cached
        if self.combined_regimes is None or len(self.combined_regimes) != len(prices):
            self.detect_combined_regime(prices, vix_data)
        
        # Get the most recent regime
        latest_regime = self.combined_regimes.iloc[-1]
        
        return {
            'trend': TrendRegime(int(latest_regime['trend_regime'])).name,
            'volatility': VolatilityRegime(int(latest_regime['volatility_regime'])).name,
            'combined': latest_regime['combined_regime'],
            'use_mean_reversion': bool(latest_regime['use_mean_reversion'])
        }
    
    def should_use_mean_reversion(
        self, 
        prices: pd.Series = None, 
        vix_value: Optional[float] = None,
        rolling_std: Optional[float] = None
    ) -> bool:
        """
        Determine if the current market regime favors mean reversion strategies.
        
        Args:
            prices: Optional price series (if regime detection needed)
            vix_value: Optional current VIX value
            rolling_std: Optional current rolling standard deviation
            
        Returns:
            Boolean indicating whether to use mean reversion strategies
        """
        # Direct check based on provided metrics
        if vix_value is not None and vix_value > self.vix_high_threshold:
            return True
            
        if rolling_std is not None and rolling_std > self.rolling_std_high_threshold:
            return True
            
        # If no direct metrics, use the cached regime
        if self.use_mean_reversion is not None and len(self.use_mean_reversion) > 0:
            return bool(self.use_mean_reversion.iloc[-1])
            
        # If no cache and no metrics, detect regime
        if prices is not None:
            regime = self.get_current_regime(prices)
            return regime['use_mean_reversion']
            
        # Default to momentum (not mean reversion)
        return False
    
    def get_optimal_factor_weights(self, current_regime: Dict[str, str]) -> Dict[str, float]:
        """
        Get optimal factor weights for the current market regime.
        
        Args:
            current_regime: Dictionary with current market regime
            
        Returns:
            Dictionary with factor weights optimized for the regime
        """
        # Define regime-specific factor weights
        regime_weights = {
            # Bullish regimes
            'BULLISH_LOW_vol': {
                'momentum': 0.3,
                'rsi': 0.1,
                'sma': 0.2,
                'volatility': 0.1,
                'liquidity': 0.1,
                'gap': 0.05,
                'price_percentile': 0.05,
                'roc': 0.1
            },
            'BULLISH_MEDIUM_vol': {
                'momentum': 0.25,
                'rsi': 0.15,
                'sma': 0.2,
                'volatility': 0.15,
                'liquidity': 0.1,
                'gap': 0.05,
                'price_percentile': 0.05,
                'roc': 0.05
            },
            'BULLISH_HIGH_vol': {
                'momentum': 0.05,
                'rsi': 0.25,
                'sma': 0.1,
                'volatility': 0.25,
                'liquidity': 0.15,
                'gap': 0.1,
                'price_percentile': 0.05,
                'roc': 0.05
            },
            
            # Sideways regimes
            'SIDEWAYS_LOW_vol': {
                'momentum': 0.1,
                'rsi': 0.2,
                'sma': 0.1,
                'volatility': 0.2,
                'liquidity': 0.1,
                'gap': 0.1,
                'price_percentile': 0.1,
                'roc': 0.1
            },
            'SIDEWAYS_MEDIUM_vol': {
                'momentum': 0.1,
                'rsi': 0.2,
                'sma': 0.1,
                'volatility': 0.2,
                'liquidity': 0.15,
                'gap': 0.1,
                'price_percentile': 0.1,
                'roc': 0.05
            },
            'SIDEWAYS_HIGH_vol': {
                'momentum': 0.05,
                'rsi': 0.25,
                'sma': 0.05,
                'volatility': 0.3,
                'liquidity': 0.2,
                'gap': 0.1,
                'price_percentile': 0.05,
                'roc': 0.0
            },
            
            # Bearish regimes
            'BEARISH_LOW_vol': {
                'momentum': 0.1,
                'rsi': 0.2,
                'sma': 0.1,
                'volatility': 0.2,
                'liquidity': 0.1,
                'gap': 0.1,
                'price_percentile': 0.15,
                'roc': 0.05
            },
            'BEARISH_MEDIUM_vol': {
                'momentum': 0.05,
                'rsi': 0.25,
                'sma': 0.05,
                'volatility': 0.25,
                'liquidity': 0.15,
                'gap': 0.15,
                'price_percentile': 0.1,
                'roc': 0.0
            },
            'BEARISH_HIGH_vol': {
                'momentum': 0.0,
                'rsi': 0.3,
                'sma': 0.0,
                'volatility': 0.3,
                'liquidity': 0.2,
                'gap': 0.15,
                'price_percentile': 0.05,
                'roc': 0.0
            }
        }
        
        # Get weights for current regime
        combined_key = f"{current_regime['trend']}_{current_regime['volatility']}_vol"
        
        if combined_key in regime_weights:
            return regime_weights[combined_key]
        else:
            # Default weights if regime is not recognized
            logger.warning(f"Unknown regime: {combined_key}, using default weights")
            return {
                'momentum': 0.15,
                'rsi': 0.15,
                'sma': 0.15,
                'volatility': 0.15,
                'liquidity': 0.15,
                'gap': 0.1,
                'price_percentile': 0.1,
                'roc': 0.05
            }
            
    def get_signal_weights_by_regime(
        self,
        use_mean_reversion: bool
    ) -> Dict[str, float]:
        """
        Get signal weights based on whether to use mean reversion or momentum strategy.
        
        Args:
            use_mean_reversion: Whether to use mean reversion strategy
            
        Returns:
            Dictionary with signal weights
        """
        if use_mean_reversion:
            # Mean reversion strategy weights - emphasize RSI, volatility, gap
            return {
                'momentum': 0.0,
                'rsi': 0.3,        # Higher weight for RSI (mean reversion)
                'sma': 0.05,
                'volatility': 0.3, # Higher weight for volatility
                'liquidity': 0.15,
                'gap': 0.15,       # Higher weight for gap
                'price_percentile': 0.05,
                'roc': 0.0
            }
        else:
            # Momentum strategy weights - emphasize momentum, SMA, ROC
            return {
                'momentum': 0.3,   # Higher weight for momentum
                'rsi': 0.1,
                'sma': 0.2,        # Higher weight for SMA crossover
                'volatility': 0.1,
                'liquidity': 0.1,
                'gap': 0.05,
                'price_percentile': 0.05,
                'roc': 0.1         # Higher weight for ROC
            } 