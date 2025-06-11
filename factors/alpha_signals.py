"""
Alpha signal generator module.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from config import (RSI_WINDOW, RSI_OVERSOLD, RSI_OVERBOUGHT, 
                    SMA_FAST, SMA_SLOW, VOLATILITY_WINDOW, MOMENTUM_WINDOW)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlphaSignalGenerator:
    """
    Generates alpha signals for a set of tickers.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        rsi_window: int = RSI_WINDOW,
        sma_fast: int = SMA_FAST,
        sma_slow: int = SMA_SLOW,
        volatility_window: int = VOLATILITY_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW
    ):
        """
        Initialize the AlphaSignalGenerator.
        
        Args:
            data: DataFrame with market data
            rsi_window: Window for RSI calculation
            sma_fast: Fast SMA window
            sma_slow: Slow SMA window
            volatility_window: Window for volatility calculation
            momentum_window: Window for momentum calculation
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        # Make sure the data has the expected structure
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Data must have OHLCV columns")
            
        if 'ticker' not in data.index.names:
            raise ValueError("Data must have a multi-index with 'ticker' as one of the levels")
        
        # Store parameters
        self.data = data.copy()
        self.rsi_window = rsi_window
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        
        # Dictionary to store computed signals
        self.signals = {}
        
        logger.info(f"AlphaSignalGenerator initialized with {len(self.data.index.get_level_values('ticker').unique())} tickers")
    
    def compute_rsi(self) -> pd.DataFrame:
        """
        Compute RSI signal for all tickers.
        
        Returns:
            DataFrame with RSI signal
        """
        logger.info(f"Computing RSI signal with window={self.rsi_window}")
        
        try:
            # Group by ticker and compute RSI
            rsi_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate price changes
                    delta = group['Close'].diff()
                    
                    # Create mask for up and down days
                    up = delta.clip(lower=0)
                    down = -delta.clip(upper=0)
                    
                    # Calculate exponential moving averages
                    roll_up = up.ewm(span=self.rsi_window).mean()
                    roll_down = down.ewm(span=self.rsi_window).mean()
                    
                    # Calculate RS and RSI
                    rs = roll_up / roll_down
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                    
                    # Create a DataFrame for this ticker
                    ticker_rsi = pd.DataFrame({
                        'RSI': rsi,
                    }, index=group.index)
                    
                    # Append to result
                    rsi_values = pd.concat([rsi_values, ticker_rsi])
                    
                except Exception as e:
                    logger.warning(f"Error computing RSI for {ticker}: {str(e)}")
            
            # Ensure the index is still a MultiIndex
            if not isinstance(rsi_values.index, pd.MultiIndex):
                logger.warning("RSI values index is not a MultiIndex. Attempting to fix...")
                rsi_values.index = pd.MultiIndex.from_tuples(
                    [idx if isinstance(idx, tuple) else (idx[0], idx) for idx in rsi_values.index],
                    names=self.data.index.names
                )
            
            # Convert RSI to signal (inverse RSI: lower RSI = higher signal)
            # This buys oversold stocks (contrarian approach)
            # For values between RSI_OVERSOLD and RSI_OVERBOUGHT, scale linearly
            signal = pd.DataFrame(index=rsi_values.index)
            
            # For RSI < OVERSOLD: Strong buy signal (highest score)
            mask_oversold = rsi_values['RSI'] <= RSI_OVERSOLD
            signal.loc[mask_oversold, 'rsi_signal'] = 1.0
            
            # For RSI > OVERBOUGHT: Strong sell signal (lowest score)
            mask_overbought = rsi_values['RSI'] >= RSI_OVERBOUGHT
            signal.loc[mask_overbought, 'rsi_signal'] = 0.0
            
            # For values in between: Linear scale
            mask_middle = (~mask_oversold) & (~mask_overbought)
            signal.loc[mask_middle, 'rsi_signal'] = 1.0 - ((rsi_values.loc[mask_middle, 'RSI'] - RSI_OVERSOLD) / (RSI_OVERBOUGHT - RSI_OVERSOLD))
            
            # Store the computed signal
            self.signals['rsi'] = signal
            
            logger.info(f"RSI signal computed with shape {signal.shape}")
            return signal
            
        except Exception as e:
            logger.error(f"Error computing RSI signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def compute_sma_crossover(self) -> pd.DataFrame:
        """
        Compute SMA crossover signal for all tickers.
        
        Returns:
            DataFrame with SMA crossover signal
        """
        logger.info(f"Computing SMA crossover signal with fast={self.sma_fast}, slow={self.sma_slow}")
        
        try:
            # Group by ticker and compute SMA crossover
            sma_signals = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate fast and slow SMAs
                    fast_sma = group['Close'].rolling(window=self.sma_fast).mean()
                    slow_sma = group['Close'].rolling(window=self.sma_slow).mean()
                    
                    # Calculate distance between SMAs as percentage
                    distance = (fast_sma - slow_sma) / slow_sma * 100
                    
                    # Create a DataFrame for this ticker
                    ticker_sma = pd.DataFrame({
                        'fast_sma': fast_sma,
                        'slow_sma': slow_sma,
                        'distance': distance
                    }, index=group.index)
                    
                    # Append to result
                    sma_signals = pd.concat([sma_signals, ticker_sma])
                    
                except Exception as e:
                    logger.warning(f"Error computing SMA crossover for {ticker}: {str(e)}")
            
            # Convert to signal
            signal = pd.DataFrame(index=sma_signals.index)
            
            # Normalize distance to a signal between 0 and 1
            # First, clip extreme values to reduce outlier impact
            clipped_distance = sma_signals['distance'].clip(-5, 5)
            
            # Scale to 0-1 range (5% above SMA_SLOW = 1.0, 5% below = 0.0)
            signal['sma_signal'] = (clipped_distance + 5) / 10
            
            # Store the computed signal
            self.signals['sma'] = signal
            
            logger.info(f"SMA crossover signal computed with shape {signal.shape}")
            return signal
            
        except Exception as e:
            logger.error(f"Error computing SMA crossover signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def compute_volatility(self) -> pd.DataFrame:
        """
        Compute volatility signal for all tickers.
        
        Returns:
            DataFrame with volatility signal
        """
        logger.info(f"Computing volatility signal with window={self.volatility_window}")
        
        try:
            # Group by ticker and compute volatility
            vol_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate returns
                    returns = group['Close'].pct_change()
                    
                    # Calculate rolling volatility
                    volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)  # Annualized
                    
                    # Create a DataFrame for this ticker
                    ticker_vol = pd.DataFrame({
                        'volatility': volatility
                    }, index=group.index)
                    
                    # Append to result
                    vol_values = pd.concat([vol_values, ticker_vol])
                    
                except Exception as e:
                    logger.warning(f"Error computing volatility for {ticker}: {str(e)}")
            
            # Convert to signal
            signal = pd.DataFrame(index=vol_values.index)
            
            # Normalize volatility to a signal between 0 and 1
            # Lower volatility = higher signal (invert rank)
            # First, get cross-sectional rank of volatility by date
            rank_vol = vol_values.groupby(level='Date')['volatility'].rank(pct=True)
            
            # Invert the rank (1 - rank) so lower volatility = higher signal
            signal['volatility_signal'] = 1 - rank_vol
            
            # Store the computed signal
            self.signals['volatility'] = signal
            
            logger.info(f"Volatility signal computed with shape {signal.shape}")
            return signal
            
        except Exception as e:
            logger.error(f"Error computing volatility signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def compute_momentum(self) -> pd.DataFrame:
        """
        Compute momentum signal for all tickers.
        
        Returns:
            DataFrame with momentum signal
        """
        logger.info(f"Computing momentum signal with window={self.momentum_window}")
        
        try:
            # Group by ticker and compute momentum
            momentum_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate momentum as return over the momentum window
                    momentum = group['Close'].pct_change(periods=self.momentum_window)
                    
                    # Create a DataFrame for this ticker
                    ticker_mom = pd.DataFrame({
                        'momentum': momentum
                    }, index=group.index)
                    
                    # Append to result
                    momentum_values = pd.concat([momentum_values, ticker_mom])
                    
                except Exception as e:
                    logger.warning(f"Error computing momentum for {ticker}: {str(e)}")
            
            # Convert to signal
            signal = pd.DataFrame(index=momentum_values.index)
            
            # Normalize momentum to a signal between 0 and 1
            # First, get cross-sectional rank of momentum by date
            rank_mom = momentum_values.groupby(level='Date')['momentum'].rank(pct=True)
            
            # Use the rank directly (higher momentum = higher signal)
            signal['momentum_signal'] = rank_mom
            
            # Store the computed signal
            self.signals['momentum'] = signal
            
            logger.info(f"Momentum signal computed with shape {signal.shape}")
            return signal
            
        except Exception as e:
            logger.error(f"Error computing momentum signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def compute_liquidity(self) -> pd.DataFrame:
        """
        Compute liquidity signal for all tickers.
        
        Returns:
            DataFrame with liquidity signal
        """
        logger.info("Computing liquidity signal")
        
        try:
            # Group by ticker and compute liquidity
            liquidity_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate liquidity as dollar volume (price * volume)
                    dollar_volume = group['Close'] * group['Volume']
                    
                    # Create a DataFrame for this ticker
                    ticker_liq = pd.DataFrame({
                        'dollar_volume': dollar_volume
                    }, index=group.index)
                    
                    # Append to result
                    liquidity_values = pd.concat([liquidity_values, ticker_liq])
                    
                except Exception as e:
                    logger.warning(f"Error computing liquidity for {ticker}: {str(e)}")
            
            # Convert to signal
            signal = pd.DataFrame(index=liquidity_values.index)
            
            # Normalize liquidity to a signal between 0 and 1
            # First, get cross-sectional rank of liquidity by date
            rank_liq = liquidity_values.groupby(level='Date')['dollar_volume'].rank(pct=True)
            
            # Use the rank directly (higher liquidity = higher signal)
            signal['liquidity_signal'] = rank_liq
            
            # Store the computed signal
            self.signals['liquidity'] = signal
            
            logger.info(f"Liquidity signal computed with shape {signal.shape}")
            return signal
            
        except Exception as e:
            logger.error(f"Error computing liquidity signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def compute_gap(self) -> pd.DataFrame:
        """
        Compute overnight gap signal for all tickers.
        
        Returns:
            DataFrame with gap signal
        """
        logger.info("Computing overnight gap signal")
        
        try:
            # Group by ticker and compute gap
            gap_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate overnight gap as percentage
                    prev_close = group['Close'].shift(1)
                    gap = (group['Open'] - prev_close) / prev_close * 100
                    
                    # Create a DataFrame for this ticker
                    ticker_gap = pd.DataFrame({
                        'gap': gap
                    }, index=group.index)
                    
                    # Append to result
                    gap_values = pd.concat([gap_values, ticker_gap])
                    
                except Exception as e:
                    logger.warning(f"Error computing gap for {ticker}: {str(e)}")
            
            # Convert to signal
            signal = pd.DataFrame(index=gap_values.index)
            
            # Normalize gap to a signal between 0 and 1
            # Inverse gap - we want to buy negative gaps (mean reversion)
            # First, get cross-sectional rank of gap by date
            rank_gap = gap_values.groupby(level='Date')['gap'].rank(pct=True)
            
            # Invert the rank (1 - rank) so negative gaps = higher signal
            signal['gap_signal'] = 1 - rank_gap
            
            # Store the computed signal
            self.signals['gap'] = signal
            
            logger.info(f"Gap signal computed with shape {signal.shape}")
            return signal
            
        except Exception as e:
            logger.error(f"Error computing gap signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def compute_price_percentile(self) -> pd.DataFrame:
        """
        Compute price percentile signal (current price relative to 52-week range).
        
        Returns:
            DataFrame with price percentile signal
        """
        logger.info("Computing price percentile signal")
        
        try:
            # Group by ticker and compute price percentile
            percentile_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate 52-week high and low (approximately 252 trading days)
                    rolling_max = group['Close'].rolling(window=252, min_periods=20).max()
                    rolling_min = group['Close'].rolling(window=252, min_periods=20).min()
                    
                    # Calculate percentile (0 = at 52-week low, 1 = at 52-week high)
                    percentile = (group['Close'] - rolling_min) / (rolling_max - rolling_min)
                    
                    # Create inverse percentile (contrarian approach - buy low)
                    # Higher signal = lower price percentile
                    inverse_percentile = 1 - percentile
                    
                    # Create a DataFrame for this ticker
                    ticker_percentile = pd.DataFrame({
                        'price_percentile_signal': inverse_percentile
                    }, index=group.index)
                    
                    # Append to result
                    percentile_values = pd.concat([percentile_values, ticker_percentile])
                    
                except Exception as e:
                    logger.warning(f"Error computing price percentile for {ticker}: {str(e)}")
            
            # Handle NaN values
            percentile_values = percentile_values.fillna(0.5)  # Neutral signal for NaN
            
            # Store the computed signal
            self.signals['price_percentile'] = percentile_values
            
            logger.info(f"Price percentile signal computed with shape {percentile_values.shape}")
            return percentile_values
            
        except Exception as e:
            logger.error(f"Error computing price percentile signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def compute_roc(self) -> pd.DataFrame:
        """
        Compute Rate of Change (ROC) signal.
        
        Returns:
            DataFrame with ROC signal
        """
        logger.info(f"Computing ROC signal with window={self.momentum_window}")
        
        try:
            # Group by ticker and compute ROC
            roc_values = pd.DataFrame()
            
            for ticker, group in self.data.groupby(level='ticker'):
                try:
                    # Calculate Rate of Change
                    roc = group['Close'].pct_change(periods=self.momentum_window)
                    
                    # Normalize ROC to 0-1 range
                    # Clip extreme values to reduce outlier impact
                    clipped_roc = roc.clip(-0.5, 0.5)
                    normalized_roc = (clipped_roc + 0.5) / 1.0
                    
                    # Create a DataFrame for this ticker
                    ticker_roc = pd.DataFrame({
                        'roc_signal': normalized_roc
                    }, index=group.index)
                    
                    # Append to result
                    roc_values = pd.concat([roc_values, ticker_roc])
                    
                except Exception as e:
                    logger.warning(f"Error computing ROC for {ticker}: {str(e)}")
            
            # Handle NaN values
            roc_values = roc_values.fillna(0.5)  # Neutral signal for NaN
            
            # Store the computed signal
            self.signals['roc'] = roc_values
            
            logger.info(f"ROC signal computed with shape {roc_values.shape}")
            return roc_values
            
        except Exception as e:
            logger.error(f"Error computing ROC signal: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
    def apply_signal_smoothing(self, signal: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Apply exponential moving average smoothing to signals.
        
        Args:
            signal: DataFrame with signals
            window: Window size for EMA
            
        Returns:
            Smoothed signals DataFrame
        """
        logger.info(f"Applying signal smoothing with window={window}")
        
        try:
            smoothed = pd.DataFrame(index=signal.index)
            
            for col in signal.columns:
                # Group by ticker and apply EMA
                col_smoothed = pd.Series(index=signal.index, dtype=float)
                
                for ticker, group in signal.groupby(level='ticker'):
                    # Apply EMA
                    ema = group[col].ewm(span=window, min_periods=1).mean()
                    col_smoothed.loc[group.index] = ema
                
                smoothed[col] = col_smoothed
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error applying signal smoothing: {str(e)}")
            logger.error(traceback.format_exc())
            return signal  # Return original signal if smoothing fails
        
    def compute_all_signals(self, factor_weights: Optional[Dict[str, float]] = None, smoothing_window: int = 5) -> pd.DataFrame:
        """
        Compute all signals and combine them into a final score.
        
        Args:
            factor_weights: Dictionary of factor weights
            smoothing_window: Window for signal smoothing
            
        Returns:
            DataFrame with combined signals
        """
        logger.info("Computing all alpha signals")
        
        # Default weights if none provided
        if factor_weights is None:
            factor_weights = {
                'rsi': 0.25,        # Higher weight for RSI (contrarian)
                'sma': 0.20,        # Medium weight for trend following
                'volatility': -0.15, # Lower volatility is preferred (negative weight)
                'momentum': 0.20,   # Medium weight for momentum
                'liquidity': 0.10,  # Lower weight for liquidity
                'gap': 0.10,        # Lower weight for gap (mean reversion)
                'price_percentile': 0.20, # Medium weight for price percentile
                'roc': 0.15         # Medium weight for rate of change
            }
        
        try:
            # Compute all signals
            rsi_signal = self.compute_rsi()
            sma_signal = self.compute_sma_crossover()
            volatility_signal = self.compute_volatility()
            momentum_signal = self.compute_momentum()
            liquidity_signal = self.compute_liquidity()
            gap_signal = self.compute_gap()
            price_percentile_signal = self.compute_price_percentile()
            roc_signal = self.compute_roc()
            
            # Combine signals
            combined_signals = pd.DataFrame(index=self.data.index)
            
            # Add each signal with its weight if it exists and weight is non-zero
            if not rsi_signal.empty and factor_weights.get('rsi', 0) != 0:
                combined_signals['rsi_signal'] = rsi_signal['rsi_signal'] * factor_weights['rsi']
            
            if not sma_signal.empty and factor_weights.get('sma', 0) != 0:
                combined_signals['sma_signal'] = sma_signal['sma_signal'] * factor_weights['sma']
            
            if not volatility_signal.empty and factor_weights.get('volatility', 0) != 0:
                # Note: volatility can have negative weight (prefer lower volatility)
                combined_signals['volatility_signal'] = volatility_signal['volatility_signal'] * factor_weights['volatility']
            
            if not momentum_signal.empty and factor_weights.get('momentum', 0) != 0:
                combined_signals['momentum_signal'] = momentum_signal['momentum_signal'] * factor_weights['momentum']
            
            if not liquidity_signal.empty and factor_weights.get('liquidity', 0) != 0:
                combined_signals['liquidity_signal'] = liquidity_signal['liquidity_signal'] * factor_weights['liquidity']
            
            if not gap_signal.empty and factor_weights.get('gap', 0) != 0:
                combined_signals['gap_signal'] = gap_signal['gap_signal'] * factor_weights['gap']
                
            if not price_percentile_signal.empty and factor_weights.get('price_percentile', 0) != 0:
                combined_signals['price_percentile_signal'] = price_percentile_signal['price_percentile_signal'] * factor_weights['price_percentile']
                
            if not roc_signal.empty and factor_weights.get('roc', 0) != 0:
                combined_signals['roc_signal'] = roc_signal['roc_signal'] * factor_weights['roc']
            
            # Drop rows with NaNs in any signal
            combined_signals = combined_signals.dropna()
            
            # Apply smoothing to reduce noise
            if smoothing_window > 1:
                combined_signals = self.apply_signal_smoothing(combined_signals, window=smoothing_window)
            
            # Calculate weighted average for final score
            sum_weights = sum(abs(w) for w in factor_weights.values() if w != 0)
            if sum_weights > 0:
                combined_signals['final_score'] = combined_signals.sum(axis=1) / sum_weights
            else:
                combined_signals['final_score'] = combined_signals.mean(axis=1)
            
            # Ensure final score is normalized between 0 and 1
            min_score = combined_signals['final_score'].min()
            max_score = combined_signals['final_score'].max()
            if max_score > min_score:
                combined_signals['final_score'] = (combined_signals['final_score'] - min_score) / (max_score - min_score)
            
            # Make sure index is sorted
            combined_signals = combined_signals.sort_index()
            
            logger.info(f"Combined signals computed with shape {combined_signals.shape}")
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error computing all signals: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    from data.market_data import MarketDataFetcher
    
    # Fetch data
    fetcher = MarketDataFetcher()
    data = fetcher.load_historical_data()
    
    # Generate signals
    generator = AlphaSignalGenerator(data)
    signals = generator.compute_all_signals()
    
    # Print head of signals
    print(signals.head()) 