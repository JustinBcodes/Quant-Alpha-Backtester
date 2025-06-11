"""
Position constraints module.

This module implements:
1. Sector and ticker exposure limits
2. Volatility targeting
3. Risk-based position sizing
4. Performance-based portfolio de-risking
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class PositionSizingMethod(Enum):
    """Position sizing method enumeration."""
    EQUAL_WEIGHT = 0
    MARKET_CAP_WEIGHT = 1
    INVERSE_VOLATILITY = 2
    EQUAL_RISK_CONTRIBUTION = 3
    SIGNAL_WEIGHTED = 4
    OPTIMAL_SHARPE = 5

class ConstraintsManager:
    """
    Manages position constraints and risk targeting.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.10,  # 10% annualized
        max_stock_weight: float = 0.20,  # 20% max per stock
        max_sector_weight: float = 0.40,  # 40% max per sector
        min_stocks: int = 5,
        max_stocks: int = 20,
        position_sizing: Union[str, PositionSizingMethod] = 'equal',
        vol_lookback: int = 63,
        vol_max_adjustment: float = 2.0,  # Max leverage multiplier
        sector_neutral: bool = False,
        min_position_weight: float = 0.01  # 1% minimum position size
    ):
        """
        Initialize the ConstraintsManager.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            max_stock_weight: Maximum weight per stock
            max_sector_weight: Maximum weight per sector
            min_stocks: Minimum number of stocks to hold
            max_stocks: Maximum number of stocks to hold
            position_sizing: Method for position sizing
            vol_lookback: Lookback window for volatility calculation
            vol_max_adjustment: Maximum volatility adjustment factor
            sector_neutral: Whether to enforce sector neutrality
            min_position_weight: Minimum weight per position
        """
        self.target_volatility = target_volatility
        self.max_stock_weight = max_stock_weight
        self.max_sector_weight = max_sector_weight
        self.min_stocks = min_stocks
        self.max_stocks = max_stocks
        self.vol_lookback = vol_lookback
        self.vol_max_adjustment = vol_max_adjustment
        self.sector_neutral = sector_neutral
        self.min_position_weight = min_position_weight
        
        # Parse position sizing method
        if isinstance(position_sizing, str):
            position_sizing = position_sizing.lower()
            if position_sizing == 'equal':
                self.position_sizing = PositionSizingMethod.EQUAL_WEIGHT
            elif position_sizing == 'market_cap':
                self.position_sizing = PositionSizingMethod.MARKET_CAP_WEIGHT
            elif position_sizing == 'inverse_vol':
                self.position_sizing = PositionSizingMethod.INVERSE_VOLATILITY
            elif position_sizing == 'risk_parity':
                self.position_sizing = PositionSizingMethod.EQUAL_RISK_CONTRIBUTION
            elif position_sizing == 'signal_weighted':
                self.position_sizing = PositionSizingMethod.SIGNAL_WEIGHTED
            elif position_sizing == 'optimal_sharpe':
                self.position_sizing = PositionSizingMethod.OPTIMAL_SHARPE
            else:
                logger.warning(f"Unknown position sizing method: {position_sizing}. Using equal weight.")
                self.position_sizing = PositionSizingMethod.EQUAL_WEIGHT
        else:
            self.position_sizing = position_sizing
        
        logger.info(
            f"ConstraintsManager initialized with target_vol={target_volatility:.1%}, "
            f"max_stock={max_stock_weight:.1%}, max_sector={max_sector_weight:.1%}"
        )
    
    def calculate_stock_volatility(
        self,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> pd.Series:
        """
        Calculate annualized volatility for each stock.
        
        Args:
            returns: DataFrame with daily returns for each stock
            current_date: Current date for calculation
            
        Returns:
            Series with annualized volatility for each stock
        """
        # Get data up to current date
        past_returns = returns.loc[:current_date].copy()
        
        # Calculate rolling volatility (annualized)
        vol = past_returns.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        
        # Get the most recent volatility
        latest_vol = vol.iloc[-1]
        
        return latest_vol
    
    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for stock returns.
        
        Args:
            returns: DataFrame with daily returns for each stock
            current_date: Current date for calculation
            
        Returns:
            DataFrame with correlation matrix
        """
        # Get data up to current date
        past_returns = returns.loc[:current_date].copy()
        
        # Calculate correlation over the lookback period
        lookback_start = max(0, len(past_returns) - self.vol_lookback)
        recent_returns = past_returns.iloc[lookback_start:]
        
        # Calculate correlation matrix
        corr_matrix = recent_returns.corr()
        
        return corr_matrix
    
    def calculate_portfolio_volatility(
        self,
        weights: pd.Series,
        volatilities: pd.Series,
        correlation_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio volatility based on weights, volatilities, and correlations.
        
        Args:
            weights: Series with portfolio weights
            volatilities: Series with stock volatilities
            correlation_matrix: Correlation matrix for stocks
            
        Returns:
            Portfolio volatility (annualized)
        """
        # Align indices
        aligned_weights = weights.reindex(volatilities.index, fill_value=0)
        aligned_corr = correlation_matrix.reindex(
            index=volatilities.index,
            columns=volatilities.index,
            fill_value=0
        )
        
        # Create covariance matrix
        vol_matrix = np.diag(volatilities)
        cov_matrix = vol_matrix @ aligned_corr @ vol_matrix
        
        # Calculate portfolio volatility
        port_vol = np.sqrt(aligned_weights.T @ cov_matrix @ aligned_weights)
        
        return port_vol
    
    def get_sector_allocations(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate sector allocations from stock weights.
        
        Args:
            weights: Series with stock weights
            sector_map: Dictionary mapping tickers to sectors
            
        Returns:
            Dictionary with sector allocations
        """
        sector_weights = {}
        
        for ticker, weight in weights.items():
            if ticker in sector_map:
                sector = sector_map[ticker]
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weight
        
        return sector_weights
    
    def apply_sector_constraints(
        self,
        weights: pd.Series,
        sector_map: Dict[str, str]
    ) -> pd.Series:
        """
        Apply sector constraints to portfolio weights.
        
        Args:
            weights: Series with portfolio weights
            sector_map: Dictionary mapping tickers to sectors
            
        Returns:
            Series with constrained weights
        """
        constrained_weights = weights.copy()
        
        # Calculate current sector allocations
        sector_weights = self.get_sector_allocations(weights, sector_map)
        
        # Check for sectors exceeding the limit
        excess_sectors = {
            sector: weight for sector, weight in sector_weights.items()
            if weight > self.max_sector_weight
        }
        
        if not excess_sectors:
            return constrained_weights
        
        # Reduce weights for stocks in sectors that exceed the limit
        for sector, excess_weight in excess_sectors.items():
            reduction_factor = self.max_sector_weight / excess_weight
            
            # Get stocks in this sector
            sector_stocks = [
                ticker for ticker in weights.index
                if sector_map.get(ticker) == sector
            ]
            
            # Reduce weights proportionally
            for ticker in sector_stocks:
                constrained_weights[ticker] *= reduction_factor
        
        # Renormalize weights
        if constrained_weights.sum() > 0:
            constrained_weights = constrained_weights / constrained_weights.sum()
        
        return constrained_weights
    
    def apply_max_stock_constraint(self, weights: pd.Series) -> pd.Series:
        """
        Apply maximum stock weight constraint.
        
        Args:
            weights: Series with portfolio weights
            
        Returns:
            Series with constrained weights
        """
        constrained_weights = weights.copy()
        
        # Find stocks exceeding the limit
        excess_stocks = constrained_weights[constrained_weights > self.max_stock_weight]
        
        if excess_stocks.empty:
            return constrained_weights
        
        # Cap excess stocks at max weight
        constrained_weights[excess_stocks.index] = self.max_stock_weight
        
        # Redistribute excess weight to other stocks proportionally
        excess_weight = excess_stocks.sum() - (self.max_stock_weight * len(excess_stocks))
        non_excess_stocks = constrained_weights.index.difference(excess_stocks.index)
        
        if len(non_excess_stocks) > 0 and excess_weight > 0:
            # Calculate the total weight of non-excess stocks
            non_excess_total = constrained_weights[non_excess_stocks].sum()
            
            if non_excess_total > 0:
                # Distribute excess weight proportionally
                for ticker in non_excess_stocks:
                    proportion = constrained_weights[ticker] / non_excess_total
                    constrained_weights[ticker] += excess_weight * proportion
        
        # Renormalize weights
        if constrained_weights.sum() > 0:
            constrained_weights = constrained_weights / constrained_weights.sum()
        
        return constrained_weights
    
    def apply_min_position_constraint(self, weights: pd.Series) -> pd.Series:
        """
        Apply minimum position weight constraint.
        
        Args:
            weights: Series with portfolio weights
            
        Returns:
            Series with constrained weights
        """
        constrained_weights = weights.copy()
        
        # Find positions below the minimum
        small_positions = constrained_weights[
            (constrained_weights > 0) & (constrained_weights < self.min_position_weight)
        ]
        
        if small_positions.empty:
            return constrained_weights
        
        # Remove positions that are too small
        constrained_weights[small_positions.index] = 0
        
        # Renormalize weights
        if constrained_weights.sum() > 0:
            constrained_weights = constrained_weights / constrained_weights.sum()
        
        return constrained_weights
    
    def calculate_position_sizes(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        current_date: pd.Timestamp,
        market_caps: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Calculate position sizes based on the chosen method.
        
        Args:
            signals: Series with alpha signals for each stock
            returns: DataFrame with historical returns
            current_date: Current date for calculation
            market_caps: Optional series with market capitalizations
            sector_map: Optional dictionary mapping tickers to sectors
            
        Returns:
            Series with position sizes (weights)
        """
        # Sort tickers by signal
        sorted_tickers = signals.sort_values(ascending=False).index
        
        # Limit to max_stocks
        selected_tickers = sorted_tickers[:self.max_stocks]
        
        # Calculate volatility for each stock
        volatilities = self.calculate_stock_volatility(returns, current_date)
        
        # Align volatilities with selected tickers
        selected_volatilities = volatilities.reindex(selected_tickers)
        
        # Choose position sizing method
        if self.position_sizing == PositionSizingMethod.EQUAL_WEIGHT:
            # Equal weight allocation
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
            
        elif self.position_sizing == PositionSizingMethod.MARKET_CAP_WEIGHT:
            if market_caps is None:
                logger.warning("Market cap data not provided. Using equal weight.")
                weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
            else:
                # Market cap weighted allocation
                selected_mcaps = market_caps.reindex(selected_tickers)
                weights = selected_mcaps / selected_mcaps.sum()
                
        elif self.position_sizing == PositionSizingMethod.INVERSE_VOLATILITY:
            # Inverse volatility weighted allocation
            inverse_vol = 1.0 / selected_volatilities
            weights = inverse_vol / inverse_vol.sum()
            
        elif self.position_sizing == PositionSizingMethod.SIGNAL_WEIGHTED:
            # Signal strength weighted allocation
            selected_signals = signals.reindex(selected_tickers)
            
            # Ensure signals are positive
            min_signal = selected_signals.min()
            if min_signal < 0:
                selected_signals = selected_signals - min_signal + 0.01
                
            weights = selected_signals / selected_signals.sum()
            
        elif self.position_sizing == PositionSizingMethod.EQUAL_RISK_CONTRIBUTION:
            # Equal risk contribution (risk parity)
            correlation_matrix = self.calculate_correlation_matrix(returns, current_date)
            
            # Create covariance matrix for selected stocks
            selected_corr = correlation_matrix.reindex(
                index=selected_tickers, 
                columns=selected_tickers
            )
            vol_matrix = np.diag(selected_volatilities)
            cov_matrix = vol_matrix @ selected_corr @ vol_matrix
            
            # Calculate risk contributions
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
            
            # Simple risk parity approximation
            for _ in range(10):  # Iterative approximation
                port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                marginal_contrib = (cov_matrix @ weights) / port_vol
                risk_contrib = weights * marginal_contrib
                
                # Update weights to equalize risk contribution
                target_risk = 1.0 / len(selected_tickers)
                weights = weights * (target_risk / risk_contrib)
                weights = weights / weights.sum()
                
        else:  # Default to equal weight
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
        
        # Apply constraints
        if sector_map is not None:
            weights = self.apply_sector_constraints(weights, sector_map)
            
        weights = self.apply_max_stock_constraint(weights)
        weights = self.apply_min_position_constraint(weights)
        
        # Fill missing stocks with zero weights
        all_weights = pd.Series(0.0, index=returns.columns)
        all_weights.loc[weights.index] = weights
        
        return all_weights
    
    def apply_volatility_targeting(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Tuple[pd.Series, float]:
        """
        Apply volatility targeting to scale portfolio weights.
        
        Args:
            weights: Series with portfolio weights
            returns: DataFrame with historical returns
            current_date: Current date for calculation
            
        Returns:
            Tuple of (scaled weights, cash weight)
        """
        # Calculate volatilities and correlation matrix
        volatilities = self.calculate_stock_volatility(returns, current_date)
        correlation_matrix = self.calculate_correlation_matrix(returns, current_date)
        
        # Calculate current portfolio volatility
        current_vol = self.calculate_portfolio_volatility(
            weights, volatilities, correlation_matrix
        )
        
        if current_vol > 0:
            # Calculate the adjustment factor
            vol_adjustment = self.target_volatility / current_vol
            
            # Limit the adjustment factor
            vol_adjustment = min(vol_adjustment, self.vol_max_adjustment)
            vol_adjustment = max(vol_adjustment, 1.0 / self.vol_max_adjustment)
            
            # Scale the weights
            scaled_weights = weights * vol_adjustment
            cash_weight = 1.0 - scaled_weights.sum()
            
            logger.info(
                f"Volatility targeting: current_vol={current_vol:.2%}, "
                f"target_vol={self.target_volatility:.2%}, "
                f"adjustment={vol_adjustment:.2f}, cash={cash_weight:.2%}"
            )
            
            return scaled_weights, cash_weight
        else:
            logger.warning("Portfolio volatility is zero. Unable to apply volatility targeting.")
            return weights, 0.0
    
    def apply_performance_based_derisking(
        self,
        weights: pd.Series,
        performance_metric: float,
        threshold: float = -0.10,  # -10% performance threshold
        reduction_factor: float = 0.5  # 50% risk reduction
    ) -> Tuple[pd.Series, float]:
        """
        Apply performance-based de-risking if performance falls below threshold.
        
        Args:
            weights: Series with portfolio weights
            performance_metric: Performance metric (e.g., drawdown)
            threshold: Threshold for de-risking
            reduction_factor: Factor to reduce risk by
            
        Returns:
            Tuple of (scaled weights, cash weight)
        """
        if performance_metric < threshold:
            # Apply de-risking
            derisk_factor = max(0, 1.0 + (performance_metric - threshold) / threshold * reduction_factor)
            derisk_factor = max(derisk_factor, 1.0 - reduction_factor)
            
            scaled_weights = weights * derisk_factor
            cash_weight = 1.0 - scaled_weights.sum()
            
            logger.info(
                f"Performance de-risking: metric={performance_metric:.2%}, "
                f"threshold={threshold:.2%}, "
                f"derisk_factor={derisk_factor:.2f}, cash={cash_weight:.2%}"
            )
            
            return scaled_weights, cash_weight
        else:
            return weights, 1.0 - weights.sum()
            
    def apply_all_constraints(
        self,
        signals: pd.Series,
        returns: pd.DataFrame,
        current_date: pd.Timestamp,
        market_caps: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        current_drawdown: Optional[float] = None
    ) -> Tuple[pd.Series, float]:
        """
        Apply all position constraints in sequence.
        
        Args:
            signals: Series with alpha signals for each stock
            returns: DataFrame with historical returns
            current_date: Current date for calculation
            market_caps: Optional series with market capitalizations
            sector_map: Optional dictionary mapping tickers to sectors
            current_drawdown: Optional current drawdown for de-risking
            
        Returns:
            Tuple of (final weights, cash weight)
        """
        # 1. Calculate initial position sizes
        weights = self.calculate_position_sizes(
            signals, returns, current_date, market_caps, sector_map
        )
        
        # 2. Apply volatility targeting
        weights, cash_weight = self.apply_volatility_targeting(
            weights, returns, current_date
        )
        
        # 3. Apply performance-based de-risking if drawdown is provided
        if current_drawdown is not None:
            weights, cash_weight = self.apply_performance_based_derisking(
                weights, current_drawdown
            )
        
        return weights, cash_weight 