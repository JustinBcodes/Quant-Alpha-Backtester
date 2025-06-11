"""
Signal combiner module.

This module implements dynamic signal weighting using:
1. Machine learning models (LightGBM, Ridge Regression)
2. Factor rotation based on trailing performance
3. Regime-based weighting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import pickle
import os
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Import lightgbm with error handling
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Using Ridge Regression only.")

# Configure logging
logger = logging.getLogger(__name__)

class CombinerMethod(Enum):
    """Signal combiner method enumeration."""
    EQUAL_WEIGHT = 0
    TRAILING_PERFORMANCE = 1
    RIDGE_REGRESSION = 2
    LIGHTGBM = 3
    REGIME_BASED = 4

class SignalCombiner:
    """
    Combines alpha signals using various methods including ML models.
    """
    
    def __init__(
        self,
        method: Union[str, CombinerMethod] = 'ridge',
        lookback_window: int = 126,
        refit_frequency: int = 21,
        performance_metric: str = 'information_ratio',
        min_history: int = 63,
        regularization: float = 1.0,
        model_path: Optional[str] = None
    ):
        """
        Initialize the SignalCombiner.
        
        Args:
            method: Combiner method ('equal', 'trailing', 'ridge', 'lightgbm', 'regime')
            lookback_window: Lookback window for training ML models
            refit_frequency: How often to refit the model (in trading days)
            performance_metric: Metric for trailing performance ('ir', 'sharpe', 'returns')
            min_history: Minimum history required for training
            regularization: Regularization parameter for Ridge regression
            model_path: Path to save/load models
        """
        # Convert string method to enum
        if isinstance(method, str):
            method = method.lower()
            if method == 'equal':
                self.method = CombinerMethod.EQUAL_WEIGHT
            elif method == 'trailing':
                self.method = CombinerMethod.TRAILING_PERFORMANCE
            elif method == 'ridge':
                self.method = CombinerMethod.RIDGE_REGRESSION
            elif method == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    logger.warning("LightGBM not available. Using Ridge Regression instead.")
                    self.method = CombinerMethod.RIDGE_REGRESSION
                else:
                    self.method = CombinerMethod.LIGHTGBM
            elif method == 'regime':
                self.method = CombinerMethod.REGIME_BASED
            else:
                logger.warning(f"Unknown method: {method}. Using Ridge Regression.")
                self.method = CombinerMethod.RIDGE_REGRESSION
        else:
            self.method = method
        
        self.lookback_window = lookback_window
        self.refit_frequency = refit_frequency
        self.performance_metric = performance_metric
        self.min_history = min_history
        self.regularization = regularization
        self.model_path = model_path
        
        # Initialize models
        self.ridge_model = None
        self.lgb_model = None
        self.feature_scaler = StandardScaler()
        
        # Cache for factor performance and weights
        self.factor_performance = {}
        self.current_weights = {}
        self.last_refit_date = None
        
        logger.info(
            f"SignalCombiner initialized with method={self.method.name}, "
            f"lookback_window={lookback_window}, refit_frequency={refit_frequency}"
        )
    
    def calculate_factor_performance(
        self, 
        factor_data: pd.DataFrame, 
        returns: pd.DataFrame,
        window: int = 63
    ) -> pd.DataFrame:
        """
        Calculate performance metrics for each factor.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            window: Rolling window for performance calculation
        
        Returns:
            DataFrame with performance metrics for each factor
        """
        performance = pd.DataFrame(index=factor_data.index)
        
        # For each factor, calculate performance metrics
        for factor in factor_data.columns:
            # Rank IC: correlation between factor and returns
            ic = factor_data[factor].rolling(window).corr(returns)
            
            # Information ratio: mean IC / std IC
            ir = (
                factor_data[factor].rolling(window).corr(returns).mean() / 
                factor_data[factor].rolling(window).corr(returns).std()
            )
            
            # Add to performance DataFrame
            performance[f"{factor}_ic"] = ic
            performance[f"{factor}_ir"] = ir
        
        return performance
    
    def get_trailing_weights(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        current_date: pd.Timestamp,
        window: int = 63
    ) -> Dict[str, float]:
        """
        Calculate weights based on trailing factor performance.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            current_date: Current date for calculation
            window: Window for performance calculation
        
        Returns:
            Dictionary with factor weights
        """
        # Calculate factor performance
        performance = self.calculate_factor_performance(
            factor_data.loc[:current_date],
            returns.loc[:current_date],
            window=window
        )
        
        # Get the most recent performance metrics
        latest_perf = performance.iloc[-1]
        
        # Extract information ratios
        factor_ir = {}
        for factor in factor_data.columns:
            metric_key = f"{factor}_ir" if self.performance_metric == 'information_ratio' else f"{factor}_ic"
            if metric_key in latest_perf:
                factor_ir[factor] = latest_perf[metric_key]
        
        # Calculate weights
        weights = {}
        total_ir = sum(max(0.01, ir) for ir in factor_ir.values())
        
        for factor, ir in factor_ir.items():
            # Ensure positive weight and normalize
            weights[factor] = max(0.01, ir) / total_ir
        
        # Cache the performance and weights
        self.factor_performance[current_date] = factor_ir
        self.current_weights = weights
        
        return weights
    
    def _prepare_ml_data(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        end_date: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ML model training.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            end_date: End date for data preparation
        
        Returns:
            Tuple of (X, y) arrays for ML training
        """
        # Get data up to end_date
        cutoff_idx = factor_data.index.get_loc(end_date, method='ffill')
        start_idx = max(0, cutoff_idx - self.lookback_window)
        
        # Extract training data
        X = factor_data.iloc[start_idx:cutoff_idx+1].values
        y = returns.iloc[start_idx:cutoff_idx+1].values
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        return X, y
    
    def train_ridge_model(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Ridge:
        """
        Train a Ridge regression model for signal weighting.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            current_date: Current date for training
        
        Returns:
            Trained Ridge model
        """
        # Prepare training data
        X, y = self._prepare_ml_data(factor_data, returns, current_date)
        
        if len(X) < self.min_history:
            logger.warning(f"Not enough history to train model. Using equal weights.")
            return None
        
        # Train Ridge model
        model = Ridge(alpha=self.regularization)
        model.fit(X, y)
        
        # Save model if path is provided
        if self.model_path:
            os.makedirs(self.model_path, exist_ok=True)
            model_file = os.path.join(
                self.model_path, 
                f"ridge_model_{current_date.strftime('%Y%m%d')}.pkl"
            )
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        return model
    
    def train_lightgbm_model(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[Any]:
        """
        Train a LightGBM model for signal weighting.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            current_date: Current date for training
        
        Returns:
            Trained LightGBM model
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available. Cannot train model.")
            return None
        
        # Prepare training data
        X, y = self._prepare_ml_data(factor_data, returns, current_date)
        
        if len(X) < self.min_history:
            logger.warning(f"Not enough history to train model. Using equal weights.")
            return None
        
        # Split data for validation
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx = list(tscv.split(X))[-1]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        # Parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 4,
            'num_leaves': 15,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 5,
            'verbosity': -1
        }
        
        # Train model
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=100,
            valid_sets=[dtrain, dval],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Save model if path is provided
        if self.model_path:
            os.makedirs(self.model_path, exist_ok=True)
            model_file = os.path.join(
                self.model_path, 
                f"lgb_model_{current_date.strftime('%Y%m%d')}.txt"
            )
            model.save_model(model_file)
        
        return model
    
    def get_model_weights(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Get factor weights from a trained ML model.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            current_date: Current date for prediction
        
        Returns:
            Dictionary with factor weights
        """
        # Check if we need to refit the model
        need_refit = (
            self.last_refit_date is None or
            (current_date - self.last_refit_date).days >= self.refit_frequency
        )
        
        if need_refit:
            if self.method == CombinerMethod.RIDGE_REGRESSION:
                self.ridge_model = self.train_ridge_model(factor_data, returns, current_date)
                self.last_refit_date = current_date
            elif self.method == CombinerMethod.LIGHTGBM and LIGHTGBM_AVAILABLE:
                self.lgb_model = self.train_lightgbm_model(factor_data, returns, current_date)
                self.last_refit_date = current_date
        
        # Get weights from model
        weights = {}
        
        if self.method == CombinerMethod.RIDGE_REGRESSION and self.ridge_model is not None:
            # Scale current factors
            current_factors = factor_data.loc[current_date].values.reshape(1, -1)
            scaled_factors = self.feature_scaler.transform(current_factors)
            
            # Get coefficients
            coefs = self.ridge_model.coef_
            
            # Normalize coefficients to get weights (ensure positive)
            coefs = np.abs(coefs)
            total_weight = np.sum(coefs)
            
            if total_weight > 0:
                normalized_coefs = coefs / total_weight
                
                # Assign weights to factors
                for i, factor in enumerate(factor_data.columns):
                    weights[factor] = normalized_coefs[i]
            else:
                # Equal weights if all coefficients are zero
                for factor in factor_data.columns:
                    weights[factor] = 1.0 / len(factor_data.columns)
                    
        elif self.method == CombinerMethod.LIGHTGBM and self.lgb_model is not None and LIGHTGBM_AVAILABLE:
            # Get feature importances
            importances = self.lgb_model.feature_importance(importance_type='gain')
            
            # Normalize importances
            total_importance = np.sum(importances)
            
            if total_importance > 0:
                for i, factor in enumerate(factor_data.columns):
                    weights[factor] = importances[i] / total_importance
            else:
                # Equal weights if all importances are zero
                for factor in factor_data.columns:
                    weights[factor] = 1.0 / len(factor_data.columns)
        else:
            # Default to equal weights if model not available
            for factor in factor_data.columns:
                weights[factor] = 1.0 / len(factor_data.columns)
        
        # Cache the weights
        self.current_weights = weights
        
        return weights
    
    def combine_signals(
        self,
        factor_data: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        regime_weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Combine factor signals using the specified method.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns (required for ML methods)
            current_date: Current date for combination (defaults to latest date)
            regime_weights: Optional regime-based weights to use
        
        Returns:
            Series with combined signals
        """
        if current_date is None:
            current_date = factor_data.index[-1]
        
        # Choose weights based on method
        if self.method == CombinerMethod.EQUAL_WEIGHT:
            weights = {factor: 1.0 / len(factor_data.columns) for factor in factor_data.columns}
        
        elif self.method == CombinerMethod.TRAILING_PERFORMANCE:
            if returns is None:
                logger.warning("Returns data required for trailing performance. Using equal weights.")
                weights = {factor: 1.0 / len(factor_data.columns) for factor in factor_data.columns}
            else:
                weights = self.get_trailing_weights(factor_data, returns, current_date)
        
        elif self.method in [CombinerMethod.RIDGE_REGRESSION, CombinerMethod.LIGHTGBM]:
            if returns is None:
                logger.warning("Returns data required for ML models. Using equal weights.")
                weights = {factor: 1.0 / len(factor_data.columns) for factor in factor_data.columns}
            else:
                weights = self.get_model_weights(factor_data, returns, current_date)
        
        elif self.method == CombinerMethod.REGIME_BASED:
            if regime_weights is None:
                logger.warning("Regime weights not provided. Using equal weights.")
                weights = {factor: 1.0 / len(factor_data.columns) for factor in factor_data.columns}
            else:
                weights = regime_weights
                # Fill missing factors with small weights
                for factor in factor_data.columns:
                    if factor not in weights:
                        weights[factor] = 0.01
                # Renormalize
                total_weight = sum(weights.values())
                weights = {factor: weight / total_weight for factor, weight in weights.items()}
        
        # Combine signals
        combined_signal = pd.Series(0.0, index=factor_data.index)
        
        for factor, weight in weights.items():
            if factor in factor_data.columns:
                combined_signal += factor_data[factor] * weight
        
        return combined_signal

    def calculate_monthly_factor_rotation(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        current_date: pd.Timestamp,
        window: int = 21,
        top_n: int = 3
    ) -> Dict[str, float]:
        """
        Calculate weights for monthly factor rotation based on trailing info ratio.
        
        Args:
            factor_data: DataFrame with factor signals
            returns: DataFrame with forward returns
            current_date: Current date for rotation
            window: Window for performance calculation (default: 21 days = ~1 month)
            top_n: Number of top factors to select
            
        Returns:
            Dictionary with rotated factor weights
        """
        # Calculate factor performance
        performance = self.calculate_factor_performance(
            factor_data.loc[:current_date],
            returns.loc[:current_date],
            window=window
        )
        
        # Get the most recent performance metrics
        latest_perf = performance.iloc[-1]
        
        # Extract information ratios
        factor_ir = {}
        for factor in factor_data.columns:
            metric_key = f"{factor}_ir"
            if metric_key in latest_perf:
                factor_ir[factor] = latest_perf[metric_key]
        
        # Get top N factors
        top_factors = sorted(
            factor_ir.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Calculate weights (equal weight among top factors)
        weights = {}
        for factor, _ in top_factors:
            weights[factor] = 1.0 / top_n
        
        # Set zero weight for all other factors
        for factor in factor_data.columns:
            if factor not in weights:
                weights[factor] = 0.0
        
        # Cache the weights
        self.current_weights = weights
        
        return weights 