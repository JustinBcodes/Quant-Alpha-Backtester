"""
Optimization module for hyperparameter tuning of the alpha strategy.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable, Optional

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from joblib import Parallel, delayed
from tqdm import tqdm

from backtest.engine import BacktestEngine
from factors.alpha_signals import AlphaSignalGenerator
from config import RESULTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Optimizer for alpha strategy using Bayesian optimization.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        params_space: Dict[str, Any],
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        slippage: float = 0.001,
        n_calls: int = 20,
        n_jobs: int = -1,
        random_state: int = 42,
        cv_folds: int = 3
    ):
        """
        Initialize the StrategyOptimizer.
        
        Args:
            data: DataFrame with market data (multi-index with ticker and date)
            params_space: Dictionary with parameter space for optimization
            initial_capital: Initial capital for backtest
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            n_calls: Number of optimization iterations
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds for time series
        """
        self.data = data.copy()
        self.params_space = params_space
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.n_calls = n_calls
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Results tracking
        self.best_params = None
        self.best_metrics = None
        self.optimization_results = None
        self.all_trials = []
        
        # Create directory for optimization results
        self.optimization_dir = os.path.join(RESULTS_DIR, f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.optimization_dir, exist_ok=True)
        
        logger.info(f"StrategyOptimizer initialized with {len(params_space)} parameters to optimize")
    
    def _prepare_skopt_space(self) -> List[Any]:
        """
        Convert params_space dict to skopt dimensions.
        
        Returns:
            List of skopt dimension objects
        """
        dimensions = []
        
        for param_name, param_config in self.params_space.items():
            param_type = param_config['type']
            
            if param_type == 'real':
                dimensions.append(Real(
                    param_config['low'], 
                    param_config['high'], 
                    name=param_name, 
                    prior=param_config.get('prior', 'uniform')
                ))
            elif param_type == 'integer':
                dimensions.append(Integer(
                    param_config['low'], 
                    param_config['high'], 
                    name=param_name
                ))
            elif param_type == 'categorical':
                dimensions.append(Categorical(
                    param_config['categories'], 
                    name=param_name
                ))
        
        return dimensions
    
    def _time_series_cv(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate time series cross-validation folds.
        
        Returns:
            List of (train_idx, test_idx) tuples
        """
        # Get all dates from the data
        all_dates = self.data.index.get_level_values('Date').unique().sort_values()
        
        # Calculate fold size
        fold_size = len(all_dates) // (self.cv_folds + 1)
        
        # Generate folds
        folds = []
        for i in range(self.cv_folds):
            # Each fold moves forward in time
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            
            # Adjust for last fold
            if i == self.cv_folds - 1:
                test_end = len(all_dates)
            
            train_idx = all_dates[:train_end]
            test_idx = all_dates[test_start:test_end]
            
            folds.append((train_idx, test_idx))
        
        return folds
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a set of parameters using time series cross-validation.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Dictionary with averaged metrics across CV folds
        """
        folds = self._time_series_cv()
        fold_metrics = []
        
        # Run backtest for each fold
        for train_idx, test_idx in folds:
            # Split data into train and test
            train_data = self.data[self.data.index.get_level_values('Date').isin(train_idx)]
            test_data = self.data[self.data.index.get_level_values('Date').isin(test_idx)]
            
            # Generate signals on training data
            generator = AlphaSignalGenerator(
                train_data,
                rsi_window=params.get('rsi_window', 14),
                sma_fast=params.get('sma_fast', 20),
                sma_slow=params.get('sma_slow', 50),
                volatility_window=params.get('volatility_window', 21),
                momentum_window=params.get('momentum_window', 20)
            )
            
            # Apply any custom factor weightings
            factor_weights = {
                'rsi': params.get('rsi_weight', 1.0),
                'sma': params.get('sma_weight', 1.0),
                'volatility': params.get('volatility_weight', 1.0),
                'momentum': params.get('momentum_weight', 1.0),
                'liquidity': params.get('liquidity_weight', 1.0),
                'gap': params.get('gap_weight', 1.0)
            }
            
            # Compute signals with custom parameters
            signals = generator.compute_all_signals(factor_weights=factor_weights)
            
            # Initialize backtest engine with test data
            engine = BacktestEngine(
                test_data,
                signals,
                initial_capital=self.initial_capital,
                commission=self.commission,
                rebalance_freq=f"{params.get('rebalance_freq', 5)}D",
                top_n=params.get('top_n', 5),
                max_position_size=params.get('max_position_size', 0.2),
                stop_loss_pct=params.get('stop_loss_pct', 0.5),
                cash_buffer_pct=params.get('cash_buffer_pct', 0.1),
                slippage=self.slippage
            )
            
            # Run backtest on test data
            results = engine.run_backtest()
            
            # Store metrics for this fold
            fold_metrics.append(results['metrics'])
        
        # Average metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
        
        # Add a combined metric (sharpe + return) for optimization
        avg_metrics['combined_score'] = avg_metrics['sharpe_ratio'] + avg_metrics['total_return'] / 100
        
        return avg_metrics
    
    def _objective_function(self, **params) -> float:
        """
        Objective function for optimization.
        
        Args:
            params: Parameter values
            
        Returns:
            Negative combined score (to minimize)
        """
        # Log trial parameters
        logger.info(f"Evaluating parameters: {params}")
        
        # Evaluate parameters
        metrics = self._evaluate_params(params)
        
        # Save trial results
        trial_results = {
            'params': params,
            'metrics': metrics
        }
        self.all_trials.append(trial_results)
        
        # Log metrics
        logger.info(f"Trial metrics: Sharpe={metrics['sharpe_ratio']:.2f}, Return={metrics['total_return']:.2f}%")
        
        # Return negative combined score (for minimization)
        return -metrics['combined_score']
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting optimization with {self.n_calls} iterations")
        
        # Prepare parameter space
        dimensions = self._prepare_skopt_space()
        
        # Create the objective function with named arguments
        @use_named_args(dimensions)
        def objective(**params):
            return self._objective_function(**params)
        
        # Run Bayesian optimization
        # gp_minimize requires n_calls >= 10, adjust if needed
        n_calls = max(10, self.n_calls)
        
        # For very small n_calls, use a simpler optimizer
        if self.n_calls < 10:
            from skopt import dummy_minimize
            opt_result = dummy_minimize(
                objective,
                dimensions,
                n_calls=self.n_calls,
                random_state=self.random_state,
                verbose=True
            )
        else:
            opt_result = gp_minimize(
                objective,
                dimensions,
                n_calls=n_calls,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=True
            )
        
        # Extract best parameters
        self.best_params = {dim.name: value for dim, value in zip(dimensions, opt_result.x)}
        
        # Evaluate best parameters
        self.best_metrics = self._evaluate_params(self.best_params)
        
        # Save optimization results
        self.optimization_results = {
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'all_trials': self.all_trials
        }
        
        # Save to file
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            else:
                return obj
                
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = convert_numpy_types(self.optimization_results)
        
        with open(os.path.join(self.optimization_dir, 'optimization_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Optimization completed. Best parameters: {self.best_params}")
        logger.info(f"Best metrics: Sharpe={self.best_metrics['sharpe_ratio']:.2f}, Return={self.best_metrics['total_return']:.2f}%")
        
        return self.optimization_results
    
    def plot_optimization_results(self) -> None:
        """
        Plot optimization results.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figures directory
            figures_dir = os.path.join(self.optimization_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            # Extract trial data
            trials_df = pd.DataFrame([
                {
                    **trial['params'],
                    'sharpe_ratio': trial['metrics']['sharpe_ratio'],
                    'total_return': trial['metrics']['total_return'],
                    'max_drawdown': trial['metrics']['max_drawdown'],
                    'volatility': trial['metrics']['volatility'],
                    'combined_score': trial['metrics']['combined_score']
                }
                for trial in self.all_trials
            ])
            
            # Plot parameter importance
            plt.figure(figsize=(12, 8))
            
            param_cols = [col for col in trials_df.columns 
                          if col not in ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility', 'combined_score']]
            
            corr_matrix = trials_df[param_cols + ['combined_score']].corr()
            param_importance = corr_matrix['combined_score'].drop('combined_score').abs().sort_values(ascending=False)
            
            sns.barplot(x=param_importance.values, y=param_importance.index)
            plt.title('Parameter Importance (Correlation with Combined Score)')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'parameter_importance.png'))
            
            # Plot top parameters vs metrics
            top_params = param_importance.index[:4]
            
            # Create pair plots for top parameters and metrics
            pair_cols = list(top_params) + ['sharpe_ratio', 'total_return', 'max_drawdown']
            pair_df = trials_df[pair_cols]
            
            sns.pairplot(pair_df)
            plt.savefig(os.path.join(figures_dir, 'parameter_pair_plot.png'))
            
            # Plot optimization convergence
            plt.figure(figsize=(10, 6))
            plt.plot(trials_df['combined_score'], marker='o')
            plt.axhline(trials_df['combined_score'].max(), color='r', linestyle='--')
            plt.title('Optimization Convergence')
            plt.xlabel('Trial')
            plt.ylabel('Combined Score')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'optimization_convergence.png'))
            
            logger.info(f"Optimization plots saved to {figures_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting optimization results: {str(e)}")

def default_param_space() -> Dict[str, Dict[str, Any]]:
    """
    Return the default parameter space for optimization.
    
    Returns:
        Dictionary with parameter space definition
    """
    return {
        # Signal generation parameters
        'rsi_window': {'type': 'integer', 'low': 7, 'high': 21},
        'rsi_weight': {'type': 'real', 'low': 0.0, 'high': 2.0},
        'sma_fast': {'type': 'integer', 'low': 5, 'high': 30},
        'sma_slow': {'type': 'integer', 'low': 30, 'high': 100},
        'sma_weight': {'type': 'real', 'low': 0.0, 'high': 2.0},
        'volatility_window': {'type': 'integer', 'low': 10, 'high': 30},
        'volatility_weight': {'type': 'real', 'low': -2.0, 'high': 0.0},  # Negative: prefer lower volatility
        'momentum_window': {'type': 'integer', 'low': 5, 'high': 40},
        'momentum_weight': {'type': 'real', 'low': 0.0, 'high': 2.0},
        'liquidity_weight': {'type': 'real', 'low': 0.0, 'high': 1.0},
        'gap_weight': {'type': 'real', 'low': 0.0, 'high': 1.0},
        
        # Portfolio construction parameters
        'top_n': {'type': 'integer', 'low': 3, 'high': 10},
        'max_position_size': {'type': 'real', 'low': 0.05, 'high': 0.25},
        'rebalance_freq': {'type': 'integer', 'low': 1, 'high': 10},
        'cash_buffer_pct': {'type': 'real', 'low': 0.05, 'high': 0.20},
        'stop_loss_pct': {'type': 'real', 'low': 0.1, 'high': 0.5}
    } 