"""
Strategy optimization script for the Quant Alpha system.
"""

import argparse
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.market_data import MarketDataFetcher
from utils.optimize import StrategyOptimizer, default_param_space
from config import RESULTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to optimize the trading strategy."""
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)', 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)', 
                        default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--n_calls', type=int, help='Number of optimization iterations', default=20)
    parser.add_argument('--n_jobs', type=int, help='Number of parallel jobs (-1 for all cores)', default=-1)
    parser.add_argument('--cv_folds', type=int, help='Number of cross-validation folds', default=3)
    parser.add_argument('--plot', action='store_true', help='Plot optimization results')
    parser.add_argument('--save_to_s3', action='store_true', help='Save results to S3')
    
    args = parser.parse_args()
    
    logger.info(f"Starting strategy optimization from {args.start_date} to {args.end_date}")
    
    # Create results directory
    results_dir = os.path.join(RESULTS_DIR, f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Fetch market data
        fetcher = MarketDataFetcher()
        data = fetcher.load_historical_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if data.empty:
            logger.error("No market data available. Exiting.")
            return 1
            
        logger.info(f"Fetched market data for {len(data.index.get_level_values('ticker').unique())} tickers")
        
        # Create parameter space
        param_space = default_param_space()
        
        # Create optimizer
        optimizer = StrategyOptimizer(
            data=data,
            params_space=param_space,
            n_calls=args.n_calls,
            n_jobs=args.n_jobs,
            cv_folds=args.cv_folds
        )
        
        # Run optimization
        results = optimizer.run_optimization()
        
        # Plot results if requested
        if args.plot:
            optimizer.plot_optimization_results()
        
        # Save best parameters to results directory
        best_params_file = os.path.join(results_dir, 'best_params.txt')
        with open(best_params_file, 'w') as f:
            f.write("Best Parameters:\n")
            for param, value in results['best_params'].items():
                f.write(f"{param}: {value}\n")
            
            f.write("\nBest Metrics:\n")
            for metric, value in results['best_metrics'].items():
                f.write(f"{metric}: {value:.4f}\n")
        
        logger.info(f"Optimization results saved to {results_dir}")
        
        # Save to S3 if requested
        if args.save_to_s3:
            try:
                from aws.utils import S3Handler
                
                s3 = S3Handler()
                s3_path = f"optimization/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                s3.save_json(results, s3_path)
                logger.info(f"Optimization results saved to S3: {s3_path}")
            except Exception as e:
                logger.warning(f"Failed to save results to S3: {str(e)}")
        
        # Run backtest with the best parameters
        logger.info("Running backtest with best parameters...")
        
        from factors.alpha_signals import AlphaSignalGenerator
        from backtest.engine import BacktestEngine
        
        # Extract best parameters
        best_params = results['best_params']
        
        # Generate signals with best parameters
        generator = AlphaSignalGenerator(
            data,
            rsi_window=int(best_params.get('rsi_window', 14)),
            sma_fast=int(best_params.get('sma_fast', 20)),
            sma_slow=int(best_params.get('sma_slow', 50)),
            volatility_window=int(best_params.get('volatility_window', 21)),
            momentum_window=int(best_params.get('momentum_window', 20))
        )
        
        # Create factor weights
        factor_weights = {
            'rsi': best_params.get('rsi_weight', 1.0),
            'sma': best_params.get('sma_weight', 1.0),
            'volatility': best_params.get('volatility_weight', 1.0),
            'momentum': best_params.get('momentum_weight', 1.0),
            'liquidity': best_params.get('liquidity_weight', 1.0),
            'gap': best_params.get('gap_weight', 1.0),
            'price_percentile': best_params.get('price_percentile_weight', 1.0),
            'roc': best_params.get('roc_weight', 1.0)
        }
        
        # Generate signals
        signals = generator.compute_all_signals(factor_weights=factor_weights)
        
        # Run backtest
        engine = BacktestEngine(
            data,
            signals,
            top_n=int(best_params.get('top_n', 5)),
            max_position_size=best_params.get('max_position_size', 0.2),
            rebalance_freq=f"{int(best_params.get('rebalance_freq', 5))}D",
            cash_buffer_pct=best_params.get('cash_buffer_pct', 0.1),
            stop_loss_pct=best_params.get('stop_loss_pct', 0.5),
            position_sizing_method=best_params.get('position_sizing_method', 'equal'),
            slippage=0.001  # Fixed slippage
        )
        
        backtest_results = engine.run_backtest()
        
        # Save backtest results
        backtest_path = os.path.join(results_dir, 'backtest_results')
        engine.save_results(backtest_path)
        
        # Plot backtest results
        try:
            fig = engine.plot_results()
            plt.savefig(os.path.join(results_dir, 'backtest_plot.png'))
            plt.close(fig)
            
            # Plot portfolio weights over time
            positions = backtest_results['holdings']
            
            plt.figure(figsize=(12, 8))
            positions_norm = positions.div(positions.sum(axis=1), axis=0)
            positions_norm.plot.area(figsize=(12, 8), cmap='viridis')
            plt.title('Portfolio Allocation Over Time')
            plt.xlabel('Date')
            plt.ylabel('Weight')
            plt.savefig(os.path.join(results_dir, 'portfolio_weights.png'))
            plt.close()
            
            # Plot factor importance
            plt.figure(figsize=(10, 6))
            factor_importance = pd.Series(factor_weights).abs()
            factor_importance = factor_importance[factor_importance > 0].sort_values(ascending=False)
            factor_importance.plot(kind='bar')
            plt.title('Factor Importance')
            plt.ylabel('Weight (Absolute Value)')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'factor_importance.png'))
            plt.close()
            
            logger.info(f"Plots saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
        
        logger.info("Strategy optimization completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in strategy optimization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 