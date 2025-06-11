"""
Main entry point for the Quant Alpha AWS system.
"""

import logging
import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt

from aws.utils import S3Handler
from backtest.engine import BacktestEngine
from data.market_data import MarketDataFetcher
from factors.alpha_signals import AlphaSignalGenerator
from config import (
    INITIAL_CAPITAL, COMMISSION, TOP_N_STOCKS, MAX_POSITION_SIZE,
    STOP_LOSS_PCT, CASH_BUFFER_PCT, WEEKLY_REBALANCE, RESULTS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the quant alpha system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quant Alpha AWS System')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)', 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)', 
                        default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--optimize', action='store_true', help='Run strategy optimization')
    parser.add_argument('--use_best_params', action='store_true', help='Use best parameters from previous optimization')
    parser.add_argument('--best_params_file', type=str, help='Path to best parameters file', 
                        default=None)
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save_to_s3', action='store_true', help='Save results to S3')
    
    args = parser.parse_args()
    
    logger.info("Starting Quant Alpha system")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set date range from arguments
    start_date = args.start_date
    end_date = args.end_date
    
    logger.info(f"Using date range: {start_date} to {end_date}")
    
    if args.optimize:
        # Run optimization
        logger.info("Running strategy optimization...")
        from optimize_strategy import main as optimize_main
        return optimize_main()
    
    # Run standard backtest
    try:
        # Fetch market data
        fetcher = MarketDataFetcher()
        data = fetcher.load_historical_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            logger.error("No market data available. Exiting.")
            return 1
            
        logger.info(f"Fetched market data for {len(data.index.get_level_values('ticker').unique())} tickers")
        
        # Save data to S3
        if args.save_to_s3:
            try:
                s3 = S3Handler()
                s3.save_dataframe(data, "market_data_latest.parquet")
                logger.info("Saved market data to S3")
            except Exception as e:
                logger.warning(f"Failed to save market data to S3: {str(e)}")
        
        # Load best parameters if requested
        factor_weights = None
        signal_params = {}
        backtest_params = {}
        
        if args.use_best_params and args.best_params_file:
            try:
                import json
                with open(args.best_params_file, 'r') as f:
                    best_params = json.load(f)
                
                # Extract signal parameters
                signal_params = {
                    'rsi_window': int(best_params.get('rsi_window', 14)),
                    'sma_fast': int(best_params.get('sma_fast', 20)),
                    'sma_slow': int(best_params.get('sma_slow', 50)),
                    'volatility_window': int(best_params.get('volatility_window', 21)),
                    'momentum_window': int(best_params.get('momentum_window', 20))
                }
                
                # Extract factor weights
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
                
                # Extract backtest parameters
                backtest_params = {
                    'top_n': int(best_params.get('top_n', 5)),
                    'max_position_size': best_params.get('max_position_size', 0.2),
                    'rebalance_freq': f"{int(best_params.get('rebalance_freq', 5))}D",
                    'cash_buffer_pct': best_params.get('cash_buffer_pct', 0.1),
                    'stop_loss_pct': best_params.get('stop_loss_pct', 0.5),
                    'position_sizing_method': best_params.get('position_sizing_method', 'equal'),
                    'slippage': 0.001  # Fixed slippage
                }
                
                logger.info(f"Loaded best parameters from {args.best_params_file}")
            except Exception as e:
                logger.warning(f"Failed to load best parameters: {str(e)}. Using default parameters.")
        
        # Generate alpha signals
        generator = AlphaSignalGenerator(data, **signal_params)
        signals = generator.compute_all_signals(factor_weights=factor_weights)
        
        if signals.empty:
            logger.error("Failed to generate alpha signals. Exiting.")
            return 1
            
        logger.info(f"Generated alpha signals with shape {signals.shape}")
        
        # Save signals to S3
        if args.save_to_s3:
            try:
                s3.save_dataframe(signals, "signals_latest.parquet")
                logger.info("Saved signals to S3")
            except Exception as e:
                logger.warning(f"Failed to save signals to S3: {str(e)}")
        
        # Run backtest
        logger.info("Running backtest...")
        engine = BacktestEngine(
            data,
            signals,
            initial_capital=INITIAL_CAPITAL,
            commission=COMMISSION,
            **backtest_params
        )
        results = engine.run_backtest()
        
        # Display metrics
        logger.info("\nBacktest Metrics:")
        for metric, value in results['metrics'].items():
            logger.info(f"{metric}: {value:.2f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(RESULTS_DIR, f"backtest_{timestamp}")
        engine.save_results(results_path)
        logger.info(f"Saved backtest results to {results_path}")
        
        # Save metrics to S3
        if args.save_to_s3:
            try:
                s3.save_json(results['metrics'], "metrics_latest.json")
                logger.info("Saved metrics to S3")
            except Exception as e:
                logger.warning(f"Failed to save metrics to S3: {str(e)}")
        
        # Plot results
        if args.plot:
            try:
                # Save main backtest plot
                fig = engine.plot_results()
                plot_path = os.path.join(RESULTS_DIR, f"backtest_plot_{timestamp}.png")
                plt.savefig(plot_path)
                plt.close(fig)
                logger.info(f"Saved plot to {plot_path}")
                
                # Plot portfolio weights over time
                positions = results['holdings']
                
                plt.figure(figsize=(12, 8))
                positions_norm = positions.div(positions.sum(axis=1), axis=0)
                positions_norm.plot.area(figsize=(12, 8), cmap='viridis')
                plt.title('Portfolio Allocation Over Time')
                plt.xlabel('Date')
                plt.ylabel('Weight')
                weights_path = os.path.join(RESULTS_DIR, f"portfolio_weights_{timestamp}.png")
                plt.savefig(weights_path)
                plt.close()
                logger.info(f"Saved portfolio weights plot to {weights_path}")
                
                # If using factor weights, plot factor importance
                if factor_weights:
                    plt.figure(figsize=(10, 6))
                    factor_importance = pd.Series(factor_weights).abs()
                    factor_importance = factor_importance[factor_importance > 0].sort_values(ascending=False)
                    factor_importance.plot(kind='bar')
                    plt.title('Factor Importance')
                    plt.ylabel('Weight (Absolute Value)')
                    plt.tight_layout()
                    factors_path = os.path.join(RESULTS_DIR, f"factor_importance_{timestamp}.png")
                    plt.savefig(factors_path)
                    plt.close()
                    logger.info(f"Saved factor importance plot to {factors_path}")
                
            except Exception as e:
                logger.error(f"Error plotting results: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 