#!/usr/bin/env python
"""
Enhanced Backtest Runner

This script runs the enhanced backtesting engine with all the advanced features:
1. Market regime detection
2. Signal combining with ML models
3. Position constraints and risk management
4. Performance monitoring and de-risking

Usage:
    python run_enhanced_backtest.py [options]

Options:
    --data PATH            Path to market data CSV
    --signals PATH         Path to signals data CSV
    --benchmark PATH       Path to benchmark data CSV
    --vix PATH             Path to VIX data CSV
    --config PATH          Path to configuration JSON
    --output PATH          Output directory
    --s3-bucket NAME       S3 bucket for storing results
    --start-date DATE      Start date for backtest (YYYY-MM-DD)
    --end-date DATE        End date for backtest (YYYY-MM-DD)
    --plot                 Generate performance plots
"""

import os
import sys
import logging
from datetime import datetime

from backtest.enhanced_engine import EnhancedBacktestEngine
from config import VIX_HIGH_THRESHOLD, ROLLING_STD_HIGH_THRESHOLD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run enhanced backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced backtesting engine')
    parser.add_argument('--data', type=str, help='Path to market data CSV')
    parser.add_argument('--signals', type=str, help='Path to signals data CSV')
    parser.add_argument('--benchmark', type=str, help='Path to benchmark data CSV')
    parser.add_argument('--vix', type=str, help='Path to VIX data CSV')
    parser.add_argument('--config', type=str, help='Path to configuration JSON')
    parser.add_argument('--output', type=str, default='results/enhanced', help='Output directory')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket for storing results')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    parser.add_argument('--vix-threshold', type=float, default=VIX_HIGH_THRESHOLD, 
                        help=f'VIX threshold for regime detection (default: {VIX_HIGH_THRESHOLD})')
    parser.add_argument('--vol-threshold', type=float, default=ROLLING_STD_HIGH_THRESHOLD, 
                        help=f'Volatility threshold for regime detection (default: {ROLLING_STD_HIGH_THRESHOLD:.3f})')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Print configuration
    logger.info("="*50)
    logger.info("ENHANCED BACKTEST RUNNER")
    logger.info("="*50)
    logger.info(f"Data path: {args.data}")
    logger.info(f"Signals path: {args.signals}")
    logger.info(f"Benchmark path: {args.benchmark}")
    logger.info(f"VIX data path: {args.vix}")
    logger.info(f"Config path: {args.config}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"S3 bucket: {args.s3_bucket}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Generate plots: {args.plot}")
    logger.info(f"Regime detection thresholds: VIX > {args.vix_threshold}, Vol > {args.vol_threshold}")
    logger.info("="*50)
    
    try:
        # Create output subdirectory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save VIX data to the standard location if provided
        if args.vix and os.path.exists(args.vix):
            import shutil
            from pathlib import Path
            
            # Make sure data/raw directory exists
            raw_dir = os.path.join('data', 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            
            # Copy VIX data
            vix_dest = os.path.join(raw_dir, 'vix.csv')
            shutil.copy(args.vix, vix_dest)
            logger.info(f"Copied VIX data to {vix_dest}")
        
        # Create engine
        engine = EnhancedBacktestEngine(
            data_path=args.data,
            signals_path=args.signals,
            benchmark_path=args.benchmark,
            output_path=output_dir,
            config_path=args.config,
            s3_bucket=args.s3_bucket
        )
        
        # Set custom thresholds if provided
        if hasattr(engine.strategy, 'vix_high_threshold'):
            engine.strategy.vix_high_threshold = args.vix_threshold
            
        if hasattr(engine.strategy, 'rolling_std_high_threshold'):
            engine.strategy.rolling_std_high_threshold = args.vol_threshold
            
        # Update the regime detector if it exists
        if hasattr(engine.strategy, 'regime_detector') and engine.strategy.regime_detector is not None:
            engine.strategy.regime_detector.vix_high_threshold = args.vix_threshold
            engine.strategy.regime_detector.rolling_std_high_threshold = args.vol_threshold
            
        logger.info(f"Using regime thresholds: VIX > {args.vix_threshold}, Vol > {args.vol_threshold}")
        
        # Run backtest
        logger.info("Starting backtest run...")
        results = engine.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print summary
        engine.print_summary()
        
        # Generate plots if requested
        if args.plot:
            plot_path = os.path.join(output_dir, 'plots')
            os.makedirs(plot_path, exist_ok=True)
            
            logger.info("Generating performance plots...")
            engine.plot_results(save_path=os.path.join(plot_path, 'performance.png'))
        
        logger.info(f"Backtest completed successfully. Results saved to {output_dir}")
        logger.info("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return 1

if __name__ == '__main__':
    sys.exit(main()) 