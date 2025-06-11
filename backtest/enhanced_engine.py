"""
Enhanced backtesting engine.

This module integrates all enhanced features for backtesting:
1. Market regime detection
2. Signal combining with ML models
3. Position constraints and risk management
4. Performance monitoring and de-risking
"""

import logging
import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import DataLoader
from factors.alpha_signals import AlphaSignalGenerator
from utils.enhanced_strategy import EnhancedStrategy
from utils.market_regime import MarketRegimeDetector
from aws.utils import save_to_s3

# Ensure data directories exist
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedBacktestEngine:
    """
    Enhanced backtesting engine that integrates advanced features.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        signals_path: Optional[str] = None,
        benchmark_path: Optional[str] = None,
        output_path: str = 'results',
        config_path: Optional[str] = None,
        s3_bucket: Optional[str] = None
    ):
        """
        Initialize the EnhancedBacktestEngine.
        
        Args:
            data_path: Path to market data
            signals_path: Path to signal data
            benchmark_path: Path to benchmark data
            output_path: Path to save output
            config_path: Path to configuration file
            s3_bucket: S3 bucket name
        """
        self.data_path = data_path
        self.signals_path = signals_path
        self.benchmark_path = benchmark_path
        self.output_path = output_path
        self.config_path = config_path
        self.s3_bucket = s3_bucket
        
        self.data = None
        self.signals = None
        self.benchmark_data = None
        self.config = self._load_config()
        
        self.strategy = None
        self.results = None
        
        logger.info(f"EnhancedBacktestEngine initialized with output path: {output_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary with configuration
        """
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # Default configuration
        default_config = {
            'initial_capital': 1_000_000,
            'commission': 0.001,
            'rebalance_frequency': 5,  # Trading days
            'strategy': {
                'use_market_regime': True,
                'signal_combiner_method': 'ridge',
                'position_sizing_method': 'inverse_vol',
                'target_volatility': 0.15,
                'max_stock_weight': 0.20,
                'max_sector_weight': 0.40,
                'enable_performance_monitoring': True
            },
            'sector_map': {
                'AAPL': 'Technology',
                'MSFT': 'Technology',
                'GOOGL': 'Technology',
                'AMZN': 'Consumer Cyclical',
                'META': 'Technology',
                'TSLA': 'Consumer Cyclical',
                'JNJ': 'Healthcare',
                'PFE': 'Healthcare',
                'MRK': 'Healthcare',
                'ABBV': 'Healthcare',
                'LLY': 'Healthcare',
                'JPM': 'Financial',
                'BAC': 'Financial',
                'MS': 'Financial',
                'GS': 'Financial',
                'WFC': 'Financial',
                'XOM': 'Energy',
                'CVX': 'Energy',
                'COP': 'Energy',
                'EOG': 'Energy',
                'SLB': 'Energy'
            }
        }
        
        return default_config
    
    def load_data(self) -> None:
        """
        Load market data, signals, and benchmark data.
        """
        try:
            # Load market data using DataLoader
            data_loader = DataLoader(
                data_path=self.data_path,
                start_date=None,  # Will be adjusted in run_backtest
                end_date=None
            )
            self.data = data_loader.load_data()
            logger.info(f"Market data loaded with shape: {self.data.shape}")
            
            # Load signals
            if self.signals_path and os.path.exists(self.signals_path):
                if self.signals_path.endswith('.csv'):
                    self.signals = pd.read_csv(self.signals_path, parse_dates=['Date'])
                    if 'Date' in self.signals.columns:
                        self.signals.set_index('Date', inplace=True)
                    elif 'date' in self.signals.columns:
                        self.signals.set_index('date', inplace=True)
                elif self.signals_path.endswith('.parquet'):
                    self.signals = pd.read_parquet(self.signals_path)
                logger.info(f"Signal data loaded with shape: {self.signals.shape}")
            else:
                # Try to generate signals
                logger.info("Generating signals using AlphaSignalGenerator")
                signal_generator = AlphaSignalGenerator(self.data)
                self.signals = signal_generator.compute_all_signals()
                logger.info(f"Signals generated with shape: {self.signals.shape}")
            
            # Load benchmark data
            if self.benchmark_path and os.path.exists(self.benchmark_path):
                if self.benchmark_path.endswith('.csv'):
                    self.benchmark_data = pd.read_csv(self.benchmark_path, parse_dates=['Date'])
                    if 'Date' in self.benchmark_data.columns:
                        self.benchmark_data.set_index('Date', inplace=True)
                    elif 'date' in self.benchmark_data.columns:
                        self.benchmark_data.set_index('date', inplace=True)
                elif self.benchmark_path.endswith('.parquet'):
                    self.benchmark_data = pd.read_parquet(self.benchmark_path)
                logger.info(f"Benchmark data loaded with shape: {self.benchmark_data.shape}")
            else:
                # Use SPY from data if available
                if self.data is not None:
                    if isinstance(self.data, pd.DataFrame) and not isinstance(self.data.index, pd.MultiIndex):
                        if 'SPY' in self.data.columns:
                            self.benchmark_data = self.data[['SPY']]
                    elif isinstance(self.data.index, pd.MultiIndex):
                        # Try to extract SPY from multi-index data
                        if 'SPY' in self.data.index.get_level_values('ticker').unique():
                            self.benchmark_data = self.data.xs('SPY', level='ticker')
                            
                            # Reset the index to have a single-level date index
                            if isinstance(self.benchmark_data.index, pd.MultiIndex):
                                self.benchmark_data = self.benchmark_data.reset_index(level=0, drop=True)
                
                if self.benchmark_data is None:
                    logger.warning("No benchmark data found. Proceeding without benchmark.")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def initialize_strategy(self) -> None:
        """
        Initialize enhanced strategy.
        """
        strategy_config = self.config.get('strategy', {})
        sector_map = self.config.get('sector_map', {})
        
        self.strategy = EnhancedStrategy(
            use_market_regime=strategy_config.get('use_market_regime', True),
            signal_combiner_method=strategy_config.get('signal_combiner_method', 'ridge'),
            position_sizing_method=strategy_config.get('position_sizing_method', 'inverse_vol'),
            target_volatility=strategy_config.get('target_volatility', 0.15),
            max_stock_weight=strategy_config.get('max_stock_weight', 0.20),
            max_sector_weight=strategy_config.get('max_sector_weight', 0.40),
            enable_performance_monitoring=strategy_config.get('enable_performance_monitoring', True),
            model_path=os.path.join(self.output_path, 'models'),
            sector_map=sector_map
        )
        
        logger.info("Enhanced strategy initialized")
    
    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the enhanced backtest.
        
        Args:
            start_date: Optional start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Returns:
            Dictionary with backtest results
        """
        if self.data is None:
            self.load_data()
            
        if self.data is None:
            logger.error("No data loaded. Cannot run backtest.")
            return {}
        
        # Convert date strings to timestamps if provided
        start_ts = pd.Timestamp(start_date) if start_date else None
        end_ts = pd.Timestamp(end_date) if end_date else None
        
        # Initialize strategy if not done yet
        if self.strategy is None:
            self.initialize_strategy()
            
        # Try to load VIX data using DataLoader
        vix_data = None
        try:
            data_loader = DataLoader()
            vix_data = data_loader.load_vix_data()
            
            if vix_data is not None:
                logger.info(f"VIX data loaded with shape: {vix_data.shape}")
            else:
                logger.warning("No VIX data found. Continuing without volatility regime detection.")
                
        except Exception as e:
            logger.warning(f"Error loading VIX data: {str(e)}. Continuing without it.")
            
        # Run the backtest
        self.results = self.strategy.run_backtest(
            data=self.data,
            signals=self.signals,
            benchmark_data=self.benchmark_data,
            vix_data=vix_data,
            initial_capital=self.config.get('initial_capital', 1_000_000),
            commission=self.config.get('commission', 0.001),
            rebalance_freq=self.config.get('rebalance_frequency', 5),
            start_date=start_ts,
            end_date=end_ts,
            save_path=os.path.join(self.output_path, 'enhanced', 'latest')
        )
        
        # Save results to S3 if configured
        if self.s3_bucket:
            self._save_to_s3()
            
        return self.results
    
    def _save_to_s3(self) -> None:
        """
        Save results to S3.
        """
        if not self.s3_bucket or not self.results:
            return
        
        try:
            # Create a summary dataframe
            summary = pd.DataFrame({
                'date': self.results['portfolio_value'].index,
                'portfolio_value': self.results['portfolio_value'].values,
                'cash': self.results['cash'].values
            })
            
            # Add benchmark if available
            if self.benchmark_data is not None:
                benchmark_values = self.benchmark_data.loc[summary['date']].values
                summary['benchmark'] = benchmark_values
            
            # Save to S3
            s3_key = f"backtest_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}/summary.csv"
            save_to_s3(summary, self.s3_bucket, s3_key)
            
            # Save metrics
            metrics_df = pd.DataFrame([self.results['metrics']])
            s3_metrics_key = f"backtest_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}/metrics.csv"
            save_to_s3(metrics_df, self.s3_bucket, s3_metrics_key)
            
            logger.info(f"Results saved to S3 bucket: {self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            logger.warning(f"Error saving results to S3: {str(e)}")
    
    def plot_results(
        self,
        figsize: Tuple[int, int] = (16, 24),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot backtest results.
        
        Args:
            figsize: Figure size
            save_path: Path to save plot
        """
        if not self.results:
            logger.warning("No results to plot. Run backtest first.")
            return
        
        try:
            # Set plotting style
            sns.set(style="whitegrid")
            plt.figure(figsize=figsize)
            
            # Create subplots
            fig, axes = plt.subplots(6, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1, 2]})
            
            # Plot portfolio value
            portfolio_values = self.results['portfolio_value']
            axes[0].plot(portfolio_values.index, portfolio_values, label='Portfolio Value')
            
            # Add benchmark if available
            if self.benchmark_data is not None and isinstance(self.benchmark_data, pd.DataFrame):
                # Normalize benchmark to match initial portfolio value
                if 'SPY' in self.benchmark_data.columns:
                    benchmark = self.benchmark_data['SPY']
                else:
                    benchmark = self.benchmark_data.iloc[:, 0]
                
                benchmark = benchmark.loc[portfolio_values.index]
                benchmark = benchmark / benchmark.iloc[0] * portfolio_values.iloc[0]
                axes[0].plot(benchmark.index, benchmark, label='Benchmark', alpha=0.7)
            
            axes[0].set_title('Portfolio Performance')
            axes[0].set_ylabel('Value ($)')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot cash allocation
            cash_pct = (self.results['cash'] / portfolio_values) * 100
            axes[1].plot(cash_pct.index, cash_pct, label='Cash Allocation', color='orange')
            axes[1].set_title('Cash Allocation')
            axes[1].set_ylabel('Cash (%)')
            axes[1].grid(True)
            
            # Plot drawdown
            rolling_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values / rolling_max - 1) * 100
            axes[2].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            axes[2].plot(drawdown.index, drawdown, color='red', label='Drawdown')
            axes[2].set_title('Drawdown')
            axes[2].set_ylabel('Drawdown (%)')
            axes[2].grid(True)
            
            # Plot daily returns
            daily_returns = portfolio_values.pct_change() * 100
            axes[3].plot(daily_returns.index, daily_returns, label='Daily Returns', alpha=0.7)
            axes[3].set_title('Daily Returns')
            axes[3].set_ylabel('Return (%)')
            axes[3].grid(True)
            
            # Plot regime changes if available
            if 'regimes' in self.results and self.results['regimes']:
                regime_df = pd.DataFrame(self.results['regimes'])
                regime_df['date'] = pd.to_datetime(regime_df['date'])
                
                # Extract regime info
                regime_df['trend'] = regime_df['regime'].apply(lambda x: x['trend'] if isinstance(x, dict) else 'N/A')
                regime_df['volatility'] = regime_df['regime'].apply(lambda x: x['volatility'] if isinstance(x, dict) else 'N/A')
                regime_df['combined'] = regime_df['regime'].apply(lambda x: x['combined'] if isinstance(x, dict) else 'N/A')
                regime_df['use_mean_reversion'] = regime_df['regime'].apply(lambda x: x.get('use_mean_reversion', False) if isinstance(x, dict) else False)
                
                # Create color mappings
                trend_colors = {
                    'BULLISH': 'green',
                    'BEARISH': 'red',
                    'SIDEWAYS': 'gray',
                    'N/A': 'black'
                }
                
                vol_markers = {
                    'LOW': 'o',
                    'MEDIUM': 's',
                    'HIGH': '^',
                    'N/A': 'x'
                }
                
                # Plot regimes on a separate axis
                for trend in trend_colors.keys():
                    if trend in regime_df['trend'].values:
                        trend_data = regime_df[regime_df['trend'] == trend]
                        for vol in vol_markers.keys():
                            if vol in trend_data['volatility'].values:
                                vol_data = trend_data[trend_data['volatility'] == vol]
                                axes[4].scatter(
                                    vol_data['date'],
                                    [0] * len(vol_data),
                                    color=trend_colors[trend],
                                    marker=vol_markers[vol],
                                    s=100,
                                    label=f"{trend}_{vol}"
                                )
                
                axes[4].set_title('Market Regimes')
                axes[4].set_yticks([])
                axes[4].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                axes[4].grid(True)
                
                # Plot Mean Reversion vs Momentum strategy usage
                axes[5].plot(portfolio_values.index, portfolio_values, alpha=0.3, color='gray', label='Portfolio Value')
                
                # Plot vertical lines showing strategy shifts
                mean_reversion_points = regime_df[regime_df['use_mean_reversion'] == True]['date']
                momentum_points = regime_df[regime_df['use_mean_reversion'] == False]['date']
                
                for date in mean_reversion_points:
                    axes[5].axvline(x=date, color='red', linestyle='--', alpha=0.5)
                
                for date in momentum_points:
                    axes[5].axvline(x=date, color='green', linestyle='--', alpha=0.5)
                
                # Add markers for strategy type
                if len(mean_reversion_points) > 0:
                    axes[5].scatter(mean_reversion_points, [portfolio_values.max() * 0.9] * len(mean_reversion_points), 
                                  color='red', marker='o', s=80, label='Mean Reversion')
                
                if len(momentum_points) > 0:
                    axes[5].scatter(momentum_points, [portfolio_values.max() * 0.8] * len(momentum_points), 
                                  color='green', marker='o', s=80, label='Momentum')
                
                # Plot VIX and rolling std if available
                if 'vix' in regime_df.columns:
                    vix_values = regime_df['vix'].dropna()
                    if not vix_values.empty:
                        # Normalize VIX to fit the scale
                        max_vix = float(vix_values.max()) if not vix_values.empty else 1.0
                        max_portfolio = float(portfolio_values.max()) if not portfolio_values.empty else 1.0
                        vix_scaled = vix_values / max_vix * max_portfolio * 0.7
                        axes[5].plot(regime_df['date'], vix_scaled, color='purple', alpha=0.6, label='VIX (scaled)')
                        
                        # Mark high VIX points
                        high_vix = regime_df[regime_df['vix'] > self.strategy.vix_high_threshold]
                        if not high_vix.empty:
                            axes[5].scatter(high_vix['date'], vix_scaled.loc[high_vix.index], 
                                          color='purple', marker='^', s=100, label=f'VIX > {self.strategy.vix_high_threshold}')
                
                if 'rolling_std' in regime_df.columns:
                    rolling_std = regime_df['rolling_std'].dropna()
                    if not rolling_std.empty:
                        # Normalize rolling std to fit the scale
                        max_std = float(rolling_std.max()) if not rolling_std.empty else 1.0
                        max_portfolio = float(portfolio_values.max()) if not portfolio_values.empty else 1.0
                        std_scaled = rolling_std / max_std * max_portfolio * 0.5
                        axes[5].plot(regime_df['date'], std_scaled, color='blue', alpha=0.6, label='Volatility (scaled)')
                        
                        # Mark high volatility points
                        high_vol = regime_df[regime_df['rolling_std'] > self.strategy.rolling_std_high_threshold]
                        if not high_vol.empty:
                            axes[5].scatter(high_vol['date'], std_scaled.loc[high_vol.index], 
                                          color='blue', marker='^', s=100, label=f'Vol > {self.strategy.rolling_std_high_threshold:.2f}')
                
                axes[5].set_title('Strategy Shifts: Mean Reversion vs Momentum')
                axes[5].legend(loc='upper right')
                axes[5].grid(True)
                
            else:
                # Plot allocation changes instead
                if 'position_weights' in self.results and self.results['position_weights']:
                    # Create a dataframe with position counts over time
                    position_counts = []
                    for entry in self.results['position_weights']:
                        position_counts.append({
                            'date': entry['date'],
                            'count': len([w for w in entry['positions'].values() if w > 0])
                        })
                    
                    pos_df = pd.DataFrame(position_counts)
                    axes[4].plot(pos_df['date'], pos_df['count'], label='Number of Positions')
                    axes[4].set_title('Position Count')
                    axes[4].set_ylabel('Count')
                    axes[4].grid(True)
                    
                    # Remove the extra axis if regime data not available
                    fig.delaxes(axes[5])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot if path is provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def print_summary(self) -> None:
        """
        Print backtest summary.
        """
        if not self.results or 'metrics' not in self.results:
            logger.warning("No results to display. Run backtest first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*50)
        print(" "*15 + "BACKTEST SUMMARY")
        print("="*50)
        
        print(f"\nTotal Return: {metrics['total_return']:.2f}%")
        print(f"Annualized Return: {metrics['annual_return']:.2f}%")
        print(f"Annualized Volatility: {metrics['annual_volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Return/Max Drawdown: {metrics['return_over_max_drawdown']:.2f}")
        
        print("\nTRADING STATISTICS:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Total Commission: ${metrics['total_commission']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Profit: ${metrics['avg_profit']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        
        print("\nEXPOSURE:")
        print(f"Average Cash: {metrics['avg_cash_pct']:.2f}%")
        print(f"Average Market Exposure: {metrics['avg_exposure']:.2f}%")
        
        print("\nRECENT PERFORMANCE:")
        print(f"30-Day Sharpe Ratio: {metrics['final_30d_sharpe']:.2f}")
        
        # Print regime statistics if available
        if 'regimes' in self.results and self.results['regimes']:
            regime_df = pd.DataFrame(self.results['regimes'])
            regime_df['trend'] = regime_df['regime'].apply(lambda x: x['trend'] if isinstance(x, dict) else 'N/A')
            regime_df['volatility'] = regime_df['regime'].apply(lambda x: x['volatility'] if isinstance(x, dict) else 'N/A')
            
            print("\nREGIME STATISTICS:")
            trend_counts = regime_df['trend'].value_counts()
            vol_counts = regime_df['volatility'].value_counts()
            
            print("Market Trends:")
            for trend, count in trend_counts.items():
                print(f"  {trend}: {count} days ({count/len(regime_df)*100:.1f}%)")
            
            print("Volatility Regimes:")
            for vol, count in vol_counts.items():
                print(f"  {vol}: {count} days ({count/len(regime_df)*100:.1f}%)")
        
        print("="*50)


def main():
    """Main function for running enhanced backtest."""
    parser = argparse.ArgumentParser(description='Run enhanced backtesting engine')
    parser.add_argument('--data', type=str, help='Path to market data CSV')
    parser.add_argument('--signals', type=str, help='Path to signals data CSV')
    parser.add_argument('--benchmark', type=str, help='Path to benchmark data CSV')
    parser.add_argument('--config', type=str, help='Path to configuration JSON')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket for storing results')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create engine
    engine = EnhancedBacktestEngine(
        data_path=args.data,
        signals_path=args.signals,
        benchmark_path=args.benchmark,
        output_path=args.output,
        config_path=args.config,
        s3_bucket=args.s3_bucket
    )
    
    # Run backtest
    results = engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Print summary
    engine.print_summary()
    
    # Generate plots if requested
    if args.plot:
        plot_path = os.path.join(args.output, 'backtest_plots')
        os.makedirs(plot_path, exist_ok=True)
        engine.plot_results(save_path=os.path.join(plot_path, 'performance.png'))


if __name__ == '__main__':
    main() 