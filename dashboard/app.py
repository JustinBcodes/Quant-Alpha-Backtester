"""
Dashboard application for the Quant Alpha backtesting system.
"""

import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from aws.utils import S3Handler
from data.market_data import MarketDataFetcher
from factors.alpha_signals import AlphaSignalGenerator
from backtest.engine import BacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize AWS handler
s3 = S3Handler()

def plot_signals(signals: pd.DataFrame) -> go.Figure:
    """
    Create plotly figure for signals.
    
    Args:
        signals: DataFrame with signal data
        
    Returns:
        plotly Figure object
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Signal Scores', 'Signal Distribution'),
            vertical_spacing=0.2
        )
        
        # Ensure signals is not empty
        if signals.empty:
            fig.add_annotation(
                text="No signal data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Check if 'final_score' exists
        if 'final_score' not in signals.columns:
            fig.add_annotation(
                text="No 'final_score' column in signal data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Check if 'ticker' is in index
        if 'ticker' not in signals.index.names:
            fig.add_annotation(
                text="Signal data does not have 'ticker' in index",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Get unique tickers
        tickers = signals.index.get_level_values('ticker').unique()
        
        # Plot signal scores for each ticker (limit to top 10 for readability)
        for ticker in list(tickers)[:10]:
            try:
                ticker_signals = signals.loc[ticker, 'final_score']
                # Convert index to datetime if needed
                if not isinstance(ticker_signals.index, pd.DatetimeIndex):
                    ticker_signals.index = pd.to_datetime(ticker_signals.index)
                
                fig.add_trace(
                    go.Scatter(
                        x=ticker_signals.index,
                        y=ticker_signals.values,
                        name=ticker,
                        mode='lines'
                    ),
                    row=1,
                    col=1
                )
            except Exception as e:
                logger.warning(f"Error plotting signals for {ticker}: {str(e)}")
        
        # Plot signal distribution
        try:
            latest_signals = signals.groupby('ticker').last()
            if 'final_score' in latest_signals.columns and not latest_signals.empty:
                fig.add_trace(
                    go.Histogram(
                        x=latest_signals['final_score'].values,
                        name='Signal Distribution',
                        nbinsx=20
                    ),
                    row=2,
                    col=1
                )
        except Exception as e:
            logger.warning(f"Error plotting signal distribution: {str(e)}")
            # Add text annotation instead
            fig.add_annotation(
                text=f"Could not plot distribution: {str(e)}",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text='Alpha Signals Analysis',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in plot_signals: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a blank figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting signals: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=800)
        return fig

def plot_backtest_results(results: Dict[str, Any]) -> go.Figure:
    """
    Create plotly figure for backtest results.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        plotly Figure object
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                'Portfolio Value',
                'Drawdown',
                'Position Allocation'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Check if results contain positions
        if 'positions' not in results or results['positions'] is None or len(results['positions']) < 2:
            fig.add_annotation(
                text="Insufficient backtest data to plot results",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=1000)
            return fig
        
        # Extract positions
        positions = results['positions']
        
        # Ensure index is datetime
        if not isinstance(positions.index, pd.DatetimeIndex):
            positions.index = pd.to_datetime(positions.index)
        
        # Sort by date
        positions = positions.sort_index()
        
        # Plot portfolio value
        fig.add_trace(
            go.Scatter(
                x=positions.index,
                y=positions.values,
                name='Portfolio Value',
                mode='lines'
            ),
            row=1,
            col=1
        )
        
        # Add initial capital line
        if 'initial_capital' in results:
            initial_capital = results['initial_capital']
        else:
            initial_capital = positions.iloc[0]
            
        fig.add_trace(
            go.Scatter(
                x=[positions.index[0], positions.index[-1]],
                y=[initial_capital, initial_capital],
                name='Initial Capital',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ),
            row=1,
            col=1
        )
        
        # Plot drawdown
        try:
            returns = positions.pct_change().fillna(0)
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns / peak - 1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name='Drawdown',
                    mode='lines',
                    line=dict(color='red')
                ),
                row=2,
                col=1
            )
            
            # Add stop loss line at -50%
            fig.add_trace(
                go.Scatter(
                    x=[drawdown.index[0], drawdown.index[-1]],
                    y=[-50, -50],
                    name='Stop Loss (-50%)',
                    mode='lines',
                    line=dict(dash='dash', color='black')
                ),
                row=2,
                col=1
            )
            
        except Exception as e:
            logger.warning(f"Error calculating drawdown: {str(e)}")
            fig.add_annotation(
                text=f"Could not calculate drawdown: {str(e)}",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=1
            )
        
        # Plot position allocation
        try:
            if 'holdings' in results and 'cash' in results:
                holdings = results['holdings'].copy()
                cash = results['cash']
                
                # Get total value
                total_value = holdings.sum(axis=1) + cash
                
                # Convert to percentage allocation
                for col in holdings.columns:
                    holdings[col] = holdings[col] / total_value * 100
                
                # Add cash allocation
                holdings['Cash'] = cash / total_value * 100
                
                # Keep only columns with meaningful allocations (>1% at some point)
                meaningful_cols = [col for col in holdings.columns if holdings[col].max() > 1]
                
                if meaningful_cols:
                    # Plot top 5 positions + cash, aggregate others
                    top_positions = holdings[meaningful_cols].mean().nlargest(5).index.tolist()
                    if 'Cash' in meaningful_cols and 'Cash' not in top_positions:
                        top_positions.append('Cash')
                    
                    # Add 'Other' category for remaining positions
                    other_positions = [col for col in meaningful_cols if col not in top_positions]
                    if other_positions:
                        holdings['Other'] = holdings[other_positions].sum(axis=1)
                        plot_cols = top_positions + ['Other']
                    else:
                        plot_cols = top_positions
                    
                    # Plot each position
                    for i, col in enumerate(plot_cols):
                        fig.add_trace(
                            go.Scatter(
                                x=holdings.index,
                                y=holdings[col],
                                name=col,
                                mode='lines',
                                stackgroup='one'
                            ),
                            row=3,
                            col=1
                        )
                else:
                    fig.add_annotation(
                        text="No significant positions in portfolio",
                        xref="x domain", yref="y domain",
                        x=0.5, y=0.5,
                        showarrow=False,
                        row=3, col=1
                    )
            else:
                fig.add_annotation(
                    text="No holdings data available",
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5,
                    showarrow=False,
                    row=3, col=1
                )
                
        except Exception as e:
            logger.warning(f"Error plotting position allocation: {str(e)}")
            fig.add_annotation(
                text=f"Could not plot position allocation: {str(e)}",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text='Backtest Results',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in plot_backtest_results: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a blank figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting backtest results: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=1000)
        return fig

def plot_trades(trades: pd.DataFrame) -> go.Figure:
    """
    Create plotly figure for trade analysis.
    
    Args:
        trades: DataFrame with trade information
        
    Returns:
        plotly Figure object
    """
    try:
        if trades.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trade data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Trade Count by Ticker',
                'Trade Value Distribution',
                'Buy vs Sell Counts',
                'Profit/Loss Distribution'
            ),
            specs=[[{}, {}], [{"type": "domain"}, {}]],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # Plot trade count by ticker
        ticker_counts = trades['ticker'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=ticker_counts.index,
                y=ticker_counts.values,
                name='Trade Count'
            ),
            row=1,
            col=1
        )
        
        # Plot trade value distribution
        fig.add_trace(
            go.Histogram(
                x=trades['value'],
                name='Trade Value',
                nbinsx=20
            ),
            row=1,
            col=2
        )
        
        # Plot buy vs sell counts
        action_counts = trades['action'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=action_counts.index,
                values=action_counts.values,
                name='Action'
            ),
            row=2,
            col=1
        )
        
        # Plot profit/loss distribution
        if 'profit' in trades.columns:
            fig.add_trace(
                go.Histogram(
                    x=trades['profit'],
                    name='Profit/Loss',
                    nbinsx=20
                ),
                row=2,
                col=2
            )
        else:
            fig.add_annotation(
                text="No profit/loss data available",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text='Trade Analysis',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in plot_trades: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a blank figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting trades: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=800)
        return fig

def display_metrics(metrics: Dict[str, float]) -> None:
    """
    Display backtest metrics in a nice format.
    
    Args:
        metrics: Dictionary with backtest metrics
    """
    try:
        # Group metrics into categories
        return_metrics = {
            'Total Return': metrics.get('total_return', 0),
            'Annual Return': metrics.get('annual_return', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Final 30d Sharpe': metrics.get('final_30d_sharpe', 0)
        }
        
        risk_metrics = {
            'Max Drawdown': metrics.get('max_drawdown', 0),
            'Annual Volatility': metrics.get('annual_volatility', 0),
            'Return/Max Drawdown': metrics.get('return_over_max_drawdown', 0)
        }
        
        exposure_metrics = {
            'Avg Cash (%)': metrics.get('avg_cash_pct', 0),
            'Avg Exposure (%)': metrics.get('avg_exposure', 0)
        }
        
        trade_metrics = {
            'Total Trades': metrics.get('total_trades', 0),
            'Win Rate (%)': metrics.get('win_rate', 0),
            'Total Commission': metrics.get('total_commission', 0)
        }
        
        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Risk", "Exposure", "Trades"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Return", f"{return_metrics['Total Return']:.2f}%")
                st.metric("Sharpe Ratio", f"{return_metrics['Sharpe Ratio']:.2f}")
            
            with col2:
                st.metric("Annual Return", f"{return_metrics['Annual Return']:.2f}%")
                st.metric("Final 30d Sharpe", f"{return_metrics['Final 30d Sharpe']:.2f}")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Max Drawdown", f"{risk_metrics['Max Drawdown']:.2f}%")
                st.metric("Return/Max Drawdown", f"{risk_metrics['Return/Max Drawdown']:.2f}")
            
            with col2:
                st.metric("Annual Volatility", f"{risk_metrics['Annual Volatility']:.2f}%")
                
                # Add visual indicator for strategy quality
                if risk_metrics['Return/Max Drawdown'] > 1:
                    st.success("Good risk/return ratio (>1)")
                elif risk_metrics['Return/Max Drawdown'] > 0:
                    st.warning("Mediocre risk/return ratio (0-1)")
                else:
                    st.error("Poor risk/return ratio (<0)")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Cash", f"{exposure_metrics['Avg Cash (%)']:.2f}%")
            
            with col2:
                st.metric("Average Exposure", f"{exposure_metrics['Avg Exposure (%)']:.2f}%")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Trades", f"{int(trade_metrics['Total Trades'])}")
                st.metric("Win Rate", f"{trade_metrics['Win Rate (%)']:.2f}%")
            
            with col2:
                st.metric("Total Commission", f"${trade_metrics['Total Commission']:.2f}")
                
                # Add trade quality indicator
                if trade_metrics['Win Rate (%)'] > 50:
                    st.success("Good win rate (>50%)")
                elif trade_metrics['Win Rate (%)'] > 40:
                    st.warning("Mediocre win rate (40-50%)")
                else:
                    st.error("Poor win rate (<40%)")
        
        # Display all metrics in expandable section
        with st.expander("Show All Metrics"):
            # Convert metrics to DataFrame
            metrics_df = pd.Series(metrics).to_frame('Value')
            
            # Format values
            formatted_metrics = {}
            for metric, value in metrics.items():
                if 'return' in metric.lower() or 'drawdown' in metric.lower() or 'volatility' in metric.lower() or 'rate' in metric.lower() or 'pct' in metric.lower() or 'exposure' in metric.lower():
                    formatted_metrics[metric] = f"{value:.2f}%"
                elif 'ratio' in metric.lower() or 'sharpe' in metric.lower() or 'sortino' in metric.lower() or 'alpha' in metric.lower() or 'beta' in metric.lower():
                    formatted_metrics[metric] = f"{value:.2f}"
                elif 'trades' in metric.lower() or 'buys' in metric.lower() or 'sells' in metric.lower():
                    formatted_metrics[metric] = f"{int(value)}"
                elif 'commission' in metric.lower() or 'profit' in metric.lower() or 'loss' in metric.lower():
                    formatted_metrics[metric] = f"${value:.2f}"
                else:
                    formatted_metrics[metric] = f"{value:.4f}"
            
            # Create formatted DataFrame
            formatted_df = pd.Series(formatted_metrics).to_frame('Value')
            
            # Display metrics
            st.dataframe(formatted_df)
        
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying metrics: {str(e)}")

def main():
    """Main Streamlit app."""
    try:
        st.set_page_config(
            page_title="Quant Alpha Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("Quant Alpha Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=start_date
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=end_date
        )
        
        # Initial capital
        initial_capital = st.sidebar.number_input(
            "Initial Capital",
            value=1_000_000,
            step=100_000,
            min_value=10000
        )
        
        # Commission
        commission = st.sidebar.number_input(
            "Commission (%)",
            value=0.1,
            step=0.05,
            min_value=0.0,
            max_value=5.0
        ) / 100
        
        # Stock selection
        tickers_input = st.sidebar.text_area(
            "Stock Tickers (comma separated, leave empty for default)",
            value="",
            help="Enter comma-separated stock tickers (e.g. AAPL, MSFT, GOOGL)"
        )
        
        # Advanced options in expander
        with st.sidebar.expander("Advanced Options"):
            top_n = st.number_input(
                "Top N Stocks",
                value=5,
                min_value=1,
                max_value=20,
                help="Number of top stocks to hold in portfolio"
            )
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                value=20,
                min_value=5,
                max_value=50,
                help="Maximum position size as percentage of portfolio"
            ) / 100
            
            stop_loss_pct = st.slider(
                "Stop Loss (%)",
                value=50,
                min_value=10,
                max_value=90,
                help="Stop trading if portfolio value falls below this percentage of initial capital"
            ) / 100
            
            cash_buffer_pct = st.slider(
                "Cash Buffer (%)",
                value=10,
                min_value=0,
                max_value=30,
                help="Percentage of portfolio to keep in cash"
            ) / 100
            
            weekly_rebalance = st.checkbox(
                "Weekly Rebalance",
                value=True,
                help="If checked, rebalance weekly regardless of rebalance frequency"
            )
        
        # Process tickers input
        custom_tickers = None
        if tickers_input.strip():
            custom_tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
            if len(custom_tickers) > 0:
                st.sidebar.info(f"Using {len(custom_tickers)} custom tickers")
            else:
                custom_tickers = None
        
        # Run analysis
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                try:
                    # Create status containers
                    data_status = st.empty()
                    signals_status = st.empty()
                    backtest_status = st.empty()
                    
                    # Fetch data
                    data_status.info("Fetching market data...")
                    fetcher = MarketDataFetcher(tickers=custom_tickers)
                    data = fetcher.load_historical_data(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    
                    if data.empty:
                        st.error("No market data available for the selected tickers and date range.")
                        return
                        
                    data_status.success(f"Fetched market data for {len(data.index.get_level_values('ticker').unique())} tickers")
                    
                    # Generate signals
                    signals_status.info("Generating alpha signals...")
                    generator = AlphaSignalGenerator(data)
                    signals = generator.compute_all_signals()
                    
                    if signals.empty:
                        st.error("No signals could be generated from the market data.")
                        return
                        
                    signals_status.success(f"Generated alpha signals with shape {signals.shape}")
                    
                    # Run backtest
                    backtest_status.info("Running backtest...")
                    engine = BacktestEngine(
                        data,
                        signals,
                        initial_capital=initial_capital,
                        commission=commission,
                        top_n=top_n,
                        max_position_size=max_position_size,
                        stop_loss_pct=stop_loss_pct,
                        cash_buffer_pct=cash_buffer_pct,
                        weekly_rebalance=weekly_rebalance
                    )
                    results = engine.run_backtest()
                    backtest_status.success("Backtest completed successfully")
                    
                    # Add initial capital to results for plotting
                    results['initial_capital'] = initial_capital
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3 = st.tabs(["Performance", "Signals", "Trades"])
                    
                    with tab1:
                        # Display metrics
                        st.header("Performance Metrics")
                        display_metrics(results['metrics'])
                        
                        # Display backtest results
                        st.header("Backtest Results")
                        st.plotly_chart(plot_backtest_results(results), use_container_width=True)
                    
                    with tab2:
                        # Display signals
                        st.header("Alpha Signals")
                        st.plotly_chart(plot_signals(signals), use_container_width=True)
                    
                    with tab3:
                        # Display trade analysis
                        st.header("Trade Analysis")
                        if 'trades' in results and not results['trades'].empty:
                            st.plotly_chart(plot_trades(results['trades']), use_container_width=True)
                        else:
                            st.info("No trade data available")
                    
                    # Save results to S3
                    try:
                        # Save signals
                        s3.save_dataframe(signals, "signals_latest.parquet")
                        
                        # Save backtest results (metrics only, as positions might be too large)
                        s3.save_json(results['metrics'], "backtest_latest.json")
                        
                        st.sidebar.success("Results saved to S3")
                    except Exception as e:
                        logger.warning(f"Error saving results to S3: {str(e)}")
                        st.sidebar.warning("Could not save results to S3")
                    
                except Exception as e:
                    logger.error(f"Error running analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"Error running analysis: {str(e)}")
                    st.error("Check the logs for more details")
        
        # Display saved results
        st.sidebar.header("Saved Results")
        if st.sidebar.button("Load Latest Results"):
            with st.spinner("Loading saved results..."):
                try:
                    # Load latest signals
                    signals = s3.load_dataframe("signals_latest.parquet")
                    if signals is not None and not signals.empty:
                        st.header("Latest Alpha Signals")
                        st.plotly_chart(plot_signals(signals), use_container_width=True)
                    else:
                        st.info("No saved signals found")
                    
                    # Load latest backtest
                    metrics = s3.load_json("backtest_latest.json")
                    if metrics is not None:
                        st.header("Latest Performance Metrics")
                        display_metrics(metrics)
                    else:
                        st.info("No saved backtest results found")
                        
                except Exception as e:
                    logger.error(f"Error loading results: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"Error loading results: {str(e)}")
    
    except Exception as e:
        logger.error(f"Fatal error in main app: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Fatal error in application: {str(e)}")
        st.error("Please check the logs for more details")

if __name__ == "__main__":
    main() 