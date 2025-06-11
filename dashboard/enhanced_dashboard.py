"""
Enhanced Dashboard for Quant Alpha Backtesting System.

This dashboard visualizes the results from the enhanced backtesting system,
showing market regimes, portfolio performance, and position allocations.
"""

import logging
import os
import sys
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.enhanced_engine import EnhancedBacktestEngine
from utils.enhanced_strategy import EnhancedStrategy
from utils.market_regime import MarketRegimeDetector
from utils.signal_combiner import SignalCombiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_backtest_results(results_dir: str) -> Dict[str, Any]:
    """
    Load backtest results from directory.
    
    Args:
        results_dir: Directory containing backtest results
        
    Returns:
        Dictionary with backtest results
    """
    try:
        if not os.path.exists(results_dir):
            return {}
        
        results = {}
        
        # Load portfolio value
        portfolio_value_path = os.path.join(results_dir, 'portfolio_value.csv')
        if os.path.exists(portfolio_value_path):
            portfolio_value = pd.read_csv(portfolio_value_path, index_col=0, parse_dates=True)
            results['portfolio_value'] = portfolio_value
        
        # Load holdings
        holdings_path = os.path.join(results_dir, 'holdings.csv')
        if os.path.exists(holdings_path):
            holdings = pd.read_csv(holdings_path, index_col=0, parse_dates=True)
            results['holdings'] = holdings
        
        # Load cash
        cash_path = os.path.join(results_dir, 'cash.csv')
        if os.path.exists(cash_path):
            cash = pd.read_csv(cash_path, index_col=0, parse_dates=True)
            results['cash'] = cash.iloc[:, 0]
        
        # Load trades
        trades_path = os.path.join(results_dir, 'trades.csv')
        if os.path.exists(trades_path):
            trades = pd.read_csv(trades_path, parse_dates=['date'])
            results['trades'] = trades
        
        # Load metrics
        metrics_path = os.path.join(results_dir, 'metrics.csv')
        if os.path.exists(metrics_path):
            metrics = pd.read_csv(metrics_path)
            results['metrics'] = metrics.iloc[0].to_dict()
        
        # Load regimes
        regimes_path = os.path.join(results_dir, 'regimes.csv')
        if os.path.exists(regimes_path):
            regimes = pd.read_csv(regimes_path, parse_dates=['date'])
            results['regimes'] = regimes
        
        # Load factor weights
        factor_weights_path = os.path.join(results_dir, 'factor_weights.csv')
        if os.path.exists(factor_weights_path):
            factor_weights = pd.read_csv(factor_weights_path, parse_dates=['date'])
            results['factor_weights'] = factor_weights
        
        # Load cash weights
        cash_weights_path = os.path.join(results_dir, 'cash_weights.csv')
        if os.path.exists(cash_weights_path):
            cash_weights = pd.read_csv(cash_weights_path, parse_dates=['date'])
            results['cash_weights'] = cash_weights
            
        return results
        
    except Exception as e:
        logger.error(f"Error loading backtest results: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def plot_performance(results: Dict[str, Any]) -> go.Figure:
    """
    Create a plot of portfolio performance.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        Plotly figure
    """
    try:
        if 'portfolio_value' not in results:
            fig = go.Figure()
            fig.add_annotation(
                text="No portfolio data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        portfolio_value = results['portfolio_value']
        
        # Create subplots
        fig = make_subplots(
            rows=3, 
            cols=1,
            subplot_titles=(
                'Portfolio Value',
                'Drawdown',
                'Rolling 30-Day Returns'
            ),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.iloc[:, 0],
                name='Portfolio',
                line=dict(color='blue')
            ),
            row=1,
            col=1
        )
        
        # Add benchmark if available
        if 'benchmark' in results:
            benchmark = results['benchmark']
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark.iloc[:, 0],
                    name='Benchmark',
                    line=dict(color='gray', dash='dash')
                ),
                row=1,
                col=1
            )
        
        # Drawdown
        if len(portfolio_value) > 1:
            returns = portfolio_value.iloc[:, 0].pct_change().fillna(0)
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative / peak - 1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red')
                ),
                row=2,
                col=1
            )
            
            # Add horizontal lines for 5%, 10%, 20% drawdowns
            for dd_level in [-5, -10, -20]:
                fig.add_trace(
                    go.Scatter(
                        x=[drawdown.index[0], drawdown.index[-1]],
                        y=[dd_level, dd_level],
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                        name=f'{abs(dd_level)}% Drawdown',
                        showlegend=False
                    ),
                    row=2,
                    col=1
                )
            
            # Rolling 30-day returns
            rolling_returns = returns.rolling(30).mean() * 252 * 100  # Annualized
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_returns.index,
                    y=rolling_returns,
                    name='30-Day Returns (Ann.)',
                    line=dict(color='green')
                ),
                row=3,
                col=1
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[rolling_returns.index[0], rolling_returns.index[-1]],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    showlegend=False
                ),
                row=3,
                col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title='Portfolio Performance',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text='Value ($)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        fig.update_yaxes(title_text='Returns (%)', row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting performance: {str(e)}")
        logger.error(traceback.format_exc())
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting performance: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

def plot_market_regimes(results: Dict[str, Any]) -> go.Figure:
    """
    Plot market regimes from backtest results.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        Plotly figure
    """
    try:
        if 'regimes' not in results or results['regimes'].empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No regime data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        regimes_df = results['regimes']
        
        # Parse regime dictionaries
        if 'regime' in regimes_df.columns:
            if isinstance(regimes_df['regime'].iloc[0], str):
                # Convert string representations to dictionaries
                regimes_df['regime'] = regimes_df['regime'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            
            # Extract trend and volatility
            regimes_df['trend'] = regimes_df['regime'].apply(
                lambda x: x.get('trend', 'UNKNOWN') if isinstance(x, dict) else 'UNKNOWN'
            )
            regimes_df['volatility'] = regimes_df['regime'].apply(
                lambda x: x.get('volatility', 'UNKNOWN') if isinstance(x, dict) else 'UNKNOWN'
            )
            regimes_df['combined'] = regimes_df['regime'].apply(
                lambda x: x.get('combined', 'unknown') if isinstance(x, dict) else 'unknown'
            )
        
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Market Trend', 'Market Volatility'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Define colors for trends and volatility
        trend_colors = {
            'BULLISH': 'green',
            'BEARISH': 'red',
            'SIDEWAYS': 'gray',
            'UNKNOWN': 'black'
        }
        
        vol_colors = {
            'LOW': 'green',
            'MEDIUM': 'orange',
            'HIGH': 'red',
            'UNKNOWN': 'black'
        }
        
        # Plot trends
        for trend in trend_colors:
            trend_df = regimes_df[regimes_df['trend'] == trend]
            if not trend_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=trend_df['date'],
                        y=[1] * len(trend_df),
                        mode='markers',
                        name=f'Trend: {trend}',
                        marker=dict(
                            color=trend_colors[trend],
                            size=10,
                            symbol='circle'
                        )
                    ),
                    row=1,
                    col=1
                )
        
        # Plot volatility
        for vol in vol_colors:
            vol_df = regimes_df[regimes_df['volatility'] == vol]
            if not vol_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=vol_df['date'],
                        y=[1] * len(vol_df),
                        mode='markers',
                        name=f'Volatility: {vol}',
                        marker=dict(
                            color=vol_colors[vol],
                            size=10,
                            symbol='square'
                        )
                    ),
                    row=2,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            height=500,
            title='Market Regimes',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_yaxes(showticklabels=False, range=[0, 2], row=1, col=1)
        fig.update_yaxes(showticklabels=False, range=[0, 2], row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting market regimes: {str(e)}")
        logger.error(traceback.format_exc())
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting market regimes: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

def plot_factor_weights(results: Dict[str, Any]) -> go.Figure:
    """
    Plot factor weights over time.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        Plotly figure
    """
    try:
        if 'factor_weights' not in results or results['factor_weights'].empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No factor weight data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        weights_df = results['factor_weights']
        
        # Get factor columns (excluding date)
        factor_cols = [col for col in weights_df.columns if col != 'date']
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each factor
        for factor in factor_cols:
            fig.add_trace(
                go.Scatter(
                    x=weights_df['date'],
                    y=weights_df[factor],
                    mode='lines',
                    name=factor,
                    stackgroup='one'  # Stack the weights
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Factor Weights Over Time',
            xaxis_title='Date',
            yaxis_title='Weight',
            height=500,
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
        logger.error(f"Error plotting factor weights: {str(e)}")
        logger.error(traceback.format_exc())
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting factor weights: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

def plot_position_allocation(results: Dict[str, Any]) -> go.Figure:
    """
    Plot position allocation over time.
    
    Args:
        results: Dictionary with backtest results
        
    Returns:
        Plotly figure
    """
    try:
        if 'holdings' not in results or results['holdings'].empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No holdings data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        holdings = results['holdings']
        cash = results['cash'] if 'cash' in results else None
        
        # Calculate total value
        if cash is not None:
            total_value = holdings.sum(axis=1) + cash
        else:
            total_value = holdings.sum(axis=1)
        
        # Convert to percentage
        percentage_holdings = holdings.div(total_value, axis=0) * 100
        
        # Add cash percentage if available
        if cash is not None:
            percentage_holdings['Cash'] = cash / total_value * 100
        
        # Get top 8 positions (by average allocation)
        top_positions = percentage_holdings.mean().sort_values(ascending=False).head(8).index.tolist()
        
        # Combine other positions
        other_positions = [col for col in percentage_holdings.columns if col not in top_positions]
        if other_positions:
            percentage_holdings['Other'] = percentage_holdings[other_positions].sum(axis=1)
            plot_positions = top_positions + ['Other']
        else:
            plot_positions = top_positions
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each position
        for position in plot_positions:
            fig.add_trace(
                go.Scatter(
                    x=percentage_holdings.index,
                    y=percentage_holdings[position],
                    mode='lines',
                    name=position,
                    stackgroup='one'  # Stack the allocations
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Position Allocation Over Time',
            xaxis_title='Date',
            yaxis_title='Allocation (%)',
            height=500,
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
        logger.error(f"Error plotting position allocation: {str(e)}")
        logger.error(traceback.format_exc())
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting position allocation: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig

def display_metrics(metrics: Dict[str, float]) -> None:
    """
    Display metrics in a nicely formatted layout.
    
    Args:
        metrics: Dictionary of backtest metrics
    """
    try:
        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Risk", "Exposure", "Trades"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
            
            with col2:
                st.metric("Annual Return", f"{metrics.get('annual_return', 0):.2f}%")
            
            with col3:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
            
            with col2:
                st.metric("Annual Volatility", f"{metrics.get('annual_volatility', 0):.2f}%")
            
            with col3:
                st.metric("Return/Max DD", f"{metrics.get('return_over_max_drawdown', 0):.2f}")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Cash", f"{metrics.get('avg_cash_pct', 0):.2f}%")
            
            with col2:
                st.metric("Avg Exposure", f"{metrics.get('avg_exposure', 0):.2f}%")
        
        with tab4:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", f"{int(metrics.get('total_trades', 0))}")
            
            with col2:
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
            
            with col3:
                st.metric("Total Commission", f"${metrics.get('total_commission', 0):.2f}")
        
        # Show all metrics in an expander
        with st.expander("All Metrics"):
            # Convert to DataFrame for better display
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
            st.dataframe(metrics_df)
    
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying metrics: {str(e)}")

def main():
    """Main Streamlit app function."""
    st.set_page_config(
        page_title="Enhanced Quant Alpha Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Enhanced Quant Alpha Dashboard")
    st.markdown("### Advanced Backtest Visualization with Market Regime Detection")
    
    # Sidebar
    st.sidebar.header("Options")
    
    # Results directory selection
    results_dir = st.sidebar.text_input(
        "Results Directory",
        value="results/enhanced/latest",
        help="Path to backtest results directory"
    )
    
    # Browse results button
    if st.sidebar.button("Browse Results"):
        results_dirs = []
        # Look for results directories
        for root, dirs, files in os.walk("results"):
            for dir in dirs:
                if "run_" in dir or "optimization_" in dir:
                    results_dirs.append(os.path.join(root, dir))
        
        if results_dirs:
            results_dir = st.sidebar.selectbox(
                "Select Results Directory",
                options=results_dirs,
                index=0
            )
        else:
            st.sidebar.warning("No results directories found")
    
    # Load results button
    if st.sidebar.button("Load Results"):
        with st.spinner("Loading backtest results..."):
            # Check if directory exists
            if not os.path.exists(results_dir):
                st.error(f"Results directory not found: {results_dir}")
                return
            
            # Load results
            results = load_backtest_results(results_dir)
            
            if not results:
                st.error("No backtest results found in the specified directory")
                return
            
            # Store results in session state
            st.session_state.results = results
            st.success(f"Loaded backtest results from {results_dir}")
    
    # Check if results are loaded
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        # Display metrics
        if 'metrics' in results:
            st.subheader("Performance Metrics")
            display_metrics(results['metrics'])
        
        # Create tabs for different visualizations
        perf_tab, regime_tab, factor_tab, alloc_tab = st.tabs([
            "Performance", "Market Regimes", "Factor Weights", "Position Allocation"
        ])
        
        with perf_tab:
            st.plotly_chart(plot_performance(results), use_container_width=True)
        
        with regime_tab:
            st.plotly_chart(plot_market_regimes(results), use_container_width=True)
        
        with factor_tab:
            st.plotly_chart(plot_factor_weights(results), use_container_width=True)
        
        with alloc_tab:
            st.plotly_chart(plot_position_allocation(results), use_container_width=True)
    
    else:
        # Show instructions if no results are loaded
        st.info("Please load backtest results using the sidebar options")
        
        # Create empty placeholder for each tab
        perf_tab, regime_tab, factor_tab, alloc_tab = st.tabs([
            "Performance", "Market Regimes", "Factor Weights", "Position Allocation"
        ])
        
        with perf_tab:
            st.info("Load backtest results to view performance charts")
        
        with regime_tab:
            st.info("Load backtest results to view market regime analysis")
        
        with factor_tab:
            st.info("Load backtest results to view factor weight evolution")
        
        with alloc_tab:
            st.info("Load backtest results to view position allocation")

if __name__ == "__main__":
    main()