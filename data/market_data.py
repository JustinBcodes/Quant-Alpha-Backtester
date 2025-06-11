import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import yfinance as yf
import numpy as np
import os

from config import RAW_DATA_DIR, DEFAULT_TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketData:
    """Class to handle market data operations."""
    
    def __init__(self, tickers: Optional[List[str]] = None):
        """
        Initialize the MarketData class.
        
        Args:
            tickers: List of ticker symbols to fetch data for. Defaults to config.DEFAULT_TICKERS.
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.raw_data_dir = Path(RAW_DATA_DIR)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.data = None

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Load market data from local storage or fetch from API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with market data
        """
        self.data = MarketDataFetcher(self.tickers).load_historical_data(
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        return self.data
        
    def get_returns(self, window: int = 1) -> pd.DataFrame:
        """
        Calculate returns for the loaded data.
        
        Args:
            window: Window for returns calculation
            
        Returns:
            DataFrame with returns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Group by ticker and calculate returns
        returns = pd.DataFrame()
        
        for ticker, group in self.data.groupby(level='ticker'):
            # Calculate returns
            ticker_returns = group['Close'].pct_change(window)
            ticker_returns = pd.DataFrame({'returns': ticker_returns})
            ticker_returns.index = group.index
            returns = pd.concat([returns, ticker_returns])
        
        return returns

class DataLoader:
    """Data loader for the backtesting engine."""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to load data from
            tickers: List of tickers to load data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.data_path = data_path
        self.tickers = tickers or DEFAULT_TICKERS
        self.start_date = start_date
        self.end_date = end_date
        
        # Default to 1 year of data if no dates provided
        if not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load market data from file or API.
        
        Returns:
            DataFrame with market data
        """
        # Try to load from specified path
        if self.data_path and os.path.exists(self.data_path):
            if self.data_path.endswith('.csv'):
                data = pd.read_csv(self.data_path, parse_dates=['Date'])
                
                # Check if this is multi-index data or flat data
                if 'ticker' in data.columns:
                    # Set multi-index
                    data.set_index(['ticker', 'Date'], inplace=True)
                else:
                    # Assume this is a pivot table with tickers as columns
                    data.set_index('Date', inplace=True)
                
                logger.info(f"Loaded data from {self.data_path} with shape {data.shape}")
                return data
            elif self.data_path.endswith('.parquet'):
                data = pd.read_parquet(self.data_path)
                logger.info(f"Loaded data from {self.data_path} with shape {data.shape}")
                return data
            else:
                logger.warning(f"Unsupported file format: {self.data_path}")
        
        # If no path or file doesn't exist, use MarketData
        logger.info("Loading data using MarketData")
        market_data = MarketData(self.tickers)
        return market_data.load_data(
            start_date=self.start_date,
            end_date=self.end_date
        )
    
    def load_vix_data(self) -> Optional[pd.Series]:
        """
        Load VIX data for regime detection.
        
        Returns:
            Series with VIX data or None if not available
        """
        vix_path = os.path.join(RAW_DATA_DIR, 'vix.csv')
        if os.path.exists(vix_path):
            try:
                vix_df = pd.read_csv(vix_path, parse_dates=['Date'])
                vix_df.set_index('Date', inplace=True)
                
                # If multiple columns, assume 'Close' is the VIX value
                if 'Close' in vix_df.columns:
                    vix_data = vix_df['Close']
                else:
                    # Otherwise use the first column
                    vix_data = vix_df.iloc[:, 0]
                    
                logger.info(f"Loaded VIX data with shape {vix_data.shape}")
                return vix_data
            except Exception as e:
                logger.warning(f"Error loading VIX data: {str(e)}")
                return None
        else:
            # Try to fetch VIX data
            try:
                vix_ticker = "^VIX"
                vix_data = yf.download(
                    vix_ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )['Close']
                
                # Save for future use
                # Convert to DataFrame if it's not already one
                if isinstance(vix_data, pd.Series):
                    vix_df = vix_data.to_frame(name='Close')
                else:
                    vix_df = vix_data
                
                vix_df.reset_index().to_csv(vix_path, index=False)
                
                logger.info(f"Fetched and saved VIX data to {vix_path}")
                return vix_data
            except Exception as e:
                logger.warning(f"Error fetching VIX data: {str(e)}")
                return None

class MarketDataFetcher:
    """Class to handle fetching and processing market data."""
    
    def __init__(self, tickers: Optional[List[str]] = None):
        """
        Initialize the MarketDataFetcher.
        
        Args:
            tickers: List of ticker symbols to fetch data for. Defaults to config.DEFAULT_TICKERS.
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.raw_data_dir = Path(RAW_DATA_DIR)
        self.raw_data_dir.mkdir(exist_ok=True)
    
    def fetch_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for all tickers.
        
        Args:
            start_date: Start date in YYYY-MM-DD format. Defaults to 1 year ago.
            end_date: End date in YYYY-MM-DD format. Defaults to today.
            interval: Data interval (1d, 1wk, 1mo). Defaults to daily.
            
        Returns:
            DataFrame with multi-index (ticker, date) and OHLCV columns
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {start_date} to {end_date}")
        
        try:
            # Fetch data for all tickers
            data = yf.download(
                self.tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker'
            )
            
            # Process the data into a multi-index DataFrame
            all_data = []
            
            if len(self.tickers) == 1:
                # Single ticker case
                ticker_data = data.copy()
                ticker_data.index.name = 'Date'
                ticker_data.reset_index(inplace=True)
                ticker_data['ticker'] = self.tickers[0]
                all_data.append(ticker_data)
            else:
                # Multiple tickers case
                for ticker in self.tickers:
                    if ticker in data.columns.levels[0]:
                        ticker_data = data[ticker].copy()
                        ticker_data.index.name = 'Date'
                        ticker_data.reset_index(inplace=True)
                        ticker_data['ticker'] = ticker
                        all_data.append(ticker_data)
            
            # Combine all data and set multi-index
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data.set_index(['ticker', 'Date'], inplace=True)
                
                # Save to parquet
                self._save_to_parquet(combined_data, start_date, end_date, interval)
                
                return combined_data
            else:
                logger.error("No data fetched for any ticker")
                raise ValueError("No data fetched for any ticker")
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _save_to_parquet(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        interval: str
    ) -> None:
        """
        Save data to parquet file.
        
        Args:
            data: DataFrame to save
            start_date: Start date of data
            end_date: End date of data
            interval: Data interval
        """
        filename = f"market_data_{start_date}_{end_date}_{interval}.parquet"
        filepath = self.raw_data_dir / filename
        
        try:
            data.to_parquet(filepath)
            logger.info(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Load historical data from parquet file or fetch if not available.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        filename = f"market_data_{start_date}_{end_date}_{interval}.parquet"
        filepath = self.raw_data_dir / filename
        
        if filepath.exists():
            logger.info(f"Loading data from {filepath}")
            data = pd.read_parquet(filepath)
            # Ensure the index is set correctly
            if 'ticker' not in data.index.names:
                if 'ticker' in data.columns:
                    data.set_index(['ticker', 'Date'], inplace=True)
                else:
                    logger.error("Loaded data does not have a 'ticker' column")
                    raise ValueError("Invalid data format: 'ticker' column not found")
            return data
        else:
            logger.info(f"No existing data found at {filepath}, fetching new data")
            return self.fetch_historical_data(start_date, end_date, interval)

if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader()
    data = data_loader.load_data()
    print(f"Loaded data shape: {data.shape}")
    print("\nSample data:")
    print(data.head())
    
    vix_data = data_loader.load_vix_data()
    if vix_data is not None:
        print(f"\nLoaded VIX data shape: {vix_data.shape}")
        print("\nSample VIX data:")
        print(vix_data.head()) 