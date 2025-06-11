"""
AWS API Gateway handler for the Quant Alpha system.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException
from mangum import Mangum

from config import AWS_REGION
from data.market_data import MarketDataFetcher
from factors.alpha_signals import AlphaSignalGenerator
from backtest.engine import BacktestEngine
from aws.utils import S3Handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIGatewayHandler:
    """Handler for API Gateway interactions."""
    
    def __init__(self):
        """Initialize the API Gateway handler."""
        self.client = boto3.client('apigateway', region_name=AWS_REGION)
        self.api_id = None
        self.stage_name = None
    
    def create_api(self, name: str, description: str) -> str:
        """
        Create a new API Gateway API.
        
        Args:
            name: Name of the API
            description: Description of the API
            
        Returns:
            API ID
        """
        try:
            response = self.client.create_rest_api(
                name=name,
                description=description
            )
            self.api_id = response['id']
            return self.api_id
        except Exception as e:
            logger.error(f"Error creating API: {str(e)}")
            raise
    
    def create_stage(self, stage_name: str) -> None:
        """
        Create a new stage for the API.
        
        Args:
            stage_name: Name of the stage (e.g., 'prod', 'dev')
        """
        if not self.api_id:
            raise ValueError("API ID not set. Create API first.")
        
        try:
            self.client.create_deployment(
                restApiId=self.api_id,
                stageName=stage_name
            )
            self.stage_name = stage_name
        except Exception as e:
            logger.error(f"Error creating stage: {str(e)}")
            raise
    
    def get_api_url(self) -> str:
        """
        Get the API URL.
        
        Returns:
            API URL
        """
        if not self.api_id or not self.stage_name:
            raise ValueError("API ID or stage name not set")
        
        return f"https://{self.api_id}.execute-api.{AWS_REGION}.amazonaws.com/{self.stage_name}"

# Initialize FastAPI app
app = FastAPI(
    title="Quant Alpha API",
    description="API for generating alpha signals and running backtests",
    version="1.0.0"
)

# Initialize AWS handlers
s3 = S3Handler()
api_gateway = APIGatewayHandler()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/signals")
async def get_signals(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """
    Get latest alpha signals.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary with signal data
    """
    try:
        # Check if signals exist in S3
        key = f"signals_{start_date}_{end_date}.parquet"
        signals = s3.load_dataframe(key)
        
        if signals is None:
            # Generate new signals
            fetcher = MarketDataFetcher()
            data = fetcher.load_historical_data(start_date, end_date)
            
            generator = AlphaSignalGenerator(data)
            signals = generator.compute_all_signals()
            
            # Save to S3
            s3.save_dataframe(signals, key)
        
        # Get latest signals
        latest_signals = signals.groupby('ticker').last()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "signals": latest_signals.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def run_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    commission: float = 0.001
) -> Dict:
    """
    Run backtest and return results.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Initial capital for backtest
        commission: Commission rate per trade
        
    Returns:
        Dictionary with backtest results
    """
    try:
        # Check if results exist in S3
        key = f"backtest_{start_date}_{end_date}.json"
        results = s3.load_json(key)
        
        if results is None:
            # Run new backtest
            fetcher = MarketDataFetcher()
            data = fetcher.load_historical_data(start_date, end_date)
            
            generator = AlphaSignalGenerator(data)
            signals = generator.compute_all_signals()
            
            engine = BacktestEngine(
                data,
                signals,
                initial_capital=initial_capital,
                commission=commission
            )
            results = engine.run_backtest()
            
            # Save to S3
            s3.save_json(results['metrics'], key)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Create handler for AWS Lambda
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 