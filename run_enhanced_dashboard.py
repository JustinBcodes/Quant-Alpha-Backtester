#!/usr/bin/env python
"""
Run the enhanced dashboard for visualizing backtest results.

This script runs the Streamlit dashboard that visualizes the results from the enhanced backtesting system,
showing market regimes, portfolio performance, and position allocations.
"""

import os
import subprocess
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the enhanced dashboard."""
    try:
        # Check if streamlit is installed
        try:
            import streamlit
            logger.info("Streamlit is installed")
        except ImportError:
            logger.warning("Streamlit is not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            logger.info("Streamlit installed successfully")
        
        # Make sure the dashboard directory exists
        dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
        dashboard_file = os.path.join(dashboard_dir, "enhanced_dashboard.py")
        
        if not os.path.exists(dashboard_file):
            logger.error(f"Dashboard file not found: {dashboard_file}")
            return 1
        
        # Create directories for results if they don't exist
        os.makedirs("results/enhanced", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Run streamlit
        logger.info(f"Running dashboard: {dashboard_file}")
        os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
        subprocess.call(["streamlit", "run", dashboard_file])
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 