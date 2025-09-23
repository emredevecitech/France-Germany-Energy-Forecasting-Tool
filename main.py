"""
Main application entry point for France-Germany Energy Forecasting Tool
"""
import logging
import sys
from datetime import datetime
from forecasting_engine import ForecastingEngine
from dashboard.app import app
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('energy_forecasting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    try:
        logger.info("Starting France-Germany Energy Forecasting Tool")
        
        # Initialize forecasting engine
        forecasting_engine = ForecastingEngine()
        
        if not forecasting_engine.initialize():
            logger.error("Failed to initialize forecasting engine")
            return 1
        
        logger.info("Forecasting engine initialized successfully")
        
        # Start monitoring
        forecasting_engine.start_monitoring()
        
        # Run dashboard
        logger.info("Starting dashboard on http://localhost:8050")
        app.run(debug=Config.DEBUG, host='0.0.0.0', port=8050)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
