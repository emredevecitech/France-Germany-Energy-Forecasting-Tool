"""
Configuration template for the France-Germany Energy Forecasting Tool
Copy this file to config.py and add your API keys
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # ENTSO-E API Configuration
    # Get your API key from: https://transparency.entsoe.eu/
    ENTSOE_API_KEY = os.getenv('ENTSOE_API_KEY', 'your_entsoe_api_key_here')
    ENTSOE_BASE_URL = os.getenv('ENTSOE_BASE_URL', 'https://web-api.tp.entsoe.eu/api')
    
    # Weather API Configuration
    # Get your API key from: https://openweathermap.org/api
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', 'your_weather_api_key_here')
    WEATHER_BASE_URL = os.getenv('WEATHER_BASE_URL', 'https://api.openweathermap.org/data/2.5')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///energy_forecasting.db')
    
    # Application Configuration
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    UPDATE_INTERVAL_MINUTES = int(os.getenv('UPDATE_INTERVAL_MINUTES', '15'))
    
    # Trading Configuration
    FRANCE_COUNTRY_CODE = os.getenv('FRANCE_COUNTRY_CODE', '10YFR-RTE------C')
    GERMANY_COUNTRY_CODE = os.getenv('GERMANY_COUNTRY_CODE', '10Y1001A1001A83F')
    BORDER_AC = os.getenv('BORDER_AC', '10YFR-RTE------C_10Y1001A1001A83F')
    
    # Forecasting Parameters
    FORECAST_HORIZON_HOURS = 48
    SOLAR_PEAK_HOURS = [11, 12, 13, 14, 15]  # Hours when solar generation typically peaks
    NUCLEAR_OUTAGE_THRESHOLD = 0.1  # Minimum capacity reduction to consider as outage
    
    # Risk Management
    VOLATILITY_THRESHOLD = 0.15  # 15% price volatility threshold
    HIGH_VOLATILITY_THRESHOLD = 0.25  # 25% for high volatility alerts
    
    # Demo Mode (for testing without API keys)
    DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'
