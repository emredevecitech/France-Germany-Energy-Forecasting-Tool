"""
Weather data client for solar generation forecasting
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class WeatherClient:
    """Client for fetching weather data for solar forecasting"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.WEATHER_API_KEY
        self.base_url = Config.WEATHER_BASE_URL
        self.session = requests.Session()
    
    def get_weather_forecast(self, lat: float, lon: float, days: int = 5) -> pd.DataFrame:
        """Get weather forecast for solar generation prediction"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return pd.DataFrame()
    
    def get_historical_weather(self, lat: float, lon: float, 
                             start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical weather data"""
        try:
            # For historical data, you might need a different API endpoint
            # This is a placeholder for the actual implementation
            url = f"{self.base_url}/history"
            params = {
                'lat': lat,
                'lon': lon,
                'start': int(start_date.timestamp()),
                'end': int(end_date.timestamp()),
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return pd.DataFrame()
    
    def _parse_weather_data(self, data: Dict) -> pd.DataFrame:
        """Parse weather API response into DataFrame"""
        weather_records = []
        
        if 'list' in data:
            for item in data['list']:
                record = {
                    'datetime': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'cloud_cover': item['clouds']['all'],
                    'wind_speed': item['wind']['speed'],
                    'wind_direction': item['wind'].get('deg', 0),
                    'pressure': item['main']['pressure'],
                    'visibility': item.get('visibility', 0),
                    'solar_irradiance': self._calculate_solar_irradiance(item)
                }
                weather_records.append(record)
        
        return pd.DataFrame(weather_records)
    
    def _calculate_solar_irradiance(self, weather_item: Dict) -> float:
        """Calculate solar irradiance based on weather conditions"""
        # Simplified solar irradiance calculation
        # In practice, you'd use more sophisticated models
        
        cloud_cover = weather_item['clouds']['all'] / 100.0
        base_irradiance = 1000  # W/m² for clear sky conditions
        
        # Reduce irradiance based on cloud cover
        solar_irradiance = base_irradiance * (1 - cloud_cover * 0.7)
        
        return max(0, solar_irradiance)
    
    def get_solar_forecast(self, lat: float, lon: float, days: int = 5) -> pd.DataFrame:
        """Get solar generation forecast based on weather data"""
        weather_df = self.get_weather_forecast(lat, lon, days)
        
        if weather_df.empty:
            return pd.DataFrame()
        
        # Calculate solar generation potential
        weather_df['solar_generation_potential'] = weather_df['solar_irradiance'].apply(
            self._estimate_solar_generation
        )
        
        return weather_df
    
    def _estimate_solar_generation(self, irradiance: float) -> float:
        """Estimate solar generation capacity factor from irradiance"""
        # Simplified model - in practice, you'd use more sophisticated models
        # considering panel efficiency, tilt angle, etc.
        
        max_irradiance = 1000  # W/m²
        efficiency = 0.2  # 20% panel efficiency
        
        capacity_factor = min(1.0, irradiance / max_irradiance) * efficiency
        return capacity_factor
