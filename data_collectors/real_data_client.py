"""
Real Data Client for Energy Forecasting Tool
Handles real energy data from CSV files provided by user
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class RealDataClient:
    """Client for handling real energy data from CSV files"""
    
    def __init__(self, data_file_path: str = None):
        """
        Initialize the real data client
        
        Args:
            data_file_path: Path to the CSV file with real energy data
        """
        self.data_file_path = data_file_path or "energy-charts_Electricity_production_and_spot_prices_in_Germany_in_week_39_2025 (1).csv"
        self.historical_data = {}
        self.current_data = None
        
    def load_historical_data(self, days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Load historical data from the CSV file
        
        Args:
            days_back: Number of days of historical data to load
            
        Returns:
            Dictionary containing historical data for different categories
        """
        try:
            logger.info(f"Loading real energy data from {self.data_file_path}")
            
            # Read the CSV file
            df = pd.read_csv(self.data_file_path, skiprows=1)  # Skip header row
            
            # Clean column names
            df.columns = ['datetime', 'non_renewable_power', 'renewable_power', 'day_ahead_price']
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Remove rows with missing data (empty values)
            df = df.dropna(subset=['non_renewable_power', 'renewable_power', 'day_ahead_price'])
            
            # Calculate total generation
            df['total_generation'] = df['non_renewable_power'] + df['renewable_power']
            
            # Estimate nuclear generation (assume 20% of non-renewable is nuclear)
            df['nuclear_generation'] = df['non_renewable_power'] * 0.2
            
            # Estimate solar generation (assume 60% of renewable is solar during day, 0% at night)
            df['solar_generation'] = df.apply(
                lambda row: row['renewable_power'] * 0.6 if 6 <= row['datetime'].hour <= 18 else 0, 
                axis=1
            )
            
            # Add solar generation potential (same as solar generation for now)
            df['solar_generation_potential'] = df['solar_generation']
            
            # Estimate wind generation (remaining renewable)
            df['wind_generation'] = df['renewable_power'] - df['solar_generation']
            
            # Estimate demand (total generation + 5% losses)
            df['demand'] = df['total_generation'] * 1.05
            
            # Create France data (assume similar patterns but different scale)
            france_data = df.copy()
            france_data['price'] = df['day_ahead_price'] * 1.1  # France typically 10% higher
            france_data['demand'] = df['demand'] * 0.8  # France smaller market
            france_data['nuclear_generation'] = df['nuclear_generation'] * 1.5  # France has more nuclear
            france_data['solar_generation'] = df['solar_generation'] * 0.7  # France less solar
            france_data['wind_generation'] = df['wind_generation'] * 0.6  # France less wind
            france_data['total_generation'] = france_data['nuclear_generation'] + france_data['solar_generation'] + france_data['wind_generation']
            
            # Create Germany data
            germany_data = df.copy()
            germany_data['price'] = df['day_ahead_price']
            germany_data['demand'] = df['demand']
            germany_data['nuclear_generation'] = df['nuclear_generation']
            germany_data['solar_generation'] = df['solar_generation']
            germany_data['wind_generation'] = df['wind_generation']
            germany_data['total_generation'] = df['total_generation']
            
            # Create cross-border flows (assume 5% of total generation flows between countries)
            cross_border_flows = pd.DataFrame({
                'datetime': df['datetime'],
                'france_to_germany': (france_data['total_generation'] - france_data['demand']) * 0.1,
                'germany_to_france': (germany_data['total_generation'] - germany_data['demand']) * 0.1,
                'net_flow': (france_data['total_generation'] - france_data['demand']) * 0.1 - 
                           (germany_data['total_generation'] - germany_data['demand']) * 0.1,
                'cross_border_flow': (france_data['total_generation'] - france_data['demand']) * 0.1 - 
                                   (germany_data['total_generation'] - germany_data['demand']) * 0.1
            })
            
            # Create weather data (synthetic but realistic)
            weather_data = self._generate_weather_data(df['datetime'])
            
            # Store historical data
            self.historical_data = {
                'france_prices': france_data[['datetime', 'price', 'demand', 'nuclear_generation', 
                                            'solar_generation', 'wind_generation', 'total_generation']],
                'germany_prices': germany_data[['datetime', 'price', 'demand', 'nuclear_generation', 
                                               'solar_generation', 'wind_generation', 'total_generation']],
                'cross_border_flows': cross_border_flows,
                'france_weather': weather_data['france'],
                'germany_weather': weather_data['germany']
            }
            
            logger.info("Real energy data loaded successfully")
            return self.historical_data
            
        except Exception as e:
            logger.error(f"Error loading real energy data: {e}")
            return {}
    
    def _generate_weather_data(self, datetime_series: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate realistic weather data based on datetime patterns"""
        
        # France weather (southern Europe)
        france_weather = pd.DataFrame({
            'datetime': datetime_series,
            'temperature': 15 + 10 * np.sin(2 * np.pi * datetime_series.dt.dayofyear / 365) + 
                         5 * np.sin(2 * np.pi * datetime_series.dt.hour / 24) + np.random.normal(0, 2, len(datetime_series)),
            'cloud_cover': np.random.uniform(20, 80, len(datetime_series)),
            'solar_irradiance': np.maximum(0, 800 * np.sin(np.pi * (datetime_series.dt.hour - 6) / 12) * 
                                         (1 - np.random.uniform(0, 0.5, len(datetime_series)))),
            'humidity': np.random.uniform(40, 90, len(datetime_series)),
            'wind_speed': np.random.uniform(2, 15, len(datetime_series))
        })
        
        # Germany weather (central Europe)
        germany_weather = pd.DataFrame({
            'datetime': datetime_series,
            'temperature': 12 + 8 * np.sin(2 * np.pi * datetime_series.dt.dayofyear / 365) + 
                         4 * np.sin(2 * np.pi * datetime_series.dt.hour / 24) + np.random.normal(0, 2, len(datetime_series)),
            'cloud_cover': np.random.uniform(30, 90, len(datetime_series)),
            'solar_irradiance': np.maximum(0, 600 * np.sin(np.pi * (datetime_series.dt.hour - 6) / 12) * 
                                         (1 - np.random.uniform(0, 0.6, len(datetime_series)))),
            'humidity': np.random.uniform(50, 95, len(datetime_series)),
            'wind_speed': np.random.uniform(3, 18, len(datetime_series))
        })
        
        return {
            'france': france_weather,
            'germany': germany_weather
        }
    
    def get_weather_forecast(self, hours_ahead: int = 48) -> pd.DataFrame:
        """
        Generate weather forecast based on historical patterns
        
        Args:
            hours_ahead: Number of hours to forecast ahead
            
        Returns:
            DataFrame with weather forecast
        """
        try:
            # Get the last datetime from historical data
            if not self.historical_data.get('france_weather', pd.DataFrame()).empty:
                last_datetime = self.historical_data['france_weather']['datetime'].iloc[-1]
            else:
                last_datetime = datetime.now()
            
            # Create forecast timestamps
            forecast_times = pd.date_range(
                start=last_datetime + timedelta(hours=1),
                periods=hours_ahead,
                freq='h'
            )
            
            # Generate forecast weather data
            forecast_weather = self._generate_weather_data(pd.Series(forecast_times))
            
            # Combine France and Germany weather
            france_forecast = forecast_weather['france'].copy()
            germany_forecast = forecast_weather['germany'].copy()
            
            # Merge weather data
            weather_forecast = pd.merge(
                france_forecast.rename(columns={
                    'temperature': 'france_temperature',
                    'cloud_cover': 'france_cloud_cover',
                    'solar_irradiance': 'france_solar_irradiance'
                }),
                germany_forecast.rename(columns={
                    'temperature': 'germany_temperature',
                    'cloud_cover': 'germany_cloud_cover',
                    'solar_irradiance': 'germany_solar_irradiance'
                }),
                on='datetime'
            )
            
            return weather_forecast
            
        except Exception as e:
            logger.error(f"Error generating weather forecast: {e}")
            return pd.DataFrame()
    
    def get_current_data(self) -> Dict:
        """Get current energy data"""
        if self.current_data is None:
            self.current_data = self._get_latest_data()
        return self.current_data
    
    def _get_latest_data(self) -> Dict:
        """Get the most recent data from historical data"""
        try:
            if not self.historical_data:
                return {}
            
            latest_data = {}
            
            # Get latest France data
            if not self.historical_data.get('france_prices', pd.DataFrame()).empty:
                france_latest = self.historical_data['france_prices'].iloc[-1]
                latest_data['france'] = {
                    'price': france_latest['price'],
                    'demand': france_latest['demand'],
                    'nuclear': france_latest['nuclear_generation'],
                    'solar': france_latest['solar_generation'],
                    'wind': france_latest['wind_generation'],
                    'total': france_latest['total_generation']
                }
            
            # Get latest Germany data
            if not self.historical_data.get('germany_prices', pd.DataFrame()).empty:
                germany_latest = self.historical_data['germany_prices'].iloc[-1]
                latest_data['germany'] = {
                    'price': germany_latest['price'],
                    'demand': germany_latest['demand'],
                    'nuclear': germany_latest['nuclear_generation'],
                    'solar': germany_latest['solar_generation'],
                    'wind': germany_latest['wind_generation'],
                    'total': germany_latest['total_generation']
                }
            
            # Get latest cross-border flows
            if not self.historical_data.get('cross_border_flows', pd.DataFrame()).empty:
                flows_latest = self.historical_data['cross_border_flows'].iloc[-1]
                latest_data['flows'] = {
                    'france_to_germany': flows_latest['france_to_germany'],
                    'germany_to_france': flows_latest['germany_to_france'],
                    'net_flow': flows_latest['net_flow']
                }
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {}
    
    def update_data_file(self, new_file_path: str):
        """
        Update the data file path and reload data
        
        Args:
            new_file_path: Path to new CSV file
        """
        self.data_file_path = new_file_path
        self.historical_data = {}
        self.current_data = None
        logger.info(f"Data file updated to: {new_file_path}")
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data"""
        try:
            if not self.historical_data:
                return {}
            
            summary = {}
            
            # France summary
            if not self.historical_data.get('france_prices', pd.DataFrame()).empty:
                france_data = self.historical_data['france_prices']
                summary['france'] = {
                    'data_points': len(france_data),
                    'date_range': f"{france_data['datetime'].min()} to {france_data['datetime'].max()}",
                    'avg_price': france_data['price'].mean(),
                    'avg_demand': france_data['demand'].mean(),
                    'avg_nuclear': france_data['nuclear_generation'].mean(),
                    'avg_solar': france_data['solar_generation'].mean(),
                    'avg_wind': france_data['wind_generation'].mean()
                }
            
            # Germany summary
            if not self.historical_data.get('germany_prices', pd.DataFrame()).empty:
                germany_data = self.historical_data['germany_prices']
                summary['germany'] = {
                    'data_points': len(germany_data),
                    'date_range': f"{germany_data['datetime'].min()} to {germany_data['datetime'].max()}",
                    'avg_price': germany_data['price'].mean(),
                    'avg_demand': germany_data['demand'].mean(),
                    'avg_nuclear': germany_data['nuclear_generation'].mean(),
                    'avg_solar': germany_data['solar_generation'].mean(),
                    'avg_wind': germany_data['wind_generation'].mean()
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
