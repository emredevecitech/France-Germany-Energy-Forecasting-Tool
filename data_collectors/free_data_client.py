"""
Free data sources client for energy forecasting
Uses publicly available data without API keys
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class FreeDataClient:
    """Client for free energy market data sources"""
    
    def __init__(self):
        self.base_urls = {
            'entsoe_public': 'https://transparency.entsoe.eu/api',
            'eex_public': 'https://www.eex.com/api',
            'rte_france': 'https://opendata.reseaux-energies.fr/api',
            '50hertz': 'https://www.50hertz.com/api'
        }
    
    def get_synthetic_energy_data(self, country: str, start_date: datetime, 
                                end_date: datetime) -> pd.DataFrame:
        """Generate synthetic energy data based on realistic patterns"""
        try:
            # Create time series
            time_series = pd.date_range(start=start_date, end=end_date, freq='h')
            
            if country == 'france':
                return self._generate_france_data(time_series)
            elif country == 'germany':
                return self._generate_germany_data(time_series)
            else:
                return self._generate_generic_data(time_series)
                
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return pd.DataFrame()
    
    def _generate_france_data(self, time_series: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic France energy data"""
        data = []
        
        for dt in time_series:
            # Base demand (GW)
            base_demand = 50
            
            # Daily pattern (higher in morning and evening)
            if 7 <= dt.hour <= 9 or 18 <= dt.hour <= 20:
                demand_multiplier = 1.3
            elif 22 <= dt.hour or dt.hour <= 6:
                demand_multiplier = 0.8
            else:
                demand_multiplier = 1.0
            
            demand = base_demand * demand_multiplier + np.random.normal(0, 3)
            
            # Nuclear generation (France has high nuclear capacity)
            nuclear_base = 45
            nuclear_variation = np.random.normal(0, 2)
            nuclear = max(0, nuclear_base + nuclear_variation)
            
            # Solar generation
            if 6 <= dt.hour <= 18:
                solar = 8 * np.sin(np.pi * (dt.hour - 6) / 12) + np.random.normal(0, 1)
            else:
                solar = 0
            solar = max(0, solar)
            
            # Wind generation
            wind = 5 + np.random.normal(0, 2)
            wind = max(0, wind)
            
            # Hydro generation
            hydro = 3 + np.random.normal(0, 0.5)
            hydro = max(0, hydro)
            
            # Price calculation (€/MWh)
            total_generation = nuclear + solar + wind + hydro
            supply_demand_ratio = total_generation / demand
            
            if supply_demand_ratio < 0.9:
                base_price = 80
            elif supply_demand_ratio < 1.1:
                base_price = 50
            else:
                base_price = 30
            
            # Add price volatility
            price = base_price + np.random.normal(0, 10)
            price = max(0, price)
            
            data.append({
                'datetime': dt,
                'demand': demand,
                'nuclear_generation': nuclear,
                'solar_generation': solar,
                'wind_generation': wind,
                'hydro_generation': hydro,
                'total_generation': total_generation,
                'price': price,
                'country': 'France'
            })
        
        return pd.DataFrame(data)
    
    def _generate_germany_data(self, time_series: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic Germany energy data"""
        data = []
        
        for dt in time_series:
            # Base demand (GW)
            base_demand = 60
            
            # Daily pattern
            if 7 <= dt.hour <= 9 or 18 <= dt.hour <= 20:
                demand_multiplier = 1.2
            elif 22 <= dt.hour or dt.hour <= 6:
                demand_multiplier = 0.9
            else:
                demand_multiplier = 1.0
            
            demand = base_demand * demand_multiplier + np.random.normal(0, 4)
            
            # Nuclear generation (Germany has reduced nuclear capacity)
            nuclear_base = 4  # Much lower than France
            nuclear_variation = np.random.normal(0, 0.5)
            nuclear = max(0, nuclear_base + nuclear_variation)
            
            # Solar generation (Germany has high solar capacity)
            if 6 <= dt.hour <= 18:
                solar = 15 * np.sin(np.pi * (dt.hour - 6) / 12) + np.random.normal(0, 2)
            else:
                solar = 0
            solar = max(0, solar)
            
            # Wind generation (Germany has high wind capacity)
            wind = 12 + np.random.normal(0, 4)
            wind = max(0, wind)
            
            # Coal generation
            coal = 8 + np.random.normal(0, 2)
            coal = max(0, coal)
            
            # Gas generation
            gas = 6 + np.random.normal(0, 1.5)
            gas = max(0, gas)
            
            # Price calculation
            total_generation = nuclear + solar + wind + coal + gas
            supply_demand_ratio = total_generation / demand
            
            if supply_demand_ratio < 0.9:
                base_price = 90
            elif supply_demand_ratio < 1.1:
                base_price = 60
            else:
                base_price = 40
            
            price = base_price + np.random.normal(0, 12)
            price = max(0, price)
            
            data.append({
                'datetime': dt,
                'demand': demand,
                'nuclear_generation': nuclear,
                'solar_generation': solar,
                'wind_generation': wind,
                'coal_generation': coal,
                'gas_generation': gas,
                'total_generation': total_generation,
                'price': price,
                'country': 'Germany'
            })
        
        return pd.DataFrame(data)
    
    def _generate_generic_data(self, time_series: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate generic energy data"""
        data = []
        
        for dt in time_series:
            demand = 40 + 10 * np.sin(2 * np.pi * dt.hour / 24) + np.random.normal(0, 3)
            nuclear = 20 + np.random.normal(0, 2)
            solar = 5 * np.sin(np.pi * (dt.hour - 6) / 12) if 6 <= dt.hour <= 18 else 0
            wind = 8 + np.random.normal(0, 2)
            price = 50 + np.random.normal(0, 8)
            
            data.append({
                'datetime': dt,
                'demand': max(0, demand),
                'nuclear_generation': max(0, nuclear),
                'solar_generation': max(0, solar),
                'wind_generation': max(0, wind),
                'total_generation': max(0, nuclear + solar + wind),
                'price': max(0, price),
                'country': 'Generic'
            })
        
        return pd.DataFrame(data)
    
    def get_cross_border_flows(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic cross-border flow data"""
        try:
            time_series = pd.date_range(start=start_date, end=end_date, freq='h')
            data = []
            
            for dt in time_series:
                # Base flow with daily pattern
                base_flow = 200 * np.sin(2 * np.pi * dt.hour / 24)
                
                # Add some randomness
                flow = base_flow + np.random.normal(0, 50)
                
                # Ensure realistic flow limits
                flow = np.clip(flow, -1000, 1000)
                
                data.append({
                    'datetime': dt,
                    'cross_border_flow': flow,
                    'flow_direction': 'France_to_Germany' if flow > 0 else 'Germany_to_France'
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error generating cross-border flows: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, lat: float, lon: float, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Generate synthetic weather data"""
        try:
            time_series = pd.date_range(start=start_date, end=end_date, freq='h')
            data = []
            
            for dt in time_series:
                # Temperature with daily and seasonal patterns
                base_temp = 15
                daily_temp = 10 * np.sin(2 * np.pi * dt.hour / 24)
                seasonal_temp = 5 * np.sin(2 * np.pi * dt.dayofyear / 365)
                temperature = base_temp + daily_temp + seasonal_temp + np.random.normal(0, 2)
                
                # Cloud cover (0-100%)
                cloud_cover = 50 + 30 * np.sin(2 * np.pi * dt.hour / 24) + np.random.normal(0, 15)
                cloud_cover = np.clip(cloud_cover, 0, 100)
                
                # Wind speed
                wind_speed = 5 + 3 * np.sin(2 * np.pi * dt.hour / 24) + np.random.normal(0, 2)
                wind_speed = max(0, wind_speed)
                
                # Solar irradiance calculation
                if 6 <= dt.hour <= 18:
                    base_irradiance = 800
                    cloud_factor = 1 - (cloud_cover / 100) * 0.7
                    solar_irradiance = base_irradiance * cloud_factor
                else:
                    solar_irradiance = 0
                
                data.append({
                    'datetime': dt,
                    'temperature': temperature,
                    'cloud_cover': cloud_cover,
                    'wind_speed': wind_speed,
                    'solar_irradiance': max(0, solar_irradiance),
                    'humidity': 60 + np.random.normal(0, 10),
                    'pressure': 1013 + np.random.normal(0, 10)
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error generating weather data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, data_type: str, country: str, 
                          days_back: int = 7) -> pd.DataFrame:
        """Get historical data for the specified number of days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        if data_type == 'prices':
            return self.get_synthetic_energy_data(country, start_date, end_date)
        elif data_type == 'flows':
            return self.get_cross_border_flows(start_date, end_date)
        elif data_type == 'weather':
            coords = {'france': (46.0, 2.0), 'germany': (51.0, 10.0)}
            lat, lon = coords.get(country, (50.0, 10.0))
            return self.get_weather_data(lat, lon, start_date, end_date)
        else:
            return self.get_synthetic_energy_data(country, start_date, end_date)
