"""
Main forecasting engine that coordinates all forecasting components
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
import schedule
import time
from threading import Thread

# Import all forecasting modules
from data_collectors.entsoe_client import ENTSOEClient
from data_collectors.weather_client import WeatherClient
from data_collectors.real_data_client import RealDataClient
from forecasting.solar_forecaster import SolarForecaster
from forecasting.nuclear_monitor import NuclearMonitor
from forecasting.price_forecaster import PriceForecaster
from forecasting.flow_forecaster import FlowForecaster
from risk_management.volatility_analyzer import VolatilityAnalyzer
from risk_management.risk_manager import RiskManager
from config import Config

logger = logging.getLogger(__name__)

class ForecastingEngine:
    """Main forecasting engine for France-Germany energy markets"""
    
    def __init__(self):
        # Initialize all components
        self.entsoe_client = ENTSOEClient()
        self.weather_client = WeatherClient()
        self.real_data_client = RealDataClient()
        self.solar_forecaster = SolarForecaster()
        self.nuclear_monitor = NuclearMonitor()
        self.price_forecaster = PriceForecaster()
        self.flow_forecaster = FlowForecaster()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.risk_manager = RiskManager()
        
        # Data storage
        self.historical_data = {}
        self.forecasts = {}
        self.alerts = []
        
        # Configuration
        self.forecast_horizon = Config.FORECAST_HORIZON_HOURS
        self.update_interval = Config.UPDATE_INTERVAL_MINUTES
        
        # Coordinates for France and Germany (approximate)
        self.france_coords = (46.0, 2.0)  # (lat, lon)
        self.germany_coords = (51.0, 10.0)  # (lat, lon)
    
    def initialize(self) -> bool:
        """Initialize the forecasting engine"""
        try:
            logger.info("Initializing forecasting engine...")
            
            # Load historical data
            self._load_historical_data()
            
            # Train models
            self._train_models()
            
            logger.info("Forecasting engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing forecasting engine: {e}")
            return False
    
    def _load_historical_data(self):
        """Load historical data for model training"""
        try:
            logger.info("Loading historical data from real energy data...")
            
            # Use real data client for all data
            self.historical_data = self.real_data_client.load_historical_data()
            
            if not self.historical_data:
                logger.warning("No historical data available from real data sources")
                return
            
            logger.info("Historical data loaded successfully from real energy data")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def _train_models(self):
        """Train all forecasting models"""
        try:
            logger.info("Training forecasting models...")
            
            # Prepare training data
            training_data = self._prepare_training_data()
            
            if training_data.empty:
                logger.warning("No training data available")
                return
            
            # Train solar forecaster
            if not self.historical_data['france_weather'].empty:
                self.solar_forecaster.train(
                    self.historical_data['france_weather'], 
                    'solar_generation_potential'
                )
            
            # Train price forecaster
            if not training_data.empty:
                self.price_forecaster.train(training_data, 'price')
                self.flow_forecaster.train(training_data, 'cross_border_flow')
            
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data by merging all historical data"""
        try:
            # Start with France data
            if not self.historical_data['france_prices'].empty:
                training_data = self.historical_data['france_prices'].copy()
                training_data = training_data.rename(columns={
                    'price': 'france_price',
                    'demand': 'france_demand',
                    'nuclear_generation': 'france_nuclear',
                    'solar_generation': 'france_solar',
                    'wind_generation': 'france_wind',
                    'total_generation': 'france_total_gen'
                })
            else:
                return pd.DataFrame()
            
            # Merge Germany data with proper column names
            if not self.historical_data['germany_prices'].empty:
                germany_data = self.historical_data['germany_prices'].copy()
                germany_data = germany_data.rename(columns={
                    'price': 'germany_price',
                    'demand': 'germany_demand',
                    'nuclear_generation': 'germany_nuclear',
                    'solar_generation': 'germany_solar',
                    'wind_generation': 'germany_wind',
                    'total_generation': 'germany_total_gen'
                })
                training_data = pd.merge(training_data, germany_data, on='datetime', how='left')
            
            # Merge cross-border flows
            if not self.historical_data['cross_border_flows'].empty:
                flows = self.historical_data['cross_border_flows'].copy()
                training_data = pd.merge(training_data, flows, on='datetime', how='left')
            
            # Merge weather data
            if not self.historical_data['france_weather'].empty:
                france_weather = self.historical_data['france_weather'].copy()
                france_weather = france_weather.rename(columns={
                    'temperature': 'france_temperature',
                    'cloud_cover': 'france_cloud_cover',
                    'solar_irradiance': 'france_solar_irradiance'
                })
                training_data = pd.merge(training_data, france_weather, on='datetime', how='left')
            
            if not self.historical_data['germany_weather'].empty:
                germany_weather = self.historical_data['germany_weather'].copy()
                germany_weather = germany_weather.rename(columns={
                    'temperature': 'germany_temperature',
                    'cloud_cover': 'germany_cloud_cover',
                    'solar_irradiance': 'germany_solar_irradiance'
                })
                training_data = pd.merge(training_data, germany_weather, on='datetime', how='left')
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def generate_forecast(self) -> Dict:
        """Generate comprehensive 24-48h forecast"""
        try:
            logger.info("Generating forecast...")
            
            # Create forecast timeline
            start_time = datetime.now()
            forecast_times = pd.date_range(
                start=start_time,
                periods=self.forecast_horizon + 1,
                freq='h'
            )[1:]  # Exclude current time
            
            # Get weather forecast using real data
            weather_forecast = self.real_data_client.get_weather_forecast(self.forecast_horizon)
            france_weather_forecast = weather_forecast[['datetime', 'france_temperature', 'france_cloud_cover', 'france_solar_irradiance']].rename(columns={
                'france_temperature': 'temperature',
                'france_cloud_cover': 'cloud_cover', 
                'france_solar_irradiance': 'solar_irradiance'
            })
            germany_weather_forecast = weather_forecast[['datetime', 'germany_temperature', 'germany_cloud_cover', 'germany_solar_irradiance']].rename(columns={
                'germany_temperature': 'temperature',
                'germany_cloud_cover': 'cloud_cover',
                'germany_solar_irradiance': 'solar_irradiance'
            })
            
            # Solar generation forecast
            solar_forecast = self.solar_forecaster.predict(france_weather_forecast)
            
            # Nuclear capacity forecast
            nuclear_forecast = self.nuclear_monitor.get_nuclear_forecast(
                self.historical_data['france_prices']  # Use France data for nuclear capacity
            )
            
            # Prepare forecast data
            forecast_data = self._prepare_forecast_data(
                forecast_times, solar_forecast, nuclear_forecast
            )
            
            # Price forecast
            price_forecast = self.price_forecaster.predict(forecast_data)
            
            # Flow forecast
            flow_forecast = self.flow_forecaster.predict(forecast_data)
            
            # Volatility forecast
            volatility_forecast = self.volatility_analyzer.forecast_volatility(
                self.historical_data['france_prices']
            )
            
            # Risk assessment
            risk_assessment = self._assess_forecast_risk(forecast_data)
            
            # Generate alerts
            alerts = self._generate_forecast_alerts(
                price_forecast, flow_forecast, volatility_forecast
            )
            
            # Compile results
            forecast_results = {
                'timestamp': datetime.now(),
                'forecast_horizon_hours': self.forecast_horizon,
                'price_forecast': price_forecast,
                'flow_forecast': flow_forecast,
                'solar_forecast': solar_forecast,
                'nuclear_forecast': nuclear_forecast,
                'volatility_forecast': volatility_forecast,
                'risk_assessment': risk_assessment,
                'alerts': alerts,
                'confidence_scores': self._calculate_confidence_scores(
                    price_forecast, flow_forecast, solar_forecast
                )
            }
            
            self.forecasts = forecast_results
            logger.info("Forecast generated successfully")
            
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {'error': str(e)}
    
    def _prepare_forecast_data(self, forecast_times: pd.DatetimeIndex, 
                             solar_forecast: pd.DataFrame, 
                             nuclear_forecast: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for forecasting"""
        forecast_data = pd.DataFrame({'datetime': forecast_times})
        
        # Add solar generation forecast
        if not solar_forecast.empty and 'predicted_solar_generation' in solar_forecast.columns:
            solar_forecast = solar_forecast.set_index('datetime')
            forecast_data = forecast_data.set_index('datetime')
            forecast_data['solar_generation'] = solar_forecast['predicted_solar_generation']
            forecast_data = forecast_data.reset_index()
        else:
            # Use default solar generation
            forecast_data['solar_generation'] = forecast_data['datetime'].apply(
                lambda dt: 20 * np.sin(np.pi * (dt.hour - 6) / 12) if 6 <= dt.hour <= 18 else 0
            )
        
        # Add nuclear capacity forecast
        if not nuclear_forecast.empty and 'forecasted_capacity' in nuclear_forecast.columns:
            nuclear_forecast = nuclear_forecast.set_index('datetime')
            forecast_data = forecast_data.set_index('datetime')
            forecast_data['nuclear_generation'] = nuclear_forecast['forecasted_capacity']
            forecast_data = forecast_data.reset_index()
        else:
            # Use default nuclear generation
            forecast_data['nuclear_generation'] = 60  # Default nuclear capacity
        
        # Add default values for missing data
        forecast_data['demand'] = forecast_data['datetime'].apply(self._estimate_demand)
        forecast_data['wind_generation'] = forecast_data['datetime'].apply(self._estimate_wind)
        forecast_data['temperature'] = 15  # Default temperature
        forecast_data['cloud_cover'] = 50  # Default cloud cover
        
        # Add price data for forecasting
        forecast_data['price'] = forecast_data['datetime'].apply(self._estimate_price)
        
        return forecast_data
    
    def _estimate_demand(self, dt: datetime) -> float:
        """Estimate demand based on time of day and day of week"""
        hour = dt.hour
        day_of_week = dt.weekday()
        
        # Base demand (GW)
        base_demand = 50
        
        # Hourly pattern
        if 6 <= hour <= 8:  # Morning peak
            hour_factor = 1.2
        elif 18 <= hour <= 20:  # Evening peak
            hour_factor = 1.3
        elif 0 <= hour <= 6:  # Night
            hour_factor = 0.7
        else:  # Day
            hour_factor = 1.0
        
        # Weekly pattern
        if day_of_week < 5:  # Weekday
            day_factor = 1.0
        else:  # Weekend
            day_factor = 0.9
        
        return base_demand * hour_factor * day_factor
    
    def _estimate_wind(self, dt: datetime) -> float:
        """Estimate wind generation (simplified)"""
        # This is a placeholder - in practice, you'd use weather data
        return 10  # GW
    
    def _estimate_price(self, dt: datetime) -> float:
        """Estimate price based on time of day and day of week"""
        hour = dt.hour
        day_of_week = dt.weekday()
        
        # Base price
        base_price = 50
        
        # Hourly pattern
        if 8 <= hour <= 10 or 18 <= hour <= 20:  # Peak hours
            hour_factor = 1.5
        elif 0 <= hour <= 6:  # Night hours
            hour_factor = 0.8
        else:  # Day hours
            hour_factor = 1.0
        
        # Weekly pattern
        if day_of_week < 5:  # Weekday
            day_factor = 1.0
        else:  # Weekend
            day_factor = 0.9
        
        # Add some randomness
        noise = np.random.normal(0, 5)
        
        return base_price * hour_factor * day_factor + noise
    
    def _assess_forecast_risk(self, forecast_data: pd.DataFrame) -> Dict:
        """Assess risk in the forecast"""
        try:
            # Calculate basic risk metrics
            risk_metrics = {
                'price_volatility_forecast': forecast_data['price'].std() if 'price' in forecast_data.columns else 0,
                'flow_volatility_forecast': forecast_data['cross_border_flow'].std() if 'cross_border_flow' in forecast_data.columns else 0,
                'solar_uncertainty': forecast_data['solar_generation'].std() if 'solar_generation' in forecast_data.columns else 0,
                'nuclear_uncertainty': forecast_data['nuclear_generation'].std() if 'nuclear_generation' in forecast_data.columns else 0
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing forecast risk: {e}")
            return {}
    
    def _generate_forecast_alerts(self, price_forecast: pd.DataFrame, 
                                flow_forecast: pd.DataFrame, 
                                volatility_forecast: pd.DataFrame) -> List[Dict]:
        """Generate alerts based on forecast"""
        alerts = []
        
        # Price volatility alerts
        if not price_forecast.empty and 'predicted_volatility' in price_forecast.columns:
            max_volatility = price_forecast['predicted_volatility'].max()
            if max_volatility > Config.HIGH_VOLATILITY_THRESHOLD:
                alerts.append({
                    'type': 'high_volatility_forecast',
                    'severity': 'high',
                    'message': f"High volatility forecasted: {max_volatility:.1%}",
                    'forecast_time': price_forecast['predicted_volatility'].idxmax()
                })
        
        # Solar peak alerts
        if not price_forecast.empty:
            solar_peaks = price_forecast[
                price_forecast['datetime'].dt.hour.isin(Config.SOLAR_PEAK_HOURS)
            ]
            if not solar_peaks.empty:
                alerts.append({
                    'type': 'solar_peak_forecast',
                    'severity': 'medium',
                    'message': f"Solar generation peaks forecasted for {len(solar_peaks)} hours",
                    'forecast_time': solar_peaks['datetime'].iloc[0]
                })
        
        return alerts
    
    def _calculate_confidence_scores(self, price_forecast: pd.DataFrame, 
                                   flow_forecast: pd.DataFrame, 
                                   solar_forecast: pd.DataFrame) -> Dict:
        """Calculate confidence scores for forecasts"""
        confidence_scores = {}
        
        if not price_forecast.empty and 'price_confidence' in price_forecast.columns:
            confidence_scores['price_forecast'] = price_forecast['price_confidence'].mean()
        
        if not flow_forecast.empty and 'flow_confidence' in flow_forecast.columns:
            confidence_scores['flow_forecast'] = flow_forecast['flow_confidence'].mean()
        
        if not solar_forecast.empty and 'solar_confidence' in solar_forecast.columns:
            confidence_scores['solar_forecast'] = solar_forecast['solar_confidence'].mean()
        
        return confidence_scores
    
    def start_monitoring(self):
        """Start the monitoring and forecasting loop"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        # Schedule regular updates
        schedule.every(self.update_interval).minutes.do(self._update_forecast)
        
        # Start scheduler in background thread
        scheduler_thread = Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info(f"Monitoring started with {self.update_interval} minute intervals")
    
    def _update_forecast(self):
        """Update forecast with latest data"""
        try:
            logger.info("Updating forecast...")
            
            # Generate new forecast
            forecast = self.generate_forecast()
            
            # Store forecast
            self.forecasts = forecast
            
            # Generate alerts
            if 'alerts' in forecast:
                self.alerts.extend(forecast['alerts'])
            
            logger.info("Forecast updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating forecast: {e}")
    
    def get_latest_forecast(self) -> Dict:
        """Get the latest forecast"""
        return self.forecasts
    
    def get_alerts(self) -> List[Dict]:
        """Get all alerts"""
        return self.alerts
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
