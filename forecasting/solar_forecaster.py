"""
Solar generation forecasting module
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class SolarForecaster:
    """Solar generation forecasting using weather data and historical patterns"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = [
            'hour', 'day_of_year', 'cloud_cover', 'temperature', 
            'humidity', 'wind_speed', 'solar_irradiance'
        ]
    
    def prepare_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for solar forecasting"""
        df = weather_df.copy()
        
        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Add solar position features
        df['solar_elevation'] = df.apply(self._calculate_solar_elevation, axis=1)
        df['solar_azimuth'] = df.apply(self._calculate_solar_azimuth, axis=1)
        
        # Weather features
        df['cloud_cover'] = df['cloud_cover'] / 100.0  # Normalize to 0-1
        df['temperature'] = df['temperature']
        df['humidity'] = df['humidity'] / 100.0  # Normalize to 0-1
        df['wind_speed'] = df['wind_speed']
        
        return df
    
    def _calculate_solar_elevation(self, row) -> float:
        """Calculate solar elevation angle"""
        # Simplified solar position calculation
        # In practice, you'd use more accurate astronomical calculations
        
        hour = row['hour']
        day_of_year = row['day_of_year']
        
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation (simplified)
        elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(50)) +  # Assuming 50° latitude
            np.cos(np.radians(declination)) * np.cos(np.radians(50)) * np.cos(np.radians(hour_angle))
        )
        
        return np.degrees(elevation)
    
    def _calculate_solar_azimuth(self, row) -> float:
        """Calculate solar azimuth angle"""
        # Simplified azimuth calculation
        hour = row['hour']
        day_of_year = row['day_of_year']
        
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar azimuth (simplified)
        azimuth = np.arctan2(
            np.sin(np.radians(hour_angle)),
            np.cos(np.radians(hour_angle)) * np.sin(np.radians(50)) - 
            np.tan(np.radians(declination)) * np.cos(np.radians(50))
        )
        
        return np.degrees(azimuth)
    
    def train(self, historical_data: pd.DataFrame, target_column: str = 'solar_generation'):
        """Train the solar forecasting model"""
        try:
            # Prepare features
            features_df = self.prepare_features(historical_data)
            
            # Select features and target
            X = features_df[self.feature_columns + ['solar_elevation', 'solar_azimuth']]
            y = features_df[target_column]
            
            # Remove any rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                logger.warning("No valid training data available")
                return False
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            
            logger.info(f"Solar forecasting model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training solar forecasting model: {e}")
            return False
    
    def predict(self, weather_forecast: pd.DataFrame) -> pd.DataFrame:
        """Predict solar generation for given weather forecast"""
        if not self.is_trained:
            logger.warning("Model not trained. Using simple heuristic prediction.")
            return self._heuristic_prediction(weather_forecast)
        
        try:
            # Prepare features
            features_df = self.prepare_features(weather_forecast)
            
            # Select features
            X = features_df[self.feature_columns + ['solar_elevation', 'solar_azimuth']]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Create result DataFrame
            result_df = weather_forecast.copy()
            result_df['predicted_solar_generation'] = predictions
            result_df['solar_confidence'] = self._calculate_confidence(features_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making solar predictions: {e}")
            return self._heuristic_prediction(weather_forecast)
    
    def _heuristic_prediction(self, weather_forecast: pd.DataFrame) -> pd.DataFrame:
        """Simple heuristic-based solar prediction when model is not trained"""
        if weather_forecast.empty:
            # Create empty result with proper structure
            result_df = pd.DataFrame({
                'datetime': pd.date_range(start=datetime.now(), periods=48, freq='h'),
                'predicted_solar_generation': 0,
                'solar_confidence': 0.5
            })
        else:
            result_df = weather_forecast.copy()
            
            # Simple heuristic: solar generation based on time of day and cloud cover
            result_df['predicted_solar_generation'] = result_df.apply(
                lambda row: self._simple_solar_heuristic(row), axis=1
            )
            result_df['solar_confidence'] = 0.5  # Low confidence for heuristic
        
        return result_df
    
    def _simple_solar_heuristic(self, row) -> float:
        """Simple heuristic for solar generation prediction"""
        hour = row['datetime'].hour
        
        # Solar generation is highest during midday
        if 6 <= hour <= 18:
            # Base generation based on time of day
            time_factor = np.sin(np.pi * (hour - 6) / 12)
            
            # Reduce based on cloud cover (if available)
            cloud_cover = row.get('cloud_cover', 50)  # Default 50% cloud cover
            cloud_factor = 1 - (cloud_cover / 100.0) * 0.7
            
            return time_factor * cloud_factor * 20  # Scale to GW
        else:
            return 0.0
    
    def _calculate_confidence(self, features_df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on feature quality"""
        # Simple confidence calculation based on data completeness
        confidence = 1.0
        
        # Reduce confidence for extreme weather conditions
        if 'cloud_cover' in features_df.columns:
            avg_cloud_cover = features_df['cloud_cover'].mean()
            if avg_cloud_cover > 0.8:  # Very cloudy
                confidence *= 0.7
            elif avg_cloud_cover < 0.2:  # Very clear
                confidence *= 0.9
        
        return max(0.1, min(1.0, confidence))
    
    def evaluate_model(self, test_data: pd.DataFrame, target_column: str = 'solar_generation') -> Dict:
        """Evaluate model performance on test data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Prepare features
            features_df = self.prepare_features(test_data)
            
            # Select features and target
            X = features_df[self.feature_columns + ['solar_elevation', 'solar_azimuth']]
            y = features_df[target_column]
            
            # Remove any rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return {'error': 'No valid test data'}
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate R² score
            r2 = self.model.score(X, y)
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'error': str(e)}
