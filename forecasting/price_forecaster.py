"""
Day-ahead price forecasting module
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class PriceForecaster:
    """Day-ahead electricity price forecasting"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'demand', 'solar_generation', 'wind_generation', 'nuclear_generation',
            'cross_border_flow', 'temperature', 'cloud_cover'
        ]
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for price forecasting"""
        df = data.copy()
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (previous hours)
        for lag in [1, 2, 3, 24, 48]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'price_rolling_mean_{window}'] = df['price'].rolling(window).mean()
            df[f'price_rolling_std_{window}'] = df['price'].rolling(window).std()
            df[f'demand_rolling_mean_{window}'] = df['demand'].rolling(window).mean()
        
        # Price volatility
        df['price_volatility'] = df['price'].rolling(24).std()
        
        # Generation mix ratios
        total_generation = df['solar_generation'] + df['wind_generation'] + df['nuclear_generation']
        df['solar_ratio'] = df['solar_generation'] / total_generation
        df['wind_ratio'] = df['wind_generation'] / total_generation
        df['nuclear_ratio'] = df['nuclear_generation'] / total_generation
        
        # Cross-border flow impact
        df['flow_impact'] = df['cross_border_flow'] / df['demand']
        
        return df
    
    def train(self, historical_data: pd.DataFrame, target_column: str = 'price'):
        """Train the price forecasting models"""
        try:
            # Prepare features
            features_df = self.prepare_features(historical_data)
            
            # Select features
            feature_cols = [col for col in features_df.columns 
                          if col not in ['datetime', target_column]]
            X = features_df[feature_cols]
            y = features_df[target_column]
            
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                logger.warning("No valid training data available")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train all models
            for name, model in self.models.items():
                model.fit(X_scaled, y)
                logger.info(f"Trained {name} model with {len(X)} samples")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training price forecasting models: {e}")
            return False
    
    def predict(self, forecast_data: pd.DataFrame) -> pd.DataFrame:
        """Predict day-ahead prices"""
        if not self.is_trained:
            logger.warning("Models not trained. Using simple heuristic prediction.")
            return self._heuristic_prediction(forecast_data)
        
        try:
            # Prepare features
            features_df = self.prepare_features(forecast_data)
            
            # Select features
            feature_cols = [col for col in features_df.columns 
                          if col not in ['datetime', 'price']]
            X = features_df[feature_cols]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions with all models
            predictions = {}
            for name, model in self.models.items():
                predictions[f'{name}_prediction'] = model.predict(X_scaled)
            
            # Ensemble prediction (weighted average)
            ensemble_prediction = (
                0.4 * predictions['rf_prediction'] +
                0.4 * predictions['gb_prediction'] +
                0.2 * predictions['lr_prediction']
            )
            
            # Create result DataFrame
            result_df = forecast_data.copy()
            result_df['predicted_price'] = ensemble_prediction
            result_df['rf_prediction'] = predictions['rf_prediction']
            result_df['gb_prediction'] = predictions['gb_prediction']
            result_df['lr_prediction'] = predictions['lr_prediction']
            
            # Calculate prediction confidence
            result_df['price_confidence'] = self._calculate_confidence(features_df)
            
            # Calculate volatility forecast
            result_df['predicted_volatility'] = self._predict_volatility(features_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making price predictions: {e}")
            return self._heuristic_prediction(forecast_data)
    
    def _heuristic_prediction(self, forecast_data: pd.DataFrame) -> pd.DataFrame:
        """Simple heuristic-based price prediction when models are not trained"""
        result_df = forecast_data.copy()
        
        # Simple heuristic based on demand and generation
        result_df['predicted_price'] = result_df.apply(
            lambda row: self._simple_price_heuristic(row), axis=1
        )
        result_df['price_confidence'] = 0.3  # Low confidence for heuristic
        result_df['predicted_volatility'] = 0.1  # Default volatility
        
        return result_df
    
    def _simple_price_heuristic(self, row) -> float:
        """Simple heuristic for price prediction"""
        # Base price on demand and generation balance
        demand = row.get('demand', 0)
        solar = row.get('solar_generation', 0)
        wind = row.get('wind_generation', 0)
        nuclear = row.get('nuclear_generation', 0)
        
        total_generation = solar + wind + nuclear
        supply_demand_ratio = total_generation / demand if demand > 0 else 1
        
        # Base price
        base_price = 50  # €/MWh
        
        # Adjust based on supply-demand balance
        if supply_demand_ratio < 0.8:  # High demand, low supply
            price_multiplier = 2.0
        elif supply_demand_ratio < 1.0:  # Moderate imbalance
            price_multiplier = 1.5
        else:  # Good supply
            price_multiplier = 0.8
        
        # Time of day adjustment
        hour = row['datetime'].hour
        if 8 <= hour <= 10 or 18 <= hour <= 20:  # Peak hours
            price_multiplier *= 1.3
        elif 0 <= hour <= 6:  # Night hours
            price_multiplier *= 0.7
        
        return base_price * price_multiplier
    
    def _calculate_confidence(self, features_df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on feature quality"""
        confidence = 1.0
        
        # Reduce confidence for missing data
        missing_data_ratio = features_df.isna().sum().sum() / (len(features_df) * len(features_df.columns))
        confidence *= (1 - missing_data_ratio)
        
        # Reduce confidence for extreme values
        if 'demand' in features_df.columns:
            demand_std = features_df['demand'].std()
            if demand_std > features_df['demand'].mean() * 0.5:  # High demand volatility
                confidence *= 0.8
        
        return max(0.1, min(1.0, confidence))
    
    def _predict_volatility(self, features_df: pd.DataFrame) -> float:
        """Predict price volatility"""
        # Simple volatility prediction based on historical patterns
        if 'price_volatility' in features_df.columns:
            return features_df['price_volatility'].mean()
        else:
            return 0.1  # Default volatility
    
    def evaluate_models(self, test_data: pd.DataFrame, target_column: str = 'price') -> Dict:
        """Evaluate all models on test data"""
        if not self.is_trained:
            return {'error': 'Models not trained'}
        
        try:
            # Prepare features
            features_df = self.prepare_features(test_data)
            
            # Select features and target
            feature_cols = [col for col in features_df.columns 
                          if col not in ['datetime', target_column]]
            X = features_df[feature_cols]
            y = features_df[target_column]
            
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return {'error': 'No valid test data'}
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Evaluate each model
            results = {}
            for name, model in self.models.items():
                predictions = model.predict(X_scaled)
                
                results[name] = {
                    'mae': mean_absolute_error(y, predictions),
                    'mse': mean_squared_error(y, predictions),
                    'rmse': np.sqrt(mean_squared_error(y, predictions)),
                    'r2': r2_score(y, predictions)
                }
            
            # Ensemble evaluation
            ensemble_pred = (
                0.4 * self.models['rf'].predict(X_scaled) +
                0.4 * self.models['gb'].predict(X_scaled) +
                0.2 * self.models['lr'].predict(X_scaled)
            )
            
            results['ensemble'] = {
                'mae': mean_absolute_error(y, ensemble_pred),
                'mse': mean_squared_error(y, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
                'r2': r2_score(y, ensemble_pred)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return {'error': str(e)}
