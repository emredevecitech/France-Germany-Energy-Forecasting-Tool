"""
Cross-border flow forecasting module
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

class FlowForecaster:
    """Cross-border flow forecasting between France and Germany"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'france_demand', 'germany_demand', 'france_solar', 'germany_solar',
            'france_wind', 'germany_wind', 'france_nuclear', 'germany_nuclear',
            'france_price', 'germany_price', 'temperature_france', 'temperature_germany'
        ]
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for flow forecasting"""
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
        
        # Price differential (key driver of flows)
        df['price_differential'] = df['france_price'] - df['germany_price']
        
        # Generation balance
        df['france_generation'] = df['france_solar'] + df['france_wind'] + df['france_nuclear']
        df['germany_generation'] = df['germany_solar'] + df['germany_wind'] + df['germany_nuclear']
        df['generation_balance'] = df['france_generation'] - df['germany_generation']
        
        # Demand balance
        df['demand_balance'] = df['france_demand'] - df['germany_demand']
        
        # Net position (generation - demand)
        df['france_net_position'] = df['france_generation'] - df['france_demand']
        df['germany_net_position'] = df['germany_generation'] - df['germany_demand']
        df['net_position_balance'] = df['france_net_position'] - df['germany_net_position']
        
        # Lag features
        for lag in [1, 2, 3, 24]:
            df[f'flow_lag_{lag}'] = df['cross_border_flow'].shift(lag)
            df[f'price_diff_lag_{lag}'] = df['price_differential'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'flow_rolling_mean_{window}'] = df['cross_border_flow'].rolling(window).mean()
            df[f'flow_rolling_std_{window}'] = df['cross_border_flow'].rolling(window).std()
            df[f'price_diff_rolling_mean_{window}'] = df['price_differential'].rolling(window).mean()
        
        # Flow volatility
        df['flow_volatility'] = df['cross_border_flow'].rolling(24).std()
        
        return df
    
    def train(self, historical_data: pd.DataFrame, target_column: str = 'cross_border_flow'):
        """Train the flow forecasting model"""
        try:
            # Prepare features
            features_df = self.prepare_features(historical_data)
            
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
                logger.warning("No valid training data available")
                return False
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            
            logger.info(f"Flow forecasting model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training flow forecasting model: {e}")
            return False
    
    def predict(self, forecast_data: pd.DataFrame) -> pd.DataFrame:
        """Predict cross-border flows"""
        if not self.is_trained:
            logger.warning("Model not trained. Using simple heuristic prediction.")
            return self._heuristic_prediction(forecast_data)
        
        try:
            # Prepare features
            features_df = self.prepare_features(forecast_data)
            
            # Select features
            feature_cols = [col for col in features_df.columns 
                          if col not in ['datetime', 'cross_border_flow']]
            X = features_df[feature_cols]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Create result DataFrame
            result_df = forecast_data.copy()
            result_df['predicted_flow'] = predictions
            result_df['flow_confidence'] = self._calculate_confidence(features_df)
            
            # Calculate flow direction and magnitude
            result_df['flow_direction'] = np.where(
                predictions > 0, 'France_to_Germany', 'Germany_to_France'
            )
            result_df['flow_magnitude'] = np.abs(predictions)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making flow predictions: {e}")
            return self._heuristic_prediction(forecast_data)
    
    def _heuristic_prediction(self, forecast_data: pd.DataFrame) -> pd.DataFrame:
        """Simple heuristic-based flow prediction when model is not trained"""
        result_df = forecast_data.copy()
        
        # Simple heuristic based on price differential
        result_df['predicted_flow'] = result_df.apply(
            lambda row: self._simple_flow_heuristic(row), axis=1
        )
        result_df['flow_confidence'] = 0.4  # Moderate confidence for heuristic
        result_df['flow_direction'] = np.where(
            result_df['predicted_flow'] > 0, 'France_to_Germany', 'Germany_to_France'
        )
        result_df['flow_magnitude'] = np.abs(result_df['predicted_flow'])
        
        return result_df
    
    def _simple_flow_heuristic(self, row) -> float:
        """Simple heuristic for flow prediction based on price differential"""
        # Price differential is the main driver
        price_diff = row.get('france_price', 0) - row.get('germany_price', 0)
        
        # Base flow calculation
        # Positive flow = France to Germany, Negative flow = Germany to France
        base_flow = price_diff * 0.1  # Scaling factor
        
        # Adjust for generation balance
        france_net = row.get('france_generation', 0) - row.get('france_demand', 0)
        germany_net = row.get('germany_generation', 0) - row.get('germany_demand', 0)
        net_balance = france_net - germany_net
        
        # Add generation balance effect
        flow_adjustment = net_balance * 0.05
        
        total_flow = base_flow + flow_adjustment
        
        # Apply some constraints (realistic flow limits)
        max_flow = 3000  # MW
        return np.clip(total_flow, -max_flow, max_flow)
    
    def _calculate_confidence(self, features_df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on feature quality"""
        confidence = 1.0
        
        # Reduce confidence for missing data
        missing_data_ratio = features_df.isna().sum().sum() / (len(features_df) * len(features_df.columns))
        confidence *= (1 - missing_data_ratio)
        
        # Reduce confidence for extreme price differentials
        if 'price_differential' in features_df.columns:
            price_diff_std = features_df['price_differential'].std()
            if price_diff_std > 50:  # High price volatility
                confidence *= 0.7
        
        return max(0.1, min(1.0, confidence))
    
    def analyze_flow_drivers(self, data: pd.DataFrame) -> Dict:
        """Analyze the main drivers of cross-border flows"""
        if data.empty:
            return {}
        
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Calculate correlations
            correlations = {}
            target_col = 'cross_border_flow'
            
            for col in features_df.columns:
                if col != target_col and col != 'datetime':
                    corr = features_df[col].corr(features_df[target_col])
                    if not np.isnan(corr):
                        correlations[col] = corr
            
            # Sort by absolute correlation
            sorted_correlations = sorted(
                correlations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            return {
                'correlations': dict(sorted_correlations[:10]),  # Top 10 drivers
                'strongest_driver': sorted_correlations[0] if sorted_correlations else None,
                'analysis_summary': self._generate_analysis_summary(correlations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing flow drivers: {e}")
            return {}
    
    def _generate_analysis_summary(self, correlations: Dict) -> str:
        """Generate a summary of flow driver analysis"""
        if not correlations:
            return "No correlation data available"
        
        # Find strongest positive and negative drivers
        positive_drivers = [(k, v) for k, v in correlations.items() if v > 0]
        negative_drivers = [(k, v) for k, v in correlations.items() if v < 0]
        
        positive_drivers.sort(key=lambda x: x[1], reverse=True)
        negative_drivers.sort(key=lambda x: x[1])
        
        summary = "Flow Driver Analysis:\n"
        
        if positive_drivers:
            summary += f"Strongest positive driver: {positive_drivers[0][0]} (corr: {positive_drivers[0][1]:.3f})\n"
        
        if negative_drivers:
            summary += f"Strongest negative driver: {negative_drivers[0][0]} (corr: {negative_drivers[0][1]:.3f})\n"
        
        return summary
    
    def evaluate_model(self, test_data: pd.DataFrame, target_column: str = 'cross_border_flow') -> Dict:
        """Evaluate model performance on test data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
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
