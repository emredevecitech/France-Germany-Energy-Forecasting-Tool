"""
Backtesting and validation framework for forecasting models
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from typing import Dict, List, Optional, Tuple
from forecasting_engine import ForecastingEngine
from config import Config

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting framework for forecasting models"""
    
    def __init__(self):
        self.forecasting_engine = ForecastingEngine()
        self.results = {}
    
    def run_backtest(self, start_date: datetime, end_date: datetime, 
                    forecast_horizon: int = 24) -> Dict:
        """Run backtest for the specified period"""
        try:
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Generate test periods
            test_periods = self._generate_test_periods(start_date, end_date, forecast_horizon)
            
            results = {
                'price_forecast': [],
                'flow_forecast': [],
                'solar_forecast': [],
                'nuclear_forecast': [],
                'volatility_forecast': []
            }
            
            for period_start, period_end in test_periods:
                logger.info(f"Testing period: {period_start} to {period_end}")
                
                # Generate forecast for this period
                forecast = self._generate_forecast_for_period(period_start, period_end)
                
                # Get actual data for validation
                actual_data = self._get_actual_data(period_start, period_end)
                
                # Calculate metrics for each forecast type
                for forecast_type in results.keys():
                    if forecast_type in forecast and forecast_type in actual_data:
                        metrics = self._calculate_forecast_metrics(
                            forecast[forecast_type], 
                            actual_data[forecast_type],
                            forecast_type
                        )
                        results[forecast_type].append(metrics)
            
            # Aggregate results
            aggregated_results = self._aggregate_results(results)
            
            # Generate backtest report
            report = self._generate_backtest_report(aggregated_results)
            
            self.results = report
            return report
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def _generate_test_periods(self, start_date: datetime, end_date: datetime, 
                              forecast_horizon: int) -> List[Tuple[datetime, datetime]]:
        """Generate test periods for backtesting"""
        periods = []
        current_date = start_date
        
        while current_date + timedelta(hours=forecast_horizon) <= end_date:
            period_start = current_date
            period_end = current_date + timedelta(hours=forecast_horizon)
            periods.append((period_start, period_end))
            
            # Move to next period (overlapping for more test cases)
            current_date += timedelta(hours=6)  # 6-hour overlap
        
        return periods
    
    def _generate_forecast_for_period(self, start_date: datetime, 
                                    end_date: datetime) -> Dict:
        """Generate forecast for a specific period"""
        try:
            # This would use the forecasting engine with historical data up to start_date
            # For now, return placeholder data
            forecast = {
                'price_forecast': self._generate_placeholder_forecast(start_date, end_date, 'price'),
                'flow_forecast': self._generate_placeholder_forecast(start_date, end_date, 'flow'),
                'solar_forecast': self._generate_placeholder_forecast(start_date, end_date, 'solar'),
                'nuclear_forecast': self._generate_placeholder_forecast(start_date, end_date, 'nuclear'),
                'volatility_forecast': self._generate_placeholder_forecast(start_date, end_date, 'volatility')
            }
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast for period: {e}")
            return {}
    
    def _generate_placeholder_forecast(self, start_date: datetime, end_date: datetime, 
                                     forecast_type: str) -> pd.DataFrame:
        """Generate placeholder forecast data for testing"""
        # Create time series
        time_series = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate synthetic data based on forecast type
        if forecast_type == 'price':
            # Price forecast with some trend and seasonality
            base_price = 50
            trend = np.linspace(0, 10, len(time_series))
            seasonality = 10 * np.sin(2 * np.pi * np.arange(len(time_series)) / 24)
            noise = np.random.normal(0, 5, len(time_series))
            values = base_price + trend + seasonality + noise
        elif forecast_type == 'flow':
            # Flow forecast with bidirectional flow
            base_flow = 0
            trend = np.linspace(-100, 100, len(time_series))
            seasonality = 50 * np.sin(2 * np.pi * np.arange(len(time_series)) / 24)
            noise = np.random.normal(0, 20, len(time_series))
            values = base_flow + trend + seasonality + noise
        elif forecast_type == 'solar':
            # Solar forecast with daily pattern
            base_solar = 0
            daily_pattern = np.maximum(0, 20 * np.sin(np.pi * np.arange(len(time_series)) / 12))
            noise = np.random.normal(0, 2, len(time_series))
            values = base_solar + daily_pattern + noise
        elif forecast_type == 'nuclear':
            # Nuclear forecast with some outages
            base_nuclear = 60
            outages = np.random.choice([0, -10, -20], len(time_series), p=[0.8, 0.15, 0.05])
            noise = np.random.normal(0, 2, len(time_series))
            values = base_nuclear + outages + noise
        else:  # volatility
            # Volatility forecast
            base_volatility = 0.1
            trend = np.linspace(0, 0.05, len(time_series))
            noise = np.random.normal(0, 0.02, len(time_series))
            values = base_volatility + trend + noise
        
        return pd.DataFrame({
            'datetime': time_series,
            f'predicted_{forecast_type}': values
        })
    
    def _get_actual_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get actual data for validation"""
        try:
            # This would fetch actual data from the data sources
            # For now, return placeholder data
            actual_data = {
                'price_forecast': self._generate_placeholder_actual(start_date, end_date, 'price'),
                'flow_forecast': self._generate_placeholder_actual(start_date, end_date, 'flow'),
                'solar_forecast': self._generate_placeholder_actual(start_date, end_date, 'solar'),
                'nuclear_forecast': self._generate_placeholder_actual(start_date, end_date, 'nuclear'),
                'volatility_forecast': self._generate_placeholder_actual(start_date, end_date, 'volatility')
            }
            
            return actual_data
            
        except Exception as e:
            logger.error(f"Error getting actual data: {e}")
            return {}
    
    def _generate_placeholder_actual(self, start_date: datetime, end_date: datetime, 
                                   data_type: str) -> pd.DataFrame:
        """Generate placeholder actual data for testing"""
        # Create time series
        time_series = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Generate synthetic actual data (similar to forecast but with some differences)
        if data_type == 'price':
            base_price = 50
            trend = np.linspace(0, 8, len(time_series))  # Slightly different trend
            seasonality = 12 * np.sin(2 * np.pi * np.arange(len(time_series)) / 24)  # Different amplitude
            noise = np.random.normal(0, 6, len(time_series))  # Different noise
            values = base_price + trend + seasonality + noise
        elif data_type == 'flow':
            base_flow = 0
            trend = np.linspace(-80, 120, len(time_series))  # Different trend
            seasonality = 60 * np.sin(2 * np.pi * np.arange(len(time_series)) / 24)
            noise = np.random.normal(0, 25, len(time_series))
            values = base_flow + trend + seasonality + noise
        elif data_type == 'solar':
            base_solar = 0
            daily_pattern = np.maximum(0, 22 * np.sin(np.pi * np.arange(len(time_series)) / 12))
            noise = np.random.normal(0, 3, len(time_series))
            values = base_solar + daily_pattern + noise
        elif data_type == 'nuclear':
            base_nuclear = 60
            outages = np.random.choice([0, -8, -18], len(time_series), p=[0.85, 0.12, 0.03])
            noise = np.random.normal(0, 3, len(time_series))
            values = base_nuclear + outages + noise
        else:  # volatility
            base_volatility = 0.1
            trend = np.linspace(0, 0.03, len(time_series))
            noise = np.random.normal(0, 0.025, len(time_series))
            values = base_volatility + trend + noise
        
        return pd.DataFrame({
            'datetime': time_series,
            f'actual_{data_type}': values
        })
    
    def _calculate_forecast_metrics(self, forecast_df: pd.DataFrame, 
                                  actual_df: pd.DataFrame, forecast_type: str) -> Dict:
        """Calculate forecast accuracy metrics"""
        try:
            if forecast_df.empty or actual_df.empty:
                return {'error': 'Empty data'}
            
            # Merge forecast and actual data
            merged_data = pd.merge(forecast_df, actual_df, on='datetime', how='inner')
            
            if merged_data.empty:
                return {'error': 'No matching data'}
            
            # Get column names
            forecast_col = f'predicted_{forecast_type}'
            actual_col = f'actual_{forecast_type}'
            
            if forecast_col not in merged_data.columns or actual_col not in merged_data.columns:
                return {'error': 'Missing required columns'}
            
            # Calculate metrics
            forecast_values = merged_data[forecast_col].values
            actual_values = merged_data[actual_col].values
            
            # Remove any NaN values
            mask = ~(np.isnan(forecast_values) | np.isnan(actual_values))
            forecast_values = forecast_values[mask]
            actual_values = actual_values[mask]
            
            if len(forecast_values) == 0:
                return {'error': 'No valid data after cleaning'}
            
            # Calculate standard metrics
            mae = mean_absolute_error(actual_values, forecast_values)
            mse = mean_squared_error(actual_values, forecast_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_values, forecast_values)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
            bias = np.mean(forecast_values - actual_values)
            
            # Calculate directional accuracy (for price/flow)
            if forecast_type in ['price', 'flow']:
                forecast_direction = np.diff(forecast_values) > 0
                actual_direction = np.diff(actual_values) > 0
                directional_accuracy = np.mean(forecast_direction == actual_direction) * 100
            else:
                directional_accuracy = None
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'bias': bias,
                'directional_accuracy': directional_accuracy,
                'sample_size': len(forecast_values)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {forecast_type}: {e}")
            return {'error': str(e)}
    
    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate results across all test periods"""
        aggregated = {}
        
        for forecast_type, metrics_list in results.items():
            if not metrics_list:
                aggregated[forecast_type] = {'error': 'No data'}
                continue
            
            # Filter out error results
            valid_metrics = [m for m in metrics_list if 'error' not in m]
            
            if not valid_metrics:
                aggregated[forecast_type] = {'error': 'No valid metrics'}
                continue
            
            # Calculate aggregated metrics
            aggregated[forecast_type] = {
                'avg_mae': np.mean([m['mae'] for m in valid_metrics]),
                'avg_rmse': np.mean([m['rmse'] for m in valid_metrics]),
                'avg_r2': np.mean([m['r2'] for m in valid_metrics]),
                'avg_mape': np.mean([m['mape'] for m in valid_metrics]),
                'avg_bias': np.mean([m['bias'] for m in valid_metrics]),
                'avg_directional_accuracy': np.mean([m['directional_accuracy'] for m in valid_metrics if m['directional_accuracy'] is not None]),
                'total_periods': len(valid_metrics),
                'std_mae': np.std([m['mae'] for m in valid_metrics]),
                'std_rmse': np.std([m['rmse'] for m in valid_metrics]),
                'std_r2': np.std([m['r2'] for m in valid_metrics])
            }
        
        return aggregated
    
    def _generate_backtest_report(self, results: Dict) -> Dict:
        """Generate comprehensive backtest report"""
        report = {
            'summary': {},
            'detailed_results': results,
            'recommendations': []
        }
        
        # Generate summary
        for forecast_type, metrics in results.items():
            if 'error' not in metrics:
                report['summary'][forecast_type] = {
                    'mae': metrics['avg_mae'],
                    'rmse': metrics['avg_rmse'],
                    'r2': metrics['avg_r2'],
                    'mape': metrics['avg_mape']
                }
        
        # Generate recommendations
        recommendations = []
        
        for forecast_type, metrics in results.items():
            if 'error' in metrics:
                continue
            
            # MAE recommendations
            if metrics['avg_mae'] > 10:
                recommendations.append(f"{forecast_type}: High MAE ({metrics['avg_mae']:.2f}) - consider model improvements")
            
            # R² recommendations
            if metrics['avg_r2'] < 0.5:
                recommendations.append(f"{forecast_type}: Low R² ({metrics['avg_r2']:.2f}) - model explains less than 50% of variance")
            
            # Bias recommendations
            if abs(metrics['avg_bias']) > 5:
                recommendations.append(f"{forecast_type}: Significant bias ({metrics['avg_bias']:.2f}) - model may be systematically over/under-predicting")
        
        report['recommendations'] = recommendations
        
        return report
    
    def run_walk_forward_analysis(self, start_date: datetime, end_date: datetime, 
                                 retrain_frequency: int = 7) -> Dict:
        """Run walk-forward analysis with periodic retraining"""
        try:
            logger.info(f"Starting walk-forward analysis from {start_date} to {end_date}")
            
            # Generate retraining dates
            retrain_dates = pd.date_range(
                start=start_date, 
                end=end_date, 
                freq=f'{retrain_frequency}D'
            )
            
            results = []
            
            for retrain_date in retrain_dates:
                logger.info(f"Retraining at {retrain_date}")
                
                # Retrain models with data up to retrain_date
                self._retrain_models(retrain_date)
                
                # Test on next period
                test_start = retrain_date
                test_end = retrain_date + timedelta(days=retrain_frequency)
                
                if test_end > end_date:
                    test_end = end_date
                
                # Run backtest for this period
                period_results = self.run_backtest(test_start, test_end)
                
                if 'error' not in period_results:
                    results.append({
                        'retrain_date': retrain_date,
                        'test_period': (test_start, test_end),
                        'results': period_results
                    })
            
            # Aggregate walk-forward results
            aggregated_results = self._aggregate_walk_forward_results(results)
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Error running walk-forward analysis: {e}")
            return {'error': str(e)}
    
    def _retrain_models(self, retrain_date: datetime):
        """Retrain models with data up to retrain_date"""
        # This would retrain the forecasting models
        # For now, just log the action
        logger.info(f"Retraining models with data up to {retrain_date}")
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from walk-forward analysis"""
        if not results:
            return {'error': 'No results to aggregate'}
        
        # Extract metrics from all periods
        all_metrics = {}
        
        for result in results:
            period_results = result['results']
            for forecast_type, metrics in period_results.items():
                if forecast_type not in all_metrics:
                    all_metrics[forecast_type] = []
                
                if 'error' not in metrics:
                    all_metrics[forecast_type].append(metrics)
        
        # Calculate aggregated metrics
        aggregated = {}
        
        for forecast_type, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            
            aggregated[forecast_type] = {
                'avg_mae': np.mean([m['avg_mae'] for m in metrics_list]),
                'avg_rmse': np.mean([m['avg_rmse'] for m in metrics_list]),
                'avg_r2': np.mean([m['avg_r2'] for m in metrics_list]),
                'avg_mape': np.mean([m['avg_mape'] for m in metrics_list]),
                'total_periods': len(metrics_list),
                'performance_trend': self._calculate_performance_trend(metrics_list)
            }
        
        return {
            'aggregated_results': aggregated,
            'individual_periods': results,
            'total_periods': len(results)
        }
    
    def _calculate_performance_trend(self, metrics_list: List[Dict]) -> str:
        """Calculate performance trend over time"""
        if len(metrics_list) < 2:
            return 'insufficient_data'
        
        # Calculate trend in MAE
        mae_values = [m['avg_mae'] for m in metrics_list]
        mae_trend = np.polyfit(range(len(mae_values)), mae_values, 1)[0]
        
        if mae_trend < -0.1:
            return 'improving'
        elif mae_trend > 0.1:
            return 'deteriorating'
        else:
            return 'stable'
