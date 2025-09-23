"""
Volatility analysis and risk management tools
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """Analyze price and flow volatility for risk management"""
    
    def __init__(self):
        self.volatility_threshold = Config.VOLATILITY_THRESHOLD
        self.high_volatility_threshold = Config.HIGH_VOLATILITY_THRESHOLD
    
    def calculate_price_volatility(self, price_data: pd.DataFrame, 
                                window_hours: int = 24) -> pd.DataFrame:
        """Calculate price volatility metrics"""
        df = price_data.copy()
        
        # Rolling volatility (standard deviation)
        df['price_volatility'] = df['price'].rolling(window_hours).std()
        
        # Rolling coefficient of variation
        df['price_cv'] = df['price_volatility'] / df['price'].rolling(window_hours).mean()
        
        # Price range (max - min) in rolling window
        df['price_range'] = df['price'].rolling(window_hours).max() - df['price'].rolling(window_hours).min()
        
        # Volatility classification
        df['volatility_level'] = df['price_volatility'].apply(self._classify_volatility)
        
        # Volatility spikes (sudden increases)
        df['volatility_spike'] = df['price_volatility'].diff() > df['price_volatility'].rolling(24).std()
        
        return df
    
    def calculate_flow_volatility(self, flow_data: pd.DataFrame, 
                                window_hours: int = 24) -> pd.DataFrame:
        """Calculate cross-border flow volatility"""
        df = flow_data.copy()
        
        # Rolling volatility
        df['flow_volatility'] = df['cross_border_flow'].rolling(window_hours).std()
        
        # Flow direction changes
        df['flow_direction'] = np.where(df['cross_border_flow'] > 0, 1, -1)
        df['direction_changes'] = (df['flow_direction'] != df['flow_direction'].shift()).astype(int)
        
        # Flow magnitude volatility
        df['flow_magnitude'] = np.abs(df['cross_border_flow'])
        df['magnitude_volatility'] = df['flow_magnitude'].rolling(window_hours).std()
        
        # Flow volatility classification
        df['flow_volatility_level'] = df['flow_volatility'].apply(self._classify_volatility)
        
        return df
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if pd.isna(volatility):
            return 'unknown'
        elif volatility < self.volatility_threshold:
            return 'low'
        elif volatility < self.high_volatility_threshold:
            return 'medium'
        else:
            return 'high'
    
    def detect_volatility_patterns(self, data: pd.DataFrame) -> Dict:
        """Detect patterns in volatility data"""
        patterns = {}
        
        # Time-based patterns
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        
        # Hourly volatility patterns
        hourly_volatility = data.groupby('hour')['price_volatility'].mean()
        patterns['peak_volatility_hours'] = hourly_volatility.nlargest(3).index.tolist()
        patterns['low_volatility_hours'] = hourly_volatility.nsmallest(3).index.tolist()
        
        # Weekly patterns
        weekly_volatility = data.groupby('day_of_week')['price_volatility'].mean()
        patterns['volatile_days'] = weekly_volatility.nlargest(2).index.tolist()
        patterns['stable_days'] = weekly_volatility.nsmallest(2).index.tolist()
        
        # Volatility clustering
        high_volatility_periods = data[data['volatility_level'] == 'high']
        patterns['volatility_clusters'] = self._find_volatility_clusters(high_volatility_periods)
        
        return patterns
    
    def _find_volatility_clusters(self, high_volatility_data: pd.DataFrame) -> List[Dict]:
        """Find clusters of high volatility periods"""
        if high_volatility_data.empty:
            return []
        
        clusters = []
        current_cluster = []
        
        for _, row in high_volatility_data.iterrows():
            if not current_cluster:
                current_cluster = [row]
            else:
                # Check if this row is within 6 hours of the last row in current cluster
                time_diff = (row['datetime'] - current_cluster[-1]['datetime']).total_seconds() / 3600
                if time_diff <= 6:
                    current_cluster.append(row)
                else:
                    # End current cluster and start new one
                    if len(current_cluster) >= 3:  # Only keep significant clusters
                        clusters.append({
                            'start_time': current_cluster[0]['datetime'],
                            'end_time': current_cluster[-1]['datetime'],
                            'duration_hours': len(current_cluster),
                            'max_volatility': max([r['price_volatility'] for r in current_cluster]),
                            'avg_volatility': np.mean([r['price_volatility'] for r in current_cluster])
                        })
                    current_cluster = [row]
        
        # Handle the last cluster
        if len(current_cluster) >= 3:
            clusters.append({
                'start_time': current_cluster[0]['datetime'],
                'end_time': current_cluster[-1]['datetime'],
                'duration_hours': len(current_cluster),
                'max_volatility': max([r['price_volatility'] for r in current_cluster]),
                'avg_volatility': np.mean([r['price_volatility'] for r in current_cluster])
            })
        
        return clusters
    
    def forecast_volatility(self, historical_data: pd.DataFrame, 
                          forecast_hours: int = 48) -> pd.DataFrame:
        """Forecast future volatility based on historical patterns"""
        if historical_data.empty:
            return pd.DataFrame()
        
        # Calculate historical volatility
        vol_data = self.calculate_price_volatility(historical_data)
        
        # Create forecast timeline
        start_time = vol_data['datetime'].max()
        forecast_times = pd.date_range(
            start=start_time,
            periods=forecast_hours + 1,
            freq='H'
        )[1:]
        
        forecast_data = []
        
        for forecast_time in forecast_times:
            # Get historical volatility for similar time periods
            hour = forecast_time.hour
            day_of_week = forecast_time.dayofweek
            
            # Find similar historical periods
            similar_periods = vol_data[
                (vol_data['datetime'].dt.hour == hour) & 
                (vol_data['datetime'].dt.dayofweek == day_of_week)
            ]
            
            if len(similar_periods) > 0:
                # Use historical average for similar periods
                forecasted_volatility = similar_periods['price_volatility'].mean()
                confidence = min(1.0, len(similar_periods) / 100)  # More data = higher confidence
            else:
                # Fallback to overall average
                forecasted_volatility = vol_data['price_volatility'].mean()
                confidence = 0.3
            
            # Add some trend-based adjustment
            recent_volatility = vol_data['price_volatility'].tail(24).mean()
            trend_adjustment = (recent_volatility - vol_data['price_volatility'].mean()) * 0.1
            forecasted_volatility += trend_adjustment
            
            forecast_data.append({
                'datetime': forecast_time,
                'forecasted_volatility': max(0, forecasted_volatility),
                'volatility_confidence': confidence,
                'volatility_level': self._classify_volatility(forecasted_volatility)
            })
        
        return pd.DataFrame(forecast_data)
    
    def generate_risk_alerts(self, data: pd.DataFrame) -> List[Dict]:
        """Generate risk alerts based on volatility analysis"""
        alerts = []
        
        if data.empty:
            return alerts
        
        # Calculate current volatility
        vol_data = self.calculate_price_volatility(data)
        current_volatility = vol_data['price_volatility'].iloc[-1]
        current_level = self._classify_volatility(current_volatility)
        
        # High volatility alert
        if current_level == 'high':
            alerts.append({
                'type': 'high_volatility',
                'severity': 'high',
                'datetime': vol_data['datetime'].iloc[-1],
                'value': current_volatility,
                'message': f"High price volatility detected: {current_volatility:.2f} €/MWh"
            })
        
        # Volatility spike alert
        if vol_data['volatility_spike'].iloc[-1]:
            alerts.append({
                'type': 'volatility_spike',
                'severity': 'medium',
                'datetime': vol_data['datetime'].iloc[-1],
                'value': current_volatility,
                'message': f"Sudden volatility spike detected: {current_volatility:.2f} €/MWh"
            })
        
        # Check for volatility clusters
        recent_data = vol_data.tail(24)  # Last 24 hours
        high_vol_count = len(recent_data[recent_data['volatility_level'] == 'high'])
        
        if high_vol_count >= 6:  # High volatility for 6+ hours
            alerts.append({
                'type': 'volatility_cluster',
                'severity': 'high',
                'datetime': vol_data['datetime'].iloc[-1],
                'value': high_vol_count,
                'message': f"Extended high volatility period: {high_vol_count} hours"
            })
        
        return alerts
    
    def calculate_value_at_risk(self, price_data: pd.DataFrame, 
                              confidence_level: float = 0.95) -> Dict:
        """Calculate Value at Risk (VaR) for price data"""
        if price_data.empty:
            return {}
        
        # Calculate price returns
        returns = price_data['price'].pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Historical VaR
        var_historical = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        var_parametric = mean_return + std_return * np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100)
        
        # Expected Shortfall (Conditional VaR)
        var_threshold = var_historical
        tail_returns = returns[returns <= var_threshold]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_historical
        
        return {
            'var_historical': var_historical,
            'var_parametric': var_parametric,
            'expected_shortfall': expected_shortfall,
            'confidence_level': confidence_level,
            'sample_size': len(returns)
        }
    
    def generate_risk_report(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive risk analysis report"""
        if data.empty:
            return {'error': 'No data available for risk analysis'}
        
        try:
            # Calculate volatility metrics
            vol_data = self.calculate_price_volatility(data)
            
            # Calculate VaR
            var_metrics = self.calculate_value_at_risk(data)
            
            # Detect patterns
            patterns = self.detect_volatility_patterns(vol_data)
            
            # Generate alerts
            alerts = self.generate_risk_alerts(data)
            
            # Summary statistics
            summary = {
                'avg_volatility': vol_data['price_volatility'].mean(),
                'max_volatility': vol_data['price_volatility'].max(),
                'volatility_trend': self._calculate_volatility_trend(vol_data),
                'high_volatility_periods': len(vol_data[vol_data['volatility_level'] == 'high']),
                'total_periods': len(vol_data)
            }
            
            return {
                'summary': summary,
                'var_metrics': var_metrics,
                'patterns': patterns,
                'alerts': alerts,
                'recommendations': self._generate_risk_recommendations(summary, patterns, alerts)
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def _calculate_volatility_trend(self, vol_data: pd.DataFrame) -> str:
        """Calculate volatility trend (increasing, decreasing, stable)"""
        if len(vol_data) < 24:
            return 'insufficient_data'
        
        recent_vol = vol_data['price_volatility'].tail(24).mean()
        historical_vol = vol_data['price_volatility'].mean()
        
        if recent_vol > historical_vol * 1.1:
            return 'increasing'
        elif recent_vol < historical_vol * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _generate_risk_recommendations(self, summary: Dict, patterns: Dict, alerts: List) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Volatility-based recommendations
        if summary['avg_volatility'] > self.high_volatility_threshold:
            recommendations.append("Consider reducing position sizes due to high volatility")
            recommendations.append("Implement tighter stop-loss orders")
        
        # Pattern-based recommendations
        if patterns.get('volatile_days'):
            recommendations.append(f"Be cautious on days {patterns['volatile_days']} (historically more volatile)")
        
        if patterns.get('volatility_clusters'):
            recommendations.append("Monitor for volatility clustering - high volatility tends to persist")
        
        # Alert-based recommendations
        high_severity_alerts = [a for a in alerts if a['severity'] == 'high']
        if high_severity_alerts:
            recommendations.append("High severity alerts active - consider reducing exposure")
        
        # Trend-based recommendations
        if summary['volatility_trend'] == 'increasing':
            recommendations.append("Volatility is increasing - consider hedging strategies")
        elif summary['volatility_trend'] == 'decreasing':
            recommendations.append("Volatility is decreasing - may be opportunity to increase exposure")
        
        return recommendations
