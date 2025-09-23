"""
Nuclear outage detection and impact modeling
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class NuclearMonitor:
    """Monitor nuclear plant outages and their impact on energy markets"""
    
    def __init__(self):
        self.outage_threshold = Config.NUCLEAR_OUTAGE_THRESHOLD
        self.france_nuclear_capacity = 61.3  # GW (approximate total capacity)
        self.germany_nuclear_capacity = 4.0   # GW (remaining capacity)
        
    def detect_outages(self, nuclear_data: pd.DataFrame) -> pd.DataFrame:
        """Detect nuclear plant outages from capacity data"""
        if nuclear_data.empty:
            return pd.DataFrame()
        
        df = nuclear_data.copy()
        
        # Calculate capacity factor
        df['capacity_factor'] = df['generation'] / df['capacity']
        
        # Detect outages (capacity factor below threshold)
        df['is_outage'] = df['capacity_factor'] < (1 - self.outage_threshold)
        
        # Calculate outage magnitude
        df['outage_magnitude'] = np.where(
            df['is_outage'],
            df['capacity'] - df['generation'],
            0
        )
        
        # Calculate impact on total system
        df['system_impact'] = df['outage_magnitude'] / (self.france_nuclear_capacity + self.germany_nuclear_capacity)
        
        return df
    
    def predict_outage_impact(self, outage_data: pd.DataFrame, 
                            price_data: pd.DataFrame) -> Dict:
        """Predict the impact of nuclear outages on prices and flows"""
        if outage_data.empty or price_data.empty:
            return {}
        
        try:
            # Merge outage and price data
            merged_data = pd.merge(
                outage_data, price_data, 
                on='datetime', how='inner'
            )
            
            # Calculate correlation between outages and prices
            outage_correlation = merged_data['system_impact'].corr(merged_data['price'])
            
            # Calculate average price impact during outages
            outage_periods = merged_data[merged_data['is_outage']]
            if len(outage_periods) > 0:
                avg_price_during_outage = outage_periods['price'].mean()
                avg_price_normal = merged_data[~merged_data['is_outage']]['price'].mean()
                price_impact = avg_price_during_outage - avg_price_normal
            else:
                price_impact = 0
            
            # Predict future impact based on current outages
            current_outages = outage_data[outage_data['is_outage']]
            total_outage_capacity = current_outages['outage_magnitude'].sum()
            
            # Estimate price impact based on historical correlation
            estimated_price_impact = outage_correlation * total_outage_capacity * 10  # Scaling factor
            
            return {
                'outage_correlation': outage_correlation,
                'price_impact': price_impact,
                'estimated_future_impact': estimated_price_impact,
                'total_outage_capacity': total_outage_capacity,
                'outage_count': len(current_outages)
            }
            
        except Exception as e:
            logger.error(f"Error predicting outage impact: {e}")
            return {}
    
    def get_outage_alerts(self, nuclear_data: pd.DataFrame) -> List[Dict]:
        """Generate alerts for significant nuclear outages"""
        alerts = []
        
        if nuclear_data.empty:
            return alerts
        
        # Detect current outages
        outage_data = self.detect_outages(nuclear_data)
        current_outages = outage_data[outage_data['is_outage']]
        
        for _, outage in current_outages.iterrows():
            # Calculate outage severity
            severity = outage['system_impact']
            
            if severity > 0.1:  # Significant outage (>10% of total nuclear capacity)
                alert = {
                    'type': 'nuclear_outage',
                    'severity': 'high',
                    'datetime': outage['datetime'],
                    'capacity_lost': outage['outage_magnitude'],
                    'system_impact': severity,
                    'message': f"Significant nuclear outage detected: {outage['outage_magnitude']:.1f} GW offline"
                }
                alerts.append(alert)
            
            elif severity > 0.05:  # Moderate outage
                alert = {
                    'type': 'nuclear_outage',
                    'severity': 'medium',
                    'datetime': outage['datetime'],
                    'capacity_lost': outage['outage_magnitude'],
                    'system_impact': severity,
                    'message': f"Moderate nuclear outage detected: {outage['outage_magnitude']:.1f} GW offline"
                }
                alerts.append(alert)
        
        return alerts
    
    def forecast_outage_duration(self, outage_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast the duration of current outages"""
        if outage_data.empty:
            return pd.DataFrame()
        
        df = outage_data.copy()
        
        # Simple heuristic for outage duration
        # In practice, you'd use more sophisticated models
        df['estimated_duration_hours'] = df.apply(
            lambda row: self._estimate_outage_duration(row), axis=1
        )
        
        # Calculate expected end time
        df['estimated_end_time'] = df['datetime'] + pd.to_timedelta(
            df['estimated_duration_hours'], unit='h'
        )
        
        return df
    
    def _estimate_outage_duration(self, row) -> float:
        """Estimate outage duration based on magnitude and type"""
        outage_magnitude = row['outage_magnitude']
        system_impact = row['system_impact']
        
        # Base duration on outage magnitude
        if system_impact > 0.2:  # Major outage
            base_duration = 48  # 2 days
        elif system_impact > 0.1:  # Significant outage
            base_duration = 24  # 1 day
        else:  # Minor outage
            base_duration = 12  # 12 hours
        
        # Add some randomness to simulate uncertainty
        duration_variance = base_duration * 0.3
        estimated_duration = base_duration + np.random.normal(0, duration_variance)
        
        return max(1, estimated_duration)  # Minimum 1 hour
    
    def get_nuclear_forecast(self, nuclear_data: pd.DataFrame, 
                           forecast_hours: int = 48) -> pd.DataFrame:
        """Generate nuclear capacity forecast"""
        if nuclear_data.empty:
            return pd.DataFrame()
        
        # Detect current outages
        outage_data = self.detect_outages(nuclear_data)
        
        # Create forecast timeline
        start_time = nuclear_data['datetime'].max()
        forecast_times = pd.date_range(
            start=start_time,
            periods=forecast_hours + 1,
            freq='H'
        )[1:]  # Exclude start time
        
        forecast_data = []
        
        for forecast_time in forecast_times:
            # Get current outages that might still be active
            active_outages = outage_data[
                (outage_data['is_outage']) & 
                (outage_data['datetime'] <= forecast_time)
            ]
            
            # Estimate if outages will still be active
            total_outage_capacity = 0
            for _, outage in active_outages.iterrows():
                estimated_duration = self._estimate_outage_duration(outage)
                outage_end_time = outage['datetime'] + timedelta(hours=estimated_duration)
                
                if forecast_time <= outage_end_time:
                    total_outage_capacity += outage['outage_magnitude']
            
            # Calculate forecasted capacity
            total_capacity = self.france_nuclear_capacity + self.germany_nuclear_capacity
            forecasted_capacity = total_capacity - total_outage_capacity
            
            forecast_data.append({
                'datetime': forecast_time,
                'forecasted_capacity': forecasted_capacity,
                'outage_capacity': total_outage_capacity,
                'capacity_factor': forecasted_capacity / total_capacity
            })
        
        return pd.DataFrame(forecast_data)
