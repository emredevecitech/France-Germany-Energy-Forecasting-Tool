"""
Comprehensive risk management system for energy trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from risk_management.volatility_analyzer import VolatilityAnalyzer
from config import Config

logger = logging.getLogger(__name__)

class RiskManager:
    """Comprehensive risk management for energy trading"""
    
    def __init__(self):
        self.volatility_analyzer = VolatilityAnalyzer()
        self.risk_limits = {
            'max_position_size': 1000,  # MW
            'max_daily_loss': 50000,   # €
            'max_volatility_exposure': 0.3,  # 30% of portfolio
            'max_correlation_exposure': 0.5   # 50% correlation limit
        }
        self.alert_thresholds = {
            'price_volatility': 0.2,    # 20% price volatility
            'flow_volatility': 0.15,    # 15% flow volatility
            'correlation_risk': 0.8,    # 80% correlation
            'liquidity_risk': 0.1       # 10% liquidity threshold
        }
    
    def assess_portfolio_risk(self, positions: pd.DataFrame, 
                            market_data: pd.DataFrame) -> Dict:
        """Assess overall portfolio risk"""
        try:
            risk_metrics = {}
            
            # Position risk
            position_risk = self._calculate_position_risk(positions)
            risk_metrics['position_risk'] = position_risk
            
            # Market risk
            market_risk = self._calculate_market_risk(market_data)
            risk_metrics['market_risk'] = market_risk
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk(positions, market_data)
            risk_metrics['correlation_risk'] = correlation_risk
            
            # Liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(market_data)
            risk_metrics['liquidity_risk'] = liquidity_risk
            
            # Overall risk score
            risk_metrics['overall_risk_score'] = self._calculate_overall_risk_score(risk_metrics)
            
            # Risk alerts
            risk_metrics['alerts'] = self._generate_risk_alerts(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {'error': str(e)}
    
    def _calculate_position_risk(self, positions: pd.DataFrame) -> Dict:
        """Calculate position-based risk metrics"""
        if positions.empty:
            return {'error': 'No positions data'}
        
        # Position size risk
        total_exposure = positions['volume'].sum()
        max_single_position = positions['volume'].max()
        position_concentration = max_single_position / total_exposure if total_exposure > 0 else 0
        
        # Time-based risk (positions expiring soon)
        current_time = datetime.now()
        expiring_soon = positions[
            positions['expiry_time'] <= current_time + timedelta(hours=24)
        ]
        expiring_exposure = expiring_soon['volume'].sum()
        
        return {
            'total_exposure': total_exposure,
            'max_single_position': max_single_position,
            'position_concentration': position_concentration,
            'expiring_exposure': expiring_exposure,
            'position_count': len(positions)
        }
    
    def _calculate_market_risk(self, market_data: pd.DataFrame) -> Dict:
        """Calculate market-based risk metrics"""
        if market_data.empty:
            return {'error': 'No market data'}
        
        # Price volatility
        price_volatility = market_data['price'].std() / market_data['price'].mean()
        
        # Flow volatility
        flow_volatility = market_data['cross_border_flow'].std() / abs(market_data['cross_border_flow']).mean()
        
        # Price trend
        price_trend = self._calculate_trend(market_data['price'])
        
        # Volatility clustering
        volatility_clusters = self.volatility_analyzer._find_volatility_clusters(
            market_data[market_data['price_volatility'] > Config.HIGH_VOLATILITY_THRESHOLD]
        )
        
        return {
            'price_volatility': price_volatility,
            'flow_volatility': flow_volatility,
            'price_trend': price_trend,
            'volatility_clusters': len(volatility_clusters),
            'market_stress_level': self._calculate_market_stress(price_volatility, flow_volatility)
        }
    
    def _calculate_correlation_risk(self, positions: pd.DataFrame, 
                                  market_data: pd.DataFrame) -> Dict:
        """Calculate correlation-based risk"""
        if positions.empty or market_data.empty:
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Calculate correlations between different positions
        position_correlations = []
        
        # This is a simplified version - in practice, you'd calculate
        # correlations between different position types, regions, etc.
        
        return {
            'max_correlation': max(position_correlations) if position_correlations else 0,
            'avg_correlation': np.mean(position_correlations) if position_correlations else 0,
            'high_correlation_pairs': len([c for c in position_correlations if c > 0.8])
        }
    
    def _calculate_liquidity_risk(self, market_data: pd.DataFrame) -> Dict:
        """Calculate liquidity risk metrics"""
        if market_data.empty:
            return {'error': 'No market data'}
        
        # Volume-based liquidity
        avg_volume = market_data['volume'].mean() if 'volume' in market_data.columns else 0
        volume_volatility = market_data['volume'].std() if 'volume' in market_data.columns else 0
        
        # Bid-ask spread (if available)
        spread_risk = 0  # Placeholder - would need bid/ask data
        
        return {
            'avg_volume': avg_volume,
            'volume_volatility': volume_volatility,
            'spread_risk': spread_risk,
            'liquidity_score': self._calculate_liquidity_score(avg_volume, volume_volatility)
        }
    
    def _calculate_liquidity_score(self, avg_volume: float, volume_volatility: float) -> float:
        """Calculate liquidity score (0-1, higher is better)"""
        if avg_volume == 0:
            return 0
        
        # Higher volume = better liquidity
        volume_score = min(1.0, avg_volume / 1000)  # Normalize to 1000 MW
        
        # Lower volatility = better liquidity
        volatility_score = max(0, 1 - volume_volatility / avg_volume)
        
        return (volume_score + volatility_score) / 2
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_market_stress(self, price_volatility: float, flow_volatility: float) -> str:
        """Calculate market stress level"""
        stress_score = (price_volatility + flow_volatility) / 2
        
        if stress_score > 0.3:
            return 'high'
        elif stress_score > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict) -> float:
        """Calculate overall risk score (0-1, higher is riskier)"""
        scores = []
        
        # Position risk score
        if 'position_risk' in risk_metrics and 'error' not in risk_metrics['position_risk']:
            pos_risk = risk_metrics['position_risk']
            position_score = min(1.0, pos_risk['total_exposure'] / self.risk_limits['max_position_size'])
            scores.append(position_score)
        
        # Market risk score
        if 'market_risk' in risk_metrics and 'error' not in risk_metrics['market_risk']:
            market_risk = risk_metrics['market_risk']
            market_score = min(1.0, market_risk['price_volatility'])
            scores.append(market_score)
        
        # Correlation risk score
        if 'correlation_risk' in risk_metrics and 'error' not in risk_metrics['correlation_risk']:
            corr_risk = risk_metrics['correlation_risk']
            correlation_score = min(1.0, corr_risk['max_correlation'])
            scores.append(correlation_score)
        
        # Liquidity risk score
        if 'liquidity_risk' in risk_metrics and 'error' not in risk_metrics['liquidity_risk']:
            liq_risk = risk_metrics['liquidity_risk']
            liquidity_score = 1 - liq_risk['liquidity_score']  # Invert liquidity score
            scores.append(liquidity_score)
        
        return np.mean(scores) if scores else 0
    
    def _generate_risk_alerts(self, risk_metrics: Dict) -> List[Dict]:
        """Generate risk alerts based on risk metrics"""
        alerts = []
        
        # Position size alerts
        if 'position_risk' in risk_metrics and 'error' not in risk_metrics['position_risk']:
            pos_risk = risk_metrics['position_risk']
            
            if pos_risk['total_exposure'] > self.risk_limits['max_position_size']:
                alerts.append({
                    'type': 'position_size',
                    'severity': 'high',
                    'message': f"Total exposure {pos_risk['total_exposure']} MW exceeds limit {self.risk_limits['max_position_size']} MW"
                })
            
            if pos_risk['position_concentration'] > 0.5:
                alerts.append({
                    'type': 'position_concentration',
                    'severity': 'medium',
                    'message': f"Position concentration {pos_risk['position_concentration']:.1%} is high"
                })
        
        # Market risk alerts
        if 'market_risk' in risk_metrics and 'error' not in risk_metrics['market_risk']:
            market_risk = risk_metrics['market_risk']
            
            if market_risk['price_volatility'] > self.alert_thresholds['price_volatility']:
                alerts.append({
                    'type': 'price_volatility',
                    'severity': 'high',
                    'message': f"High price volatility: {market_risk['price_volatility']:.1%}"
                })
            
            if market_risk['market_stress_level'] == 'high':
                alerts.append({
                    'type': 'market_stress',
                    'severity': 'high',
                    'message': "High market stress level detected"
                })
        
        # Correlation risk alerts
        if 'correlation_risk' in risk_metrics and 'error' not in risk_metrics['correlation_risk']:
            corr_risk = risk_metrics['correlation_risk']
            
            if corr_risk['max_correlation'] > self.alert_thresholds['correlation_risk']:
                alerts.append({
                    'type': 'correlation_risk',
                    'severity': 'medium',
                    'message': f"High correlation detected: {corr_risk['max_correlation']:.1%}"
                })
        
        # Liquidity risk alerts
        if 'liquidity_risk' in risk_metrics and 'error' not in risk_metrics['liquidity_risk']:
            liq_risk = risk_metrics['liquidity_risk']
            
            if liq_risk['liquidity_score'] < self.alert_thresholds['liquidity_risk']:
                alerts.append({
                    'type': 'liquidity_risk',
                    'severity': 'medium',
                    'message': f"Low liquidity score: {liq_risk['liquidity_score']:.1%}"
                })
        
        return alerts
    
    def generate_risk_recommendations(self, risk_metrics: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        overall_score = risk_metrics.get('overall_risk_score', 0)
        
        if overall_score > 0.8:
            recommendations.append("CRITICAL: Immediate risk reduction required")
            recommendations.append("Consider closing high-risk positions")
            recommendations.append("Implement emergency risk controls")
        elif overall_score > 0.6:
            recommendations.append("HIGH RISK: Reduce position sizes")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider hedging strategies")
        elif overall_score > 0.4:
            recommendations.append("MODERATE RISK: Monitor positions closely")
            recommendations.append("Consider reducing exposure to volatile assets")
        else:
            recommendations.append("LOW RISK: Current risk levels are acceptable")
            recommendations.append("Continue normal operations")
        
        # Specific recommendations based on risk types
        if 'position_risk' in risk_metrics and 'error' not in risk_metrics['position_risk']:
            pos_risk = risk_metrics['position_risk']
            if pos_risk['position_concentration'] > 0.3:
                recommendations.append("Diversify positions to reduce concentration risk")
        
        if 'market_risk' in risk_metrics and 'error' not in risk_metrics['market_risk']:
            market_risk = risk_metrics['market_risk']
            if market_risk['price_volatility'] > 0.2:
                recommendations.append("Consider volatility hedging strategies")
        
        return recommendations
    
    def calculate_stress_test_scenarios(self, positions: pd.DataFrame, 
                                      market_data: pd.DataFrame) -> Dict:
        """Calculate stress test scenarios"""
        scenarios = {}
        
        # Price shock scenarios
        scenarios['price_shock_10pct'] = self._calculate_price_shock_impact(positions, 0.1)
        scenarios['price_shock_20pct'] = self._calculate_price_shock_impact(positions, 0.2)
        scenarios['price_shock_30pct'] = self._calculate_price_shock_impact(positions, 0.3)
        
        # Volatility shock scenarios
        scenarios['volatility_spike'] = self._calculate_volatility_spike_impact(positions, market_data)
        
        # Flow disruption scenarios
        scenarios['flow_disruption'] = self._calculate_flow_disruption_impact(positions, market_data)
        
        return scenarios
    
    def _calculate_price_shock_impact(self, positions: pd.DataFrame, shock_size: float) -> Dict:
        """Calculate impact of price shock scenario"""
        if positions.empty:
            return {'error': 'No positions data'}
        
        # Calculate P&L impact of price shock
        total_impact = 0
        for _, position in positions.iterrows():
            # Simplified P&L calculation
            position_impact = position['volume'] * shock_size * position.get('price', 0)
            total_impact += position_impact
        
        return {
            'shock_size': shock_size,
            'total_impact': total_impact,
            'impact_per_mw': total_impact / positions['volume'].sum() if positions['volume'].sum() > 0 else 0
        }
    
    def _calculate_volatility_spike_impact(self, positions: pd.DataFrame, 
                                         market_data: pd.DataFrame) -> Dict:
        """Calculate impact of volatility spike scenario"""
        # This would involve more complex calculations in practice
        return {
            'scenario': 'volatility_spike',
            'estimated_impact': 'moderate',
            'recommendation': 'Consider reducing position sizes'
        }
    
    def _calculate_flow_disruption_impact(self, positions: pd.DataFrame, 
                                        market_data: pd.DataFrame) -> Dict:
        """Calculate impact of flow disruption scenario"""
        # This would involve more complex calculations in practice
        return {
            'scenario': 'flow_disruption',
            'estimated_impact': 'high',
            'recommendation': 'Monitor cross-border capacity closely'
        }
