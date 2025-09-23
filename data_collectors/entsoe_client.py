"""
ENTSO-E API client for collecting energy market data
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class ENTSOEClient:
    """Client for interacting with ENTSO-E Transparency Platform API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.ENTSOE_API_KEY
        self.base_url = Config.ENTSOE_BASE_URL
        self.session = requests.Session()
        self.session.params = {'securityToken': self.api_key}
    
    def get_day_ahead_prices(self, country_code: str, start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """Get day-ahead electricity prices for a country"""
        try:
            params = {
                'documentType': 'A44',
                'in_Domain': country_code,
                'out_Domain': country_code,
                'periodStart': start_date.strftime('%Y%m%d%H%M'),
                'periodEnd': end_date.strftime('%Y%m%d%H%M')
            }
            
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response and convert to DataFrame
            # This is a simplified version - in practice, you'd need proper XML parsing
            data = self._parse_entsoe_response(response.text)
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching day-ahead prices: {e}")
            return pd.DataFrame()
    
    def get_cross_border_flows(self, border_ac: str, start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """Get cross-border physical flows"""
        try:
            params = {
                'documentType': 'A11',
                'in_Domain': border_ac.split('_')[0],
                'out_Domain': border_ac.split('_')[1],
                'periodStart': start_date.strftime('%Y%m%d%H%M'),
                'periodEnd': end_date.strftime('%Y%m%d%H%M')
            }
            
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = self._parse_entsoe_response(response.text)
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching cross-border flows: {e}")
            return pd.DataFrame()
    
    def get_generation_mix(self, country_code: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get generation mix data including solar, wind, nuclear, etc."""
        try:
            params = {
                'documentType': 'A75',
                'in_Domain': country_code,
                'out_Domain': country_code,
                'periodStart': start_date.strftime('%Y%m%d%H%M'),
                'periodEnd': end_date.strftime('%Y%m%d%H%M')
            }
            
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = self._parse_entsoe_response(response.text)
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching generation mix: {e}")
            return pd.DataFrame()
    
    def get_nuclear_capacity(self, country_code: str, start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """Get nuclear generation capacity and availability"""
        try:
            params = {
                'documentType': 'A77',
                'in_Domain': country_code,
                'out_Domain': country_code,
                'periodStart': start_date.strftime('%Y%m%d%H%M'),
                'periodEnd': end_date.strftime('%Y%m%d%H%M')
            }
            
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = self._parse_entsoe_response(response.text)
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching nuclear capacity: {e}")
            return pd.DataFrame()
    
    def _parse_entsoe_response(self, xml_content: str) -> List[Dict]:
        """Parse ENTSO-E XML response into structured data"""
        # This is a simplified parser - in practice, you'd use xml.etree.ElementTree
        # or lxml for proper XML parsing
        data = []
        
        # Placeholder for XML parsing logic
        # In a real implementation, you would:
        # 1. Parse the XML structure
        # 2. Extract time series data
        # 3. Convert to structured format
        
        return data
    
    def get_historical_data(self, data_type: str, country_code: str, 
                          days_back: int = 7) -> pd.DataFrame:
        """Get historical data for the specified number of days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        if data_type == 'prices':
            return self.get_day_ahead_prices(country_code, start_date, end_date)
        elif data_type == 'flows':
            return self.get_cross_border_flows(Config.BORDER_AC, start_date, end_date)
        elif data_type == 'generation':
            return self.get_generation_mix(country_code, start_date, end_date)
        elif data_type == 'nuclear':
            return self.get_nuclear_capacity(country_code, start_date, end_date)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
