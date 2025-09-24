#!/usr/bin/env python3
"""
Test script to verify data loading works
"""
import pandas as pd
import numpy as np
from datetime import datetime

def test_data_loading():
    print("🔍 Testing Data Loading...")
    
    try:
        # Test the new CSV format
        print("📊 Testing new CSV format...")
        df = pd.read_csv("GUI_ENERGY_PRICES_202509222200-202509232200.csv")
        print(f"✅ CSV loaded: {len(df)} rows")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Sample data:")
        print(df.head(3))
        
        # Clean column names
        df.columns = ['datetime', 'area', 'sequence', 'day_ahead_price', 'intraday_period', 'intraday_price']
        
        # Parse datetime
        df['datetime'] = df['datetime'].str.extract(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})')[0]
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')
        
        # Remove rows with missing data
        df = df.dropna(subset=['day_ahead_price'])
        
        # Convert price to numeric
        df['day_ahead_price'] = pd.to_numeric(df['day_ahead_price'], errors='coerce')
        df = df.dropna(subset=['day_ahead_price'])
        
        print(f"✅ Cleaned data: {len(df)} rows")
        print(f"💰 Price range: {df['day_ahead_price'].min():.2f} - {df['day_ahead_price'].max():.2f} €/MWh")
        print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loading()
