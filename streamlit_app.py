"""
Streamlit version of the France-Germany Energy Forecasting Tool
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="France-Germany Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_forecast_data():
    """Generate synthetic forecast data"""
    # Create 48-hour forecast
    start_time = datetime.now()
    forecast_times = pd.date_range(start=start_time, periods=48, freq='h')
    
    # Generate price forecast
    price_data = []
    for dt in forecast_times:
        base_price = 50
        daily_pattern = 20 * np.sin(2 * np.pi * dt.hour / 24)
        weekly_pattern = 5 * np.sin(2 * np.pi * dt.weekday() / 7)
        trend = np.random.normal(0, 0.5)
        noise = np.random.normal(0, 8)
        price = base_price + daily_pattern + weekly_pattern + trend + noise
        
        price_data.append({
            'datetime': dt,
            'predicted_price': max(0, price),
            'price_confidence': np.random.uniform(0.7, 0.9)
        })
    
    # Generate flow forecast
    flow_data = []
    for dt in forecast_times:
        base_flow = 150 * np.sin(2 * np.pi * dt.hour / 24)
        noise = np.random.normal(0, 30)
        flow = base_flow + noise
        
        flow_data.append({
            'datetime': dt,
            'predicted_flow': flow,
            'flow_confidence': np.random.uniform(0.8, 0.95)
        })
    
    # Generate solar forecast
    solar_data = []
    for dt in forecast_times:
        if 6 <= dt.hour <= 18:
            solar = 25 * np.sin(np.pi * (dt.hour - 6) / 12)
        else:
            solar = 0
        
        solar_data.append({
            'datetime': dt,
            'predicted_solar_generation': max(0, solar),
            'solar_confidence': np.random.uniform(0.85, 0.95)
        })
    
    # Generate nuclear forecast
    nuclear_data = []
    for dt in forecast_times:
        base_capacity = 60
        variation = np.random.normal(0, 3)
        capacity = base_capacity + variation
        
        # Simulate occasional outages
        if np.random.random() < 0.05:  # 5% chance of outage
            outage = np.random.uniform(5, 15)
            capacity -= outage
        
        nuclear_data.append({
            'datetime': dt,
            'forecasted_capacity': max(0, capacity),
            'outage_capacity': max(0, 60 - capacity)
        })
    
    # Generate volatility forecast
    volatility_data = []
    for dt in forecast_times:
        base_volatility = 0.1
        daily_pattern = 0.05 * np.sin(2 * np.pi * dt.hour / 24)
        noise = np.random.normal(0, 0.02)
        volatility = base_volatility + daily_pattern + noise
        
        volatility_data.append({
            'datetime': dt,
            'forecasted_volatility': max(0, volatility),
            'volatility_confidence': np.random.uniform(0.6, 0.8)
        })
    
    return {
        'price_forecast': pd.DataFrame(price_data),
        'flow_forecast': pd.DataFrame(flow_data),
        'solar_forecast': pd.DataFrame(solar_data),
        'nuclear_forecast': pd.DataFrame(nuclear_data),
        'volatility_forecast': pd.DataFrame(volatility_data)
    }

def generate_alerts():
    """Generate alerts"""
    alerts = []
    
    # Randomly generate alerts
    if np.random.random() < 0.3:  # 30% chance of price volatility alert
        alerts.append({
            'type': 'price_volatility',
            'severity': 'medium',
            'message': f'Price volatility: {np.random.uniform(15, 25):.1f}%',
            'datetime': datetime.now()
        })
    
    if np.random.random() < 0.2:  # 20% chance of solar peak alert
        alerts.append({
            'type': 'solar_peak',
            'severity': 'low',
            'message': 'Solar generation peak expected at 13:00',
            'datetime': datetime.now()
        })
    
    if np.random.random() < 0.15:  # 15% chance of nuclear alert
        alerts.append({
            'type': 'nuclear_outage',
            'severity': 'high',
            'message': f'Nuclear capacity reduced by {np.random.uniform(5, 15):.1f} GW',
            'datetime': datetime.now()
        })
    
    return alerts

def create_price_chart(price_data):
    """Create price forecast chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_data['datetime'],
        y=price_data['predicted_price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=3),
        hovertemplate='<b>%{x}</b><br>Price: %{y:.1f} €/MWh<extra></extra>'
    ))
    
    fig.update_layout(
        title="Day-Ahead Price Forecast (€/MWh)",
        xaxis_title="Time",
        yaxis_title="Price (€/MWh)",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig

def create_flow_chart(flow_data):
    """Create flow forecast chart"""
    fig = go.Figure()
    
    # Color based on flow direction
    colors = ['green' if x > 0 else 'red' for x in flow_data['predicted_flow']]
    
    fig.add_trace(go.Scatter(
        x=flow_data['datetime'],
        y=flow_data['predicted_flow'],
        mode='lines+markers',
        name='Predicted Flow',
        line=dict(width=3),
        marker=dict(color=colors),
        hovertemplate='<b>%{x}</b><br>Flow: %{y:.1f} MW<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Cross-Border Flow Forecast (MW)",
        xaxis_title="Time",
        yaxis_title="Flow (MW)",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig

def create_solar_chart(solar_data):
    """Create solar forecast chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=solar_data['datetime'],
        y=solar_data['predicted_solar_generation'],
        mode='lines+markers',
        name='Predicted Solar',
        line=dict(color='orange', width=3),
        fill='tozeroy',
        hovertemplate='<b>%{x}</b><br>Solar: %{y:.1f} GW<extra></extra>'
    ))
    
    fig.update_layout(
        title="Solar Generation Forecast (GW)",
        xaxis_title="Time",
        yaxis_title="Generation (GW)",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig

def create_nuclear_chart(nuclear_data):
    """Create nuclear forecast chart"""
    fig = go.Figure()
    
    # Add forecasted capacity
    fig.add_trace(go.Scatter(
        x=nuclear_data['datetime'],
        y=nuclear_data['forecasted_capacity'],
        mode='lines+markers',
        name='Forecasted Capacity',
        line=dict(color='red', width=3),
        hovertemplate='<b>%{x}</b><br>Capacity: %{y:.1f} GW<extra></extra>'
    ))
    
    # Add outage capacity if available
    if 'outage_capacity' in nuclear_data.columns:
        fig.add_trace(go.Scatter(
            x=nuclear_data['datetime'],
            y=nuclear_data['outage_capacity'],
            mode='lines+markers',
            name='Outage Capacity',
            line=dict(color='darkred', width=2),
            fill='tonexty',
            hovertemplate='<b>%{x}</b><br>Outage: %{y:.1f} GW<extra></extra>'
        ))
    
    fig.update_layout(
        title="Nuclear Capacity Forecast (GW)",
        xaxis_title="Time",
        yaxis_title="Capacity (GW)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_volatility_chart(volatility_data):
    """Create volatility chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volatility_data['datetime'],
        y=volatility_data['forecasted_volatility'],
        mode='lines+markers',
        name='Forecasted Volatility',
        line=dict(color='purple', width=3),
        hovertemplate='<b>%{x}</b><br>Volatility: %{y:.1%}<extra></extra>'
    ))
    
    # Add volatility thresholds
    fig.add_hline(y=0.15, line_dash="dash", 
                 line_color="orange", annotation_text="Medium Threshold")
    fig.add_hline(y=0.25, line_dash="dash", 
                 line_color="red", annotation_text="High Threshold")
    
    fig.update_layout(
        title="Price Volatility Forecast",
        xaxis_title="Time",
        yaxis_title="Volatility",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">⚡ France-Germany Energy Forecasting Tool</h1>', unsafe_allow_html=True)
    st.markdown("### 24-48h forecasting for cross-border flows and day-ahead prices")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 minutes)", value=True)
    if auto_refresh:
        st.sidebar.info("🔄 Auto-refresh enabled")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Generate New Forecast", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Forecast horizon selector
    forecast_hours = st.sidebar.slider("Forecast Horizon (hours)", 24, 72, 48)
    
    # Generate forecast data
    with st.spinner("🔮 Generating forecast data..."):
        forecast_data = generate_forecast_data()
    
    # Generate alerts
    alerts = generate_alerts()
    
    # Status section
    st.markdown("## 📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Operational", "Last update: " + datetime.now().strftime("%H:%M:%S"))
    
    with col2:
        price_avg = forecast_data['price_forecast']['predicted_price'].mean()
        st.metric("Avg Price", f"{price_avg:.1f} €/MWh", f"Range: {forecast_data['price_forecast']['predicted_price'].min():.1f}-{forecast_data['price_forecast']['predicted_price'].max():.1f}")
    
    with col3:
        flow_avg = forecast_data['flow_forecast']['predicted_flow'].mean()
        st.metric("Avg Flow", f"{flow_avg:.1f} MW", f"Direction: {'France→Germany' if flow_avg > 0 else 'Germany→France'}")
    
    with col4:
        solar_avg = forecast_data['solar_forecast']['predicted_solar_generation'].mean()
        st.metric("Avg Solar", f"{solar_avg:.1f} GW", f"Peak: {forecast_data['solar_forecast']['predicted_solar_generation'].max():.1f} GW")
    
    # Alerts section
    if alerts:
        st.markdown("## 🚨 Active Alerts")
        for alert in alerts:
            severity = alert['severity']
            if severity == 'high':
                st.markdown(f'<div class="alert-high"><strong>🔴 {alert["message"]}</strong></div>', unsafe_allow_html=True)
            elif severity == 'medium':
                st.markdown(f'<div class="alert-medium"><strong>🟡 {alert["message"]}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low"><strong>🟢 {alert["message"]}</strong></div>', unsafe_allow_html=True)
    else:
        st.success("✅ No active alerts")
    
    # Main forecasting section
    st.markdown("## 📈 Energy Forecasts")
    
    # Price and Flow forecasts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_price_chart(forecast_data['price_forecast']), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_flow_chart(forecast_data['flow_forecast']), use_container_width=True)
    
    # Solar and Nuclear forecasts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_solar_chart(forecast_data['solar_forecast']), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_nuclear_chart(forecast_data['nuclear_forecast']), use_container_width=True)
    
    # Volatility analysis
    st.markdown("## ⚠️ Risk Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_volatility_chart(forecast_data['volatility_forecast']), use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Risk Metrics")
        
        # Calculate risk metrics
        price_volatility = forecast_data['price_forecast']['predicted_price'].std() / forecast_data['price_forecast']['predicted_price'].mean()
        flow_volatility = forecast_data['flow_forecast']['predicted_flow'].std() / abs(forecast_data['flow_forecast']['predicted_flow']).mean()
        
        st.metric("Price Volatility", f"{price_volatility:.1%}")
        st.metric("Flow Volatility", f"{flow_volatility:.1%}")
        
        # Risk level
        if price_volatility > 0.25:
            st.error("🔴 High Risk")
        elif price_volatility > 0.15:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = [
            "Monitor price volatility closely",
            "Consider hedging strategies for high-risk periods",
            "Diversify position sizes during uncertain times"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # Data table
    st.markdown("## 📋 Forecast Data")
    
    # Create summary table
    summary_data = {
        'Metric': ['Average Price', 'Price Range', 'Average Flow', 'Average Solar', 'Average Nuclear', 'Price Volatility'],
        'Value': [
            f"{forecast_data['price_forecast']['predicted_price'].mean():.1f} €/MWh",
            f"{forecast_data['price_forecast']['predicted_price'].min():.1f} - {forecast_data['price_forecast']['predicted_price'].max():.1f} €/MWh",
            f"{forecast_data['flow_forecast']['predicted_flow'].mean():.1f} MW",
            f"{forecast_data['solar_forecast']['predicted_solar_generation'].mean():.1f} GW",
            f"{forecast_data['nuclear_forecast']['forecasted_capacity'].mean():.1f} GW",
            f"{price_volatility:.1%}"
        ]
    }
    
    st.table(pd.DataFrame(summary_data))
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 France-Germany Energy Forecasting Tool")
    st.markdown("**Purpose**: Anticipate volatility from solar peaks and nuclear outages, improving risk management and profit potential in intraday markets.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()

if __name__ == "__main__":
    main()
