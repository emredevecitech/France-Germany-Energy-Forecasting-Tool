# ⚡ France-Germany Energy Forecasting Tool

A comprehensive 24-48h forecasting tool for France-Germany cross-border flows and day-ahead prices, enabling traders to anticipate volatility from solar peaks and nuclear outages, improving risk management and profit potential in intraday markets.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📊 Features

- **Real-time Forecasting**: 24-48h energy price and flow predictions
- **Interactive Charts**: Plotly-powered visualizations
- **Risk Analysis**: Volatility monitoring and alerts
- **Auto-refresh**: Automatic data updates
- **Responsive Design**: Works on desktop and mobile

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

4. Open your browser to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

### Other Platforms

- **Heroku**: Use the included `Procfile`
- **Docker**: Build and run the container
- **AWS/GCP/Azure**: Use containerized deployment

## 📈 What You'll See

- **Price Forecasts**: Day-ahead electricity prices (€/MWh)
- **Flow Forecasts**: Cross-border electricity flows (MW)
- **Solar Generation**: Weather-based solar predictions (GW)
- **Nuclear Capacity**: Capacity and outage tracking (GW)
- **Risk Metrics**: Volatility analysis and alerts
- **Trading Insights**: Price patterns and recommendations

## 🔧 Configuration

The app uses synthetic data generation for demonstration purposes. In production, you would connect to real data sources:

- ENTSO-E API for energy data
- Weather APIs for meteorological data
- Real-time market data feeds

## 📱 Mobile Support

The app is fully responsive and works on:
- Desktop browsers
- Mobile devices
- Tablets
- Progressive Web App (PWA) compatible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:
1. Check the logs in the Streamlit interface
2. Verify all dependencies are installed
3. Ensure Python 3.8+ is being used

## 🎯 Use Cases

- **Energy Traders**: Anticipate price volatility and trading opportunities
- **Risk Managers**: Monitor portfolio exposure and market risks
- **Grid Operators**: Plan for renewable energy integration
- **Researchers**: Analyze energy market patterns and trends