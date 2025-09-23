# France-Germany Energy Forecasting Tool - Streamlit Deployment

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### 🌐 Deployment Options

#### 1. Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

#### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git add .
git commit -m "Deploy Streamlit app"
git push heroku main
```

#### 3. Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
```

#### 4. AWS/GCP/Azure
- Use containerized deployment
- Set up load balancer for production
- Configure environment variables

## 📊 Features

- **Real-time Forecasting**: 24-48h energy price and flow predictions
- **Interactive Charts**: Plotly-powered visualizations
- **Risk Analysis**: Volatility monitoring and alerts
- **Auto-refresh**: Automatic data updates
- **Responsive Design**: Works on desktop and mobile

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom port
export STREAMLIT_SERVER_PORT=8501

# Optional: Set custom host
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 📈 Performance Tips

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Data Loading**: Load data once and reuse
3. **Charts**: Use Plotly for better performance
4. **Memory**: Monitor memory usage in production

## 🚨 Troubleshooting

### Common Issues
1. **Port already in use**: Change port with `--server.port=8502`
2. **Memory issues**: Reduce data size or use pagination
3. **Slow loading**: Enable caching and optimize data processing

### Logs
```bash
# View Streamlit logs
streamlit run streamlit_app.py --logger.level=debug
```

## 🔒 Security

- Use HTTPS in production
- Set up authentication if needed
- Validate user inputs
- Monitor resource usage

## 📱 Mobile Support

The app is fully responsive and works on:
- Desktop browsers
- Mobile devices
- Tablets
- Progressive Web App (PWA) compatible

## 🎯 Production Checklist

- [ ] Set up proper logging
- [ ] Configure monitoring
- [ ] Set up backups
- [ ] Test performance
- [ ] Security review
- [ ] Documentation updated
