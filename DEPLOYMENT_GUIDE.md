# 🚀 Streamlit Community Cloud Deployment Guide

## Step 1: Create GitHub Repository

1. **Go to GitHub**: Visit [github.com](https://github.com) and sign in
2. **Create New Repository**: Click "New repository"
3. **Repository Settings**:
   - Name: `energy-forecasting` (or your preferred name)
   - Description: `France-Germany Energy Forecasting Tool`
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (we already have one)

## Step 2: Push Your Code to GitHub

Run these commands in your terminal:

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/energy-forecasting.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Streamlit Community Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**: Use your GitHub account
3. **Deploy New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/energy-forecasting`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - App URL: Choose a custom name (e.g., `energy-forecasting-tool`)

4. **Advanced Settings** (Optional):
   - Python version: 3.9
   - Secrets: Not needed for this app
   - Requirements: `requirements_streamlit.txt`

5. **Deploy**: Click "Deploy!"

## Step 4: Access Your Live App

Once deployed, your app will be available at:
`https://YOUR_APP_NAME.streamlit.app`

## 🔧 Troubleshooting

### Common Issues:

1. **App won't start**:
   - Check that `streamlit_app.py` is in the root directory
   - Verify `requirements_streamlit.txt` has all dependencies
   - Check the logs in Streamlit Cloud dashboard

2. **Import errors**:
   - Make sure all imports are available in `requirements_streamlit.txt`
   - Check for typos in import statements

3. **Performance issues**:
   - Use `@st.cache_data` for expensive operations
   - Optimize data loading

### Logs and Debugging:

1. **View Logs**: In Streamlit Cloud dashboard, click on your app → "Logs"
2. **Local Testing**: Always test locally first with `streamlit run streamlit_app.py`

## 📊 Your App Features

Once deployed, your app will have:

- **Real-time Energy Forecasts**: 24-48h predictions
- **Interactive Charts**: Price, flow, solar, and nuclear forecasts
- **Risk Analysis**: Volatility monitoring and alerts
- **Auto-refresh**: Automatic data updates
- **Mobile Support**: Responsive design

## 🎯 Next Steps

1. **Customize**: Modify the app for your specific needs
2. **Data Sources**: Connect to real APIs instead of synthetic data
3. **Authentication**: Add user authentication if needed
4. **Monitoring**: Set up performance monitoring
5. **Scaling**: Consider paid plans for higher usage

## 🔗 Useful Links

- [Streamlit Community Cloud](https://share.streamlit.io)
- [Streamlit Documentation](https://docs.streamlit.io)
- [GitHub Repository](https://github.com/YOUR_USERNAME/energy-forecasting)

## 📱 Mobile Access

Your app will work on:
- Desktop browsers
- Mobile phones
- Tablets
- Progressive Web App (PWA)

## 🎉 Success!

Once deployed, you'll have a professional energy forecasting tool accessible from anywhere in the world!

