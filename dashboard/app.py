"""
Interactive dashboard for energy traders
"""
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from forecasting_engine import ForecastingEngine
from config import Config

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "France-Germany Energy Forecasting Dashboard"

# Initialize forecasting engine
forecasting_engine = ForecastingEngine()

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("France-Germany Energy Forecasting Dashboard", 
                   className="text-center mb-4"),
            html.P("24-48h forecasting tool for cross-border flows and day-ahead prices", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Status and alerts section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("System Status"),
                dbc.CardBody([
                    html.Div(id="system-status"),
                    html.Div(id="alerts-section")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Main forecasting section
    dbc.Row([
        # Price forecast
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Day-Ahead Price Forecast"),
                dbc.CardBody([
                    dcc.Graph(id="price-forecast-chart")
                ])
            ])
        ], width=6),
        
        # Flow forecast
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Cross-Border Flow Forecast"),
                dbc.CardBody([
                    dcc.Graph(id="flow-forecast-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Solar and nuclear forecast
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Solar Generation Forecast"),
                dbc.CardBody([
                    dcc.Graph(id="solar-forecast-chart")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Nuclear Capacity Forecast"),
                dbc.CardBody([
                    dcc.Graph(id="nuclear-forecast-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Volatility and risk analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Volatility Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="volatility-chart")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Assessment"),
                dbc.CardBody([
                    html.Div(id="risk-metrics"),
                    html.Div(id="risk-recommendations")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Control panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Control Panel"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Generate Forecast", id="generate-forecast-btn", 
                                     color="primary", className="me-2"),
                            dbc.Button("Refresh Data", id="refresh-data-btn", 
                                     color="secondary", className="me-2"),
                            dbc.Button("Clear Alerts", id="clear-alerts-btn", 
                                     color="warning")
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Forecast Horizon (hours):"),
                            dcc.Slider(
                                id="forecast-horizon-slider",
                                min=24, max=72, step=6, value=48,
                                marks={i: f"{i}h" for i in range(24, 73, 12)}
                            )
                        ], className="mt-3")
                    ])
                ])
            ])
        ], width=12)
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # 5 minutes
        n_intervals=0
    )
], fluid=True)

# Callbacks
@app.callback(
    [Output("system-status", "children"),
     Output("alerts-section", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_system_status(n_intervals):
    """Update system status and alerts"""
    try:
        # Get latest forecast
        forecast = forecasting_engine.get_latest_forecast()
        alerts = forecasting_engine.get_alerts()
        
        # System status
        if forecast and 'error' not in forecast:
            status_color = "success"
            status_text = "System Operational"
            last_update = forecast.get('timestamp', datetime.now())
            status_children = [
                dbc.Badge(status_text, color=status_color, className="me-2"),
                html.Small(f"Last update: {last_update.strftime('%H:%M:%S')}")
            ]
        else:
            status_color = "danger"
            status_text = "System Error"
            status_children = [
                dbc.Badge(status_text, color=status_color, className="me-2"),
                html.Small("Check system logs")
            ]
        
        # Alerts section
        if alerts:
            alert_items = []
            for alert in alerts[-5:]:  # Show last 5 alerts
                severity_color = {
                    'high': 'danger',
                    'medium': 'warning',
                    'low': 'info'
                }.get(alert.get('severity', 'low'), 'info')
                
                alert_items.append(
                    dbc.Alert(
                        alert.get('message', 'Unknown alert'),
                        color=severity_color,
                        dismissable=True,
                        className="mb-2"
                    )
                )
            
            alerts_children = [
                html.H6("Recent Alerts"),
                html.Div(alert_items)
            ]
        else:
            alerts_children = [
                html.H6("No Recent Alerts"),
                html.Small("System running normally")
            ]
        
        return status_children, alerts_children
        
    except Exception as e:
        error_children = [
            dbc.Alert(f"Error updating status: {str(e)}", color="danger")
        ]
        return error_children, []

@app.callback(
    Output("price-forecast-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("generate-forecast-btn", "n_clicks")]
)
def update_price_forecast(n_intervals, n_clicks):
    """Update price forecast chart"""
    try:
        forecast = forecasting_engine.get_latest_forecast()
        
        if not forecast or 'price_forecast' not in forecast:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No price forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        price_data = forecast['price_forecast']
        
        if price_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No price forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create price forecast chart
        fig = go.Figure()
        
        # Add predicted price
        fig.add_trace(go.Scatter(
            x=price_data['datetime'],
            y=price_data['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add confidence interval if available
        if 'price_confidence' in price_data.columns:
            confidence = price_data['price_confidence'] * 10  # Scale for visibility
            fig.add_trace(go.Scatter(
                x=price_data['datetime'],
                y=price_data['predicted_price'] + confidence,
                mode='lines',
                name='Confidence Upper',
                line=dict(color='rgba(0,100,80,0.2)'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=price_data['datetime'],
                y=price_data['predicted_price'] - confidence,
                mode='lines',
                name='Confidence Lower',
                line=dict(color='rgba(0,100,80,0.2)'),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Day-Ahead Price Forecast (€/MWh)",
            xaxis_title="Time",
            yaxis_title="Price (€/MWh)",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading price forecast: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    Output("flow-forecast-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("generate-forecast-btn", "n_clicks")]
)
def update_flow_forecast(n_intervals, n_clicks):
    """Update cross-border flow forecast chart"""
    try:
        forecast = forecasting_engine.get_latest_forecast()
        
        if not forecast or 'flow_forecast' not in forecast:
            fig = go.Figure()
            fig.add_annotation(
                text="No flow forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        flow_data = forecast['flow_forecast']
        
        if flow_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No flow forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create flow forecast chart
        fig = go.Figure()
        
        # Add predicted flow
        fig.add_trace(go.Scatter(
            x=flow_data['datetime'],
            y=flow_data['predicted_flow'],
            mode='lines+markers',
            name='Predicted Flow',
            line=dict(color='green', width=2)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Cross-Border Flow Forecast (MW)",
            xaxis_title="Time",
            yaxis_title="Flow (MW)",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading flow forecast: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    Output("solar-forecast-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("generate-forecast-btn", "n_clicks")]
)
def update_solar_forecast(n_intervals, n_clicks):
    """Update solar generation forecast chart"""
    try:
        forecast = forecasting_engine.get_latest_forecast()
        
        if not forecast or 'solar_forecast' not in forecast:
            fig = go.Figure()
            fig.add_annotation(
                text="No solar forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        solar_data = forecast['solar_forecast']
        
        if solar_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No solar forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create solar forecast chart
        fig = go.Figure()
        
        # Add predicted solar generation
        fig.add_trace(go.Scatter(
            x=solar_data['datetime'],
            y=solar_data['predicted_solar_generation'],
            mode='lines+markers',
            name='Predicted Solar',
            line=dict(color='orange', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Solar Generation Forecast (GW)",
            xaxis_title="Time",
            yaxis_title="Generation (GW)",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading solar forecast: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    Output("nuclear-forecast-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("generate-forecast-btn", "n_clicks")]
)
def update_nuclear_forecast(n_intervals, n_clicks):
    """Update nuclear capacity forecast chart"""
    try:
        forecast = forecasting_engine.get_latest_forecast()
        
        if not forecast or 'nuclear_forecast' not in forecast:
            fig = go.Figure()
            fig.add_annotation(
                text="No nuclear forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        nuclear_data = forecast['nuclear_forecast']
        
        if nuclear_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No nuclear forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create nuclear forecast chart
        fig = go.Figure()
        
        # Add forecasted capacity
        fig.add_trace(go.Scatter(
            x=nuclear_data['datetime'],
            y=nuclear_data['forecasted_capacity'],
            mode='lines+markers',
            name='Forecasted Capacity',
            line=dict(color='red', width=2)
        ))
        
        # Add outage capacity if available
        if 'outage_capacity' in nuclear_data.columns:
            fig.add_trace(go.Scatter(
                x=nuclear_data['datetime'],
                y=nuclear_data['outage_capacity'],
                mode='lines+markers',
                name='Outage Capacity',
                line=dict(color='darkred', width=2),
                fill='tonexty'
            ))
        
        fig.update_layout(
            title="Nuclear Capacity Forecast (GW)",
            xaxis_title="Time",
            yaxis_title="Capacity (GW)",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading nuclear forecast: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    Output("volatility-chart", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("generate-forecast-btn", "n_clicks")]
)
def update_volatility_chart(n_intervals, n_clicks):
    """Update volatility analysis chart"""
    try:
        forecast = forecasting_engine.get_latest_forecast()
        
        if not forecast or 'volatility_forecast' not in forecast:
            fig = go.Figure()
            fig.add_annotation(
                text="No volatility forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        volatility_data = forecast['volatility_forecast']
        
        if volatility_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No volatility forecast data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create volatility chart
        fig = go.Figure()
        
        # Add forecasted volatility
        fig.add_trace(go.Scatter(
            x=volatility_data['datetime'],
            y=volatility_data['forecasted_volatility'],
            mode='lines+markers',
            name='Forecasted Volatility',
            line=dict(color='purple', width=2)
        ))
        
        # Add volatility thresholds
        fig.add_hline(y=Config.VOLATILITY_THRESHOLD, line_dash="dash", 
                     line_color="orange", annotation_text="Medium Threshold")
        fig.add_hline(y=Config.HIGH_VOLATILITY_THRESHOLD, line_dash="dash", 
                     line_color="red", annotation_text="High Threshold")
        
        fig.update_layout(
            title="Price Volatility Forecast",
            xaxis_title="Time",
            yaxis_title="Volatility",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading volatility forecast: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@app.callback(
    [Output("risk-metrics", "children"),
     Output("risk-recommendations", "children")],
    [Input("interval-component", "n_intervals"),
     Input("generate-forecast-btn", "n_clicks")]
)
def update_risk_assessment(n_intervals, n_clicks):
    """Update risk assessment"""
    try:
        forecast = forecasting_engine.get_latest_forecast()
        
        if not forecast or 'risk_assessment' not in forecast:
            return ["No risk data available"], ["No recommendations available"]
        
        risk_data = forecast['risk_assessment']
        
        # Risk metrics
        metrics_children = []
        for key, value in risk_data.items():
            if isinstance(value, (int, float)):
                metrics_children.append(
                    html.Div([
                        html.Strong(f"{key.replace('_', ' ').title()}: "),
                        html.Span(f"{value:.3f}")
                    ], className="mb-2")
                )
        
        # Risk recommendations (placeholder)
        recommendations = [
            "Monitor price volatility closely",
            "Consider hedging strategies for high-risk periods",
            "Diversify position sizes during uncertain times"
        ]
        
        recommendations_children = [
            html.H6("Risk Recommendations"),
            html.Ul([html.Li(rec) for rec in recommendations])
        ]
        
        return metrics_children, recommendations_children
        
    except Exception as e:
        return [f"Error loading risk data: {str(e)}"], ["Error loading recommendations"]

@app.callback(
    Output("generate-forecast-btn", "n_clicks"),
    [Input("generate-forecast-btn", "n_clicks")]
)
def generate_forecast_callback(n_clicks):
    """Generate new forecast when button is clicked"""
    if n_clicks:
        try:
            forecasting_engine.generate_forecast()
        except Exception as e:
            logging.error(f"Error generating forecast: {e}")
    return n_clicks

@app.callback(
    Output("clear-alerts-btn", "n_clicks"),
    [Input("clear-alerts-btn", "n_clicks")]
)
def clear_alerts_callback(n_clicks):
    """Clear alerts when button is clicked"""
    if n_clicks:
        try:
            forecasting_engine.clear_alerts()
        except Exception as e:
            logging.error(f"Error clearing alerts: {e}")
    return n_clicks

if __name__ == "__main__":
    # Initialize forecasting engine
    if forecasting_engine.initialize():
        print("Forecasting engine initialized successfully")
    else:
        print("Error initializing forecasting engine")
    
    # Start monitoring
    forecasting_engine.start_monitoring()
    
    # Run dashboard
    app.run(debug=True, host='0.0.0.0', port=8050)
