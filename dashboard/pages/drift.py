"""
Drift Analytics Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.ui_components import load_demo_data, render_footer


def render_drift_page():
    """Render drift analytics page"""
    st.markdown("# 📉 Drift Analytics")
    
    data = load_demo_data()
    drift = data['drift_history']
    
    # Drift overview
    st.markdown("### Drift Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Drift", f"{drift['mean_drift'][-1]:.3f}")
    
    with col2:
        st.metric("Max Drift", f"{max(drift['max_drift']):.3f}")
    
    with col3:
        st.metric("Drift Events", "3")
    
    # Drift timeline
    st.markdown("### Drift Timeline")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drift['round'],
        y=drift['mean_drift'],
        mode='lines+markers',
        name='Mean Drift',
        line=dict(color='blue', width=2),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=drift['round'],
        y=drift['max_drift'],
        mode='lines',
        name='Max Drift',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    # Add threshold line
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                  annotation_text="Drift Threshold")
    
    fig.update_layout(
        height=400,
        xaxis_title='Round',
        yaxis_title='Drift Score'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drift distribution
    st.markdown("### Drift Distribution by Client")
    
    # Generate per-client drift
    np.random.seed(42)
    client_drifts = {
        i: np.random.uniform(0.05, 0.35) for i in range(20)
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        df_drift = pd.DataFrame({
            'Client': list(client_drifts.keys()),
            'Drift Score': list(client_drifts.values())
        })
        
        fig = px.bar(df_drift, x='Client', y='Drift Score',
                    color='Drift Score', color_continuous_scale='RdYlGn_r')
        fig.add_hline(y=0.3, line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution histogram
        fig = px.histogram(df_drift, x='Drift Score', nbins=10,
                          color_discrete_sequence=['steelblue'])
        fig.add_vline(x=0.3, line_dash="dash", line_color="red",
                      annotation_text="Threshold")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Regime detection
    st.markdown("### Market Regime Detection")
    
    # Simulate regime data
    market_data = data['market_data']
    returns = market_data['returns'].values
    
    # Compute rolling volatility
    window = 20
    volatility = pd.Series(returns).rolling(window).std().values
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('Price', 'Volatility (Regime Indicator)'))
    
    fig.add_trace(go.Scatter(
        x=market_data['date'],
        y=market_data['close'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=market_data['date'],
        y=volatility,
        mode='lines',
        name='Volatility',
        line=dict(color='red'),
        fill='tozeroy'
    ), row=2, col=1)
    
    # Mark high volatility periods
    high_vol_threshold = np.nanpercentile(volatility, 90)
    high_vol_dates = market_data['date'][volatility > high_vol_threshold]
    
    for date in high_vol_dates[::5]:  # Every 5th point for readability
        fig.add_vline(x=date, line_dash="dot", line_color="orange", opacity=0.3,
                     row=1, col=1)
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance under drift
    st.markdown("### Model Performance Under Drift")
    
    # Simulate performance comparison
    performance_data = pd.DataFrame({
        'Period': ['Pre-Drift', 'During Drift', 'Post-Drift'],
        'FedFIM': [0.85, 0.78, 0.82],
        'FedAvg': [0.82, 0.68, 0.74],
        'Centralized': [0.87, 0.65, 0.70]
    })
    
    fig = px.bar(performance_data, x='Period', y=['FedFIM', 'FedAvg', 'Centralized'],
                barmode='group', color_discrete_map={
                    'FedFIM': 'blue', 'FedAvg': 'orange', 'Centralized': 'green'
                })
    fig.update_layout(height=350, yaxis_title='Accuracy')
    st.plotly_chart(fig, use_container_width=True)
    
    render_footer()


if __name__ == "__main__":
    render_drift_page()
