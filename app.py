"""
FedFIM Dashboard - Main Application
"""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import CHECKPOINTS_DIR, DATA_CONFIG
from src.utils.helpers import load_json
from dashboard.ui_components import (
    render_sidebar, render_metric_cards, render_footer,
    load_demo_data, get_asset_summary
)

# Page config
st.set_page_config(
    page_title="FedFIM Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .stMetric {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar
    render_sidebar()
    
    # Main content
    st.markdown('<div class="main-header">🏦 FedFIM Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Drift-Aware, Incentive-Compatible, Multimodal Personalized Federated Learning for Financial Intelligence")
    
    # Load data
    demo_data = load_demo_data()
    summary = get_asset_summary(demo_data)
    
    # Quick metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${summary['current_price']:.2f}", 
                  f"{summary['price_change']:.2f}%")
    
    with col2:
        st.metric("Predicted Direction", summary['direction'], 
                  delta_color="normal" if summary['direction'] == "UP" else "inverse")
    
    with col3:
        st.metric("Risk Score", f"{summary['risk_score']:.2f}", 
                  "High" if summary['risk_score'] > 0.7 else "Low")
    
    with col4:
        st.metric("Model Accuracy", f"{summary['accuracy']:.2%}", 
                  f"+{summary['accuracy_change']:.2f}%")
    
    st.divider()
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### Price Chart")
        
        # Price chart
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        df = demo_data['market_data']
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        fig.update_layout(
            height=400,
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("#### Model Predictions")
        
        # Prediction probabilities
        labels = ['Up', 'Down', 'Neutral']
        probs = summary['direction_probs']
        
        import plotly.express as px
        fig_prob = px.bar(x=labels, y=probs, 
                         color=labels,
                         color_discrete_map={'Up': 'green', 'Down': 'red', 'Neutral': 'gray'})
        fig_prob.update_layout(height=200, showlegend=False, 
                              yaxis_title="Probability",
                              xaxis_title="")
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Sentiment indicator
        st.markdown("#### Market Sentiment")
        sentiment = summary['sentiment_score']
        sentiment_label = "Bullish" if sentiment > 0.3 else "Bearish" if sentiment < -0.3 else "Neutral"
        sentiment_color = "green" if sentiment > 0.3 else "red" if sentiment < -0.3 else "gray"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f5f5f5; border-radius: 10px;">
            <h3 style="color: {sentiment_color}; margin: 0;">{sentiment_label}</h3>
            <p style="margin: 0;">Score: {sentiment:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Federated Learning Status
    st.markdown("#### Federated Learning Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Clients", summary['num_clients'], 
                  f"Participation: {summary['participation_rate']:.0%}")
    
    with col2:
        st.metric("Training Round", summary['current_round'], 
                  f"of {summary['total_rounds']}")
    
    with col3:
        st.metric("Avg Contribution", f"{summary['avg_contribution']:.2f}",
                  "Drift Score: {:.2f}".format(summary['drift_score']))
    
    # Client contribution chart
    st.markdown("#### Client Contribution Distribution")
    
    contributions = demo_data['contributions']
    fig_contrib = px.bar(
        x=list(contributions.keys()),
        y=list(contributions.values()),
        labels={'x': 'Client ID', 'y': 'Contribution Score'},
        color=list(contributions.values()),
        color_continuous_scale='Viridis'
    )
    fig_contrib.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()