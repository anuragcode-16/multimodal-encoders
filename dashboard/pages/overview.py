"""
Overview Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.ui_components import (
    load_demo_data, get_asset_summary, render_metric_cards,
    render_footer, plot_candlestick, plot_sentiment_gauge
)


def render_overview_page():
    """Render overview page"""
    st.markdown("# 📊 FedFIM Overview")
    
    # Load data
    data = load_demo_data()
    summary = get_asset_summary(data)
    
    # Top metrics
    st.markdown("### Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"${summary['current_price']:.2f}",
                  f"{summary['price_change']:.2f}%")
    
    with col2:
        direction_emoji = "📈" if summary['direction'] == "UP" else "📉"
        st.metric("Prediction", f"{direction_emoji} {summary['direction']}")
    
    with col3:
        risk_color = "🔴" if summary['risk_score'] > 0.7 else "🟢"
        st.metric("Risk Score", f"{risk_color} {summary['risk_score']:.2f}")
    
    with col4:
        sentiment_label = "Bullish" if summary['sentiment_score'] > 0.3 else "Bearish"
        st.metric("Sentiment", sentiment_label)
    
    with col5:
        st.metric("Model Acc", f"{summary['accuracy']:.1%}")
    
    st.divider()
    
    # Main charts
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### Price Chart")
        fig = plot_candlestick(data['market_data'].tail(60))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### Sentiment Gauge")
        fig = plot_sentiment_gauge(summary['sentiment_score'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Prediction Confidence")
        labels = ['Up', 'Down', 'Neutral']
        colors = ['green', 'red', 'gray']
        fig = px.bar(x=labels, y=summary['direction_probs'],
                    color=labels, color_discrete_sequence=colors)
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Federated Learning Status
    st.markdown("### Federated Learning Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Clients", summary['num_clients'],
                  f"{summary['participation_rate']:.0%} participation")
    
    with col2:
        st.metric("Training Round", f"{summary['current_round']}/{summary['total_rounds']}")
    
    with col3:
        st.metric("Avg Drift Score", f"{summary['drift_score']:.3f}")
    
    # Client contributions
    st.markdown("### Client Contributions")
    contributions = data['contributions']
    
    df_contrib = pd.DataFrame({
        'Client': list(contributions.keys()),
        'Contribution': list(contributions.values())
    })
    
    fig = px.bar(df_contrib, x='Client', y='Contribution',
                color='Contribution', color_continuous_scale='Viridis')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    render_footer()


if __name__ == "__main__":
    render_overview_page()
