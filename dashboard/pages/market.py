"""
Market Analytics Page
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


def render_market_page():
    """Render market analytics page"""
    st.markdown("# 📈 Market Analytics")
    
    data = load_demo_data()
    market_data = data['market_data']
    
    # Asset selector
    col1, col2 = st.columns([3, 1])
    with col2:
        asset = st.selectbox("Select Asset", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
        timeframe = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y"])
    
    # Price and Volume Chart
    st.markdown("### Price & Volume")
    
    # Filter data based on timeframe
    n_points = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365}.get(timeframe, 365)
    df = market_data.tail(n_points)
    
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
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sma_20'],
        mode='lines', name='SMA 20',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['sma_50'],
        mode='lines', name='SMA 50',
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    # Volume
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'],
        name='Volume', marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Indicators
    st.markdown("### Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI
        st.markdown("#### RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df['date'], y=df['rsi'],
            mode='lines', name='RSI',
            line=dict(color='purple')
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # MACD
        st.markdown("#### MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df['date'], y=df['macd'],
            mode='lines', name='MACD',
            line=dict(color='blue')
        ))
        fig_macd.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_macd.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Returns and Volatility
    st.markdown("### Returns & Volatility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        st.markdown("#### Returns Distribution")
        fig = px.histogram(df, x='returns', nbins=50,
                          color_discrete_sequence=['steelblue'])
        fig.update_layout(height=300, xaxis_title='Returns', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rolling volatility
        st.markdown("#### Rolling Volatility (20d)")
        df['volatility'] = df['returns'].rolling(20).std()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['volatility'],
            mode='lines', fill='tozeroy',
            line=dict(color='red')
        ))
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("### Summary Statistics")
    
    stats = {
        'Current Price': df['close'].iloc[-1],
        'Period High': df['high'].max(),
        'Period Low': df['low'].min(),
        'Avg Volume': df['volume'].mean(),
        'Volatility': df['returns'].std() * np.sqrt(252),
        'Total Return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    }
    
    cols = st.columns(len(stats))
    for i, (key, value) in enumerate(stats.items()):
        with cols[i]:
            if 'Price' in key or 'High' in key or 'Low' in key:
                st.metric(key, f"${value:.2f}")
            elif 'Volume' in key:
                st.metric(key, f"{value/1e6:.1f}M")
            elif 'Return' in key:
                st.metric(key, f"{value:.2f}%")
            else:
                st.metric(key, f"{value:.2f}")
    
    render_footer()


if __name__ == "__main__":
    render_market_page()
