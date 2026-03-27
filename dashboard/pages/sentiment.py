"""
Sentiment Analytics Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from collections import Counter
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.ui_components import load_demo_data, render_footer


def render_sentiment_page():
    """Render sentiment analytics page"""
    st.markdown("# 📰 Sentiment Analytics")
    
    data = load_demo_data()
    sentiment_data = data['sentiment_data']
    market_data = data['market_data']
    
    # Sentiment Overview
    st.markdown("### Sentiment Overview")
    
    latest = sentiment_data.iloc[-1]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_label = "Bullish" if latest['sentiment_score'] > 0.2 else "Bearish" if latest['sentiment_score'] < -0.2 else "Neutral"
        st.metric("Current Sentiment", sentiment_label, f"{latest['sentiment_score']:.2f}")
    
    with col2:
        st.metric("News Count (24h)", int(latest['news_count']))
    
    with col3:
        st.metric("Social Mentions", int(latest['social_mentions']))
    
    # Sentiment Trend
    st.markdown("### Sentiment Trend")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Sentiment score
    fig.add_trace(go.Scatter(
        x=sentiment_data['date'],
        y=sentiment_data['sentiment_score'],
        mode='lines', name='Sentiment',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    
    # News volume
    fig.add_trace(go.Bar(
        x=sentiment_data['date'],
        y=sentiment_data['news_count'],
        name='News Volume',
        marker_color='lightblue'
    ), row=2, col=1)
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Distribution
    st.markdown("### Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Classify sentiment
        sentiment_data['label'] = sentiment_data['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )
        
        dist = sentiment_data['label'].value_counts()
        
        fig = px.pie(values=dist.values, names=dist.index,
                    color=dist.index,
                    color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'},
                    hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Histogram
        fig = px.histogram(sentiment_data, x='sentiment_score', nbins=30,
                          color_discrete_sequence=['steelblue'])
        fig.update_layout(height=350, xaxis_title='Sentiment Score')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment vs Price
    st.markdown("### Sentiment vs Price Correlation")
    
    # Merge data
    merged = pd.merge(sentiment_data, market_data[['date', 'close', 'returns']], 
                     on='date', how='inner')
    
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(go.Scatter(
        x=merged['sentiment_score'],
        y=merged['returns'],
        mode='markers',
        marker=dict(
            color=merged['returns'],
            colorscale='RdYlGn',
            size=5,
            opacity=0.6
        ),
        name='Daily Returns'
    ))
    
    # Add trendline
    z = np.polyfit(merged['sentiment_score'].fillna(0), merged['returns'].fillna(0), 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['sentiment_score'].min(), merged['sentiment_score'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode='lines', name='Trend',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title='Sentiment Score',
        yaxis_title='Daily Returns'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation value
    corr = merged['sentiment_score'].corr(merged['returns'])
    st.info(f"Correlation between sentiment and returns: **{corr:.3f}**")
    
    # Topic Trends (Simulated)
    st.markdown("### Trending Topics")
    
    topics = pd.DataFrame({
        'Topic': ['Earnings', 'Product Launch', 'Regulation', 'Market Outlook', 
                 'Dividends', 'M&A Activity'],
        'Mentions': [245, 189, 156, 134, 98, 67],
        'Sentiment': [0.4, 0.6, -0.3, 0.2, 0.5, 0.1]
    })
    
    fig = px.bar(topics, x='Topic', y='Mentions',
                color='Sentiment', color_continuous_scale='RdYlGn')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # Word Frequency (Simplified Word Cloud)
    st.markdown("### Word Frequency")
    
    # Simulate word data
    words_data = {
        'growth': 120, 'revenue': 98, 'market': 87, 'profit': 76,
        'earnings': 65, 'stock': 54, 'dividend': 43, 'volatility': 32,
        'bullish': 28, 'bearish': 24, 'trend': 21, 'forecast': 18
    }
    
    fig = px.bar(x=list(words_data.keys()), y=list(words_data.values()),
                color=list(words_data.values()), color_continuous_scale='Blues')
    fig.update_layout(height=350, xaxis_title='', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)
    
    render_footer()


if __name__ == "__main__":
    render_sentiment_page()
