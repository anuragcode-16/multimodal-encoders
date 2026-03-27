"""
Streamlit UI Components
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FEDERATED_CONFIG


def render_sidebar():
    """Render application sidebar"""
    with st.sidebar:
        st.markdown("## FedFIM Control Panel")
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Market Analytics", "Sentiment Analytics", 
             "Predictions", "Federated Training", "Personalization",
             "Drift Analysis", "Incentives"],
            index=0
        )
        
        st.divider()
        
        # Asset Selection
        st.markdown("### Asset Selection")
        asset = st.selectbox(
            "Select Asset",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
        )
        
        st.divider()
        
        # Model Settings
        st.markdown("### Model Settings")
        use_personalization = st.checkbox("Use Personalization", value=True)
        aggregation_type = st.select_slider(
            "Aggregation Type",
            options=["FedAvg", "FedProx", "FedFIM"],
            value="FedFIM"
        )
        
        st.divider()
        
        # Training Settings
        st.markdown("### Training Settings")
        num_rounds = st.slider("Federated Rounds", 10, 100, 50)
        local_epochs = st.slider("Local Epochs", 1, 10, 5)
        
        st.divider()
        
        # Data info
        st.markdown("### Data Information")
        st.info(f"Asset: {asset}\nClients: {FEDERATED_CONFIG.num_clients}\nRounds: {num_rounds}")
        
        return {
            'page': page,
            'asset': asset,
            'use_personalization': use_personalization,
            'aggregation_type': aggregation_type,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs
        }


def render_metric_cards(metrics: Dict, columns: int = 4):
    """Render metric cards"""
    cols = st.columns(columns)
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if isinstance(value, dict):
                st.metric(key, value.get('value', ''), value.get('delta', ''))
            else:
                st.metric(key, value)


def render_footer():
    """Render page footer"""
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>FedFIM v1.0 | Federated Learning for Financial Intelligence</p>
        <p>Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)


@st.cache_data
def load_demo_data() -> Dict:
    """Generate demo data for dashboard"""
    np.random.seed(42)
    
    # Market data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    n = len(dates)
    
    base_price = 150
    returns = np.random.normal(0.0003, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        'high': prices * (1 + np.abs(np.random.normal(0.005, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.normal(0.005, 0.01, n))),
        'close': prices,
        'volume': np.random.uniform(1e7, 5e7, n),
        'returns': returns
    })
    
    # Technical indicators
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['sma_50'] = market_data['close'].rolling(50).mean()
    market_data['rsi'] = 50 + np.random.normal(0, 15, n)
    market_data['macd'] = np.random.normal(0, 1, n)
    
    # Sentiment data
    sentiment_data = pd.DataFrame({
        'date': dates,
        'sentiment_score': np.random.normal(0.1, 0.3, n),
        'news_count': np.random.poisson(10, n),
        'social_mentions': np.random.poisson(100, n)
    })
    
    # Client contributions
    contributions = {
        i: np.random.uniform(0.2, 1.0) for i in range(FEDERATED_CONFIG.num_clients)
    }
    
    # Training history
    rounds = list(range(1, 51))
    training_history = {
        'round': rounds,
        'loss': [0.8 - 0.005*r + np.random.normal(0, 0.02) for r in rounds],
        'accuracy': [0.5 + 0.008*r + np.random.normal(0, 0.01) for r in rounds],
        'val_accuracy': [0.48 + 0.007*r + np.random.normal(0, 0.015) for r in rounds]
    }
    
    # Drift scores
    drift_history = {
        'round': rounds,
        'mean_drift': [0.1 + 0.2*np.sin(r/10) + np.random.normal(0, 0.05) for r in rounds],
        'max_drift': [0.2 + 0.3*np.sin(r/10) + np.random.normal(0, 0.08) for r in rounds]
    }
    
    return {
        'market_data': market_data,
        'sentiment_data': sentiment_data,
        'contributions': contributions,
        'training_history': training_history,
        'drift_history': drift_history
    }


def get_asset_summary(data: Dict) -> Dict:
    """Get summary statistics for current asset"""
    market = data['market_data']
    latest = market.iloc[-1]
    prev = market.iloc[-2]
    
    # Direction prediction
    direction_probs = [0.45, 0.25, 0.30]  # up, down, neutral
    direction = "UP" if direction_probs[0] > 0.4 else "DOWN"
    
    return {
        'current_price': latest['close'],
        'price_change': (latest['close'] - prev['close']) / prev['close'] * 100,
        'direction': direction,
        'direction_probs': direction_probs,
        'risk_score': np.random.uniform(0.3, 0.6),
        'sentiment_score': data['sentiment_data']['sentiment_score'].iloc[-1],
        'accuracy': data['training_history']['accuracy'][-1],
        'accuracy_change': 5.2,
        'num_clients': FEDERATED_CONFIG.num_clients,
        'participation_rate': 0.85,
        'current_round': 42,
        'total_rounds': 50,
        'avg_contribution': np.mean(list(data['contributions'].values())),
        'drift_score': data['drift_history']['mean_drift'][-1]
    }


def plot_candlestick(df: pd.DataFrame) -> go.Figure:
    """Create candlestick chart"""
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
    ])
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    return fig


def plot_sentiment_gauge(score: float) -> go.Figure:
    """Create sentiment gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "lightcoral"},
                {'range': [-0.3, 0.3], 'color': "lightgray"},
                {'range': [0.3, 1], 'color': "lightgreen"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig


def plot_training_curves(history: Dict) -> go.Figure:
    """Plot training curves"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history['round'],
        y=history['loss'],
        mode='lines+markers',
        name='Loss',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=history['round'],
        y=history['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='green'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
        height=400
    )
    
    return fig