"""
Prediction Analytics Page
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


def render_prediction_page():
    """Render prediction analytics page"""
    st.markdown("# 🎯 Prediction Analytics")
    
    data = load_demo_data()
    market_data = data['market_data']
    
    # Prediction settings
    col1, col2 = st.columns([3, 1])
    with col2:
        horizon = st.select_slider("Prediction Horizon", options=[1, 3, 5, 10], value=1)
        model_type = st.selectbox("Model", ["FedFIM (Personalized)", "FedFIM (Global)", "Centralized"])
    
    # Current Prediction
    st.markdown("### Current Prediction")
    
    # Simulate predictions
    np.random.seed(42)
    probs = [0.45, 0.25, 0.30]  # Up, Down, Neutral
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        direction = "📈 UP" if predicted_class == 0 else "📉 DOWN" if predicted_class == 1 else "➡️ NEUTRAL"
        st.metric("Predicted Direction", direction)
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        risk_score = np.random.uniform(0.3, 0.6)
        st.metric("Risk Score", f"{risk_score:.2f}", "Medium" if 0.4 <= risk_score < 0.7 else "Low")
    
    # Prediction Probabilities
    st.markdown("### Prediction Probabilities")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        labels = ['Up', 'Down', 'Neutral']
        colors = ['green', 'red', 'gray']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=probs,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto'
        ))
        fig.update_layout(height=300, yaxis_title='Probability', yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "lightyellow"},
                    {'range': [0.7, 1], 'color': "lightcoral"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical Predictions
    st.markdown("### Historical Prediction Performance")
    
    # Simulate historical predictions
    df = market_data.tail(100).copy()
    df['actual_direction'] = df['returns'].apply(
        lambda x: 0 if x > 0.01 else (1 if x < -0.01 else 2)
    )
    df['predicted_direction'] = df['actual_direction'].apply(
        lambda x: x if np.random.random() > 0.25 else np.random.randint(0, 3)
    )
    df['correct'] = df['actual_direction'] == df['predicted_direction']
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Price with predictions
    colors = ['green' if p == 0 else 'red' if p == 1 else 'gray' 
              for p in df['predicted_direction']]
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        mode='lines', name='Price',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # Markers for predictions
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        mode='markers', name='Prediction',
        marker=dict(color=colors, size=4)
    ), row=1, col=1)
    
    # Correctness
    correct_colors = ['green' if c else 'red' for c in df['correct']]
    fig.add_trace(go.Bar(
        x=df['date'], y=[1]*len(df),
        name='Correctness',
        marker_color=correct_colors
    ), row=2, col=1)
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = df['correct'].mean()
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Precision (Up)", f"{np.random.uniform(0.7, 0.85):.1%}")
    with col3:
        st.metric("Recall (Up)", f"{np.random.uniform(0.65, 0.8):.1%}")
    with col4:
        st.metric("F1 Score", f"{np.random.uniform(0.7, 0.85):.1%}")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(df['actual_direction'], df['predicted_direction'])
    
    fig = px.imshow(cm,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Up', 'Down', 'Neutral'],
                   y=['Up', 'Down', 'Neutral'],
                   text_auto=True,
                   color_continuous_scale='Blues')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading Simulation
    st.markdown("### Simulated Trading Performance")
    
    # Calculate cumulative returns
    df['strategy_returns'] = df['returns'] * (df['predicted_direction'].replace({0: 1, 1: -1, 2: 0}))
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_buy_hold'] = (1 + df['returns']).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cumulative_strategy'],
        mode='lines', name='Strategy',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cumulative_buy_hold'],
        mode='lines', name='Buy & Hold',
        line=dict(color='blue', width=1, dash='dash')
    ))
    
    fig.update_layout(
        height=400,
        yaxis_title='Cumulative Return',
        legend=dict(x=0, y=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Financial metrics
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col2:
        st.metric("Total Return", f"{df['cumulative_strategy'].iloc[-1] - 1:.1%}")
    with col3:
        st.metric("Win Rate", f"{(df['strategy_returns'] > 0).mean():.1%}")
    
    render_footer()


if __name__ == "__main__":
    render_prediction_page()
