"""
Federated Training Analytics Page
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
from src.config import FEDERATED_CONFIG


def render_federated_page():
    """Render federated training page"""
    st.markdown("# 🔗 Federated Training Analytics")
    
    data = load_demo_data()
    history = data['training_history']
    
    # Training progress
    st.markdown("### Training Progress")
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Training Loss', 'Accuracy'))
    
    # Loss
    fig.add_trace(go.Scatter(
        x=history['round'],
        y=history['loss'],
        mode='lines+markers',
        name='Loss',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # Accuracy
    fig.add_trace(go.Scatter(
        x=history['round'],
        y=history['accuracy'],
        mode='lines+markers',
        name='Train Acc',
        line=dict(color='green', width=2)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=history['round'],
        y=history['val_accuracy'],
        mode='lines+markers',
        name='Val Acc',
        line=dict(color='orange', width=2)
    ), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Client metrics
    st.markdown("### Client Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Contribution Scores")
        contributions = data['contributions']
        
        # Sort by contribution
        sorted_contrib = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
        
        fig = px.bar(
            x=list(sorted_contrib.keys()),
            y=list(sorted_contrib.values()),
            labels={'x': 'Client ID', 'y': 'Contribution'},
            color=list(sorted_contrib.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Aggregation Weights")
        
        # Simulated aggregation weights
        weights = {k: v/sum(contributions.values()) for k, v in contributions.items()}
        
        fig = go.Figure(data=[go.Pie(
            labels=[f'Client {k}' for k in list(weights.keys())[:10]],
            values=list(weights.values())[:10],
            hole=.4
        )])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Communication metrics
    st.markdown("### Communication Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Estimate communication cost
    model_size_mb = 2.5  # Approximate model size
    total_rounds = len(history['round'])
    num_clients = FEDERATED_CONFIG.num_clients
    
    with col1:
        total_comm = 2 * model_size_mb * num_clients * total_rounds
        st.metric("Total Communication", f"{total_comm:.1f} MB")
    
    with col2:
        per_round = 2 * model_size_mb * num_clients
        st.metric("Per Round", f"{per_round:.1f} MB")
    
    with col3:
        st.metric("Participation Rate", f"{0.85:.0%}")
    
    with col4:
        st.metric("Convergence Round", "35")
    
    # Client performance table
    st.markdown("### Client Performance Summary")
    
    client_data = []
    for client_id in range(min(20, FEDERATED_CONFIG.num_clients)):
        client_data.append({
            'Client ID': client_id,
            'Data Size': np.random.randint(500, 2000),
            'Train Acc': np.random.uniform(0.7, 0.9),
            'Val Acc': np.random.uniform(0.65, 0.85),
            'Drift Score': np.random.uniform(0.05, 0.3),
            'Contribution': contributions.get(client_id, 0),
            'Behavior Type': np.random.choice(['Conservative', 'Aggressive', 'Momentum', 'Long-term'])
        })
    
    df_clients = pd.DataFrame(client_data)
    st.dataframe(df_clients.style.format({
        'Train Acc': '{:.2%}',
        'Val Acc': '{:.2%}',
        'Drift Score': '{:.3f}',
        'Contribution': '{:.3f}'
    }).background_gradient(subset=['Contribution'], cmap='Viridis'), 
    use_container_width=True)
    
    render_footer()


if __name__ == "__main__":
    render_federated_page()
