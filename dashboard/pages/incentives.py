"""
Incentive Analytics Page
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


def render_incentives_page():
    """Render incentives page"""
    st.markdown("# 💰 Incentive & Trust Analytics")
    
    data = load_demo_data()
    contributions = data['contributions']
    
    # Incentive overview
    st.markdown("### Incentive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reward Pool", "1000 tokens")
    
    with col2:
        st.metric("Avg Contribution", f"{np.mean(list(contributions.values())):.3f}")
    
    with col3:
        st.metric("Fairness Score", "0.87")
    
    with col4:
        st.metric("Free Riders Detected", "2")
    
    # Contribution vs Reward
    st.markdown("### Contribution vs Reward Distribution")
    
    # Generate rewards based on contributions
    total_contrib = sum(contributions.values())
    rewards = {k: (v / total_contrib) * 1000 for k, v in contributions.items()}
    
    df = pd.DataFrame({
        'Client': list(contributions.keys()),
        'Contribution': list(contributions.values()),
        'Reward': list(rewards.values())
    })
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Contribution Scores', 'Reward Distribution'))
    
    fig.add_trace(go.Bar(
        x=df['Client'],
        y=df['Contribution'],
        name='Contribution',
        marker_color='blue'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=df['Client'],
        y=df['Reward'],
        name='Reward (tokens)',
        marker_color='green'
    ), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Trust scores
    st.markdown("### Client Trust Scores")
    
    np.random.seed(42)
    trust_data = pd.DataFrame({
        'Client': list(range(20)),
        'Trust Score': np.random.uniform(0.5, 1.0, 20),
        'Reliability': np.random.uniform(0.6, 1.0, 20),
        'Freshness': np.random.uniform(0.4, 1.0, 20),
        'Consistency': np.random.uniform(0.5, 0.95, 20)
    })
    
    # Radar chart for selected client
    selected_client = st.selectbox("Select Client", range(20))
    
    client_data = trust_data[trust_data['Client'] == selected_client].iloc[0]
    
    fig = go.Figure()
    
    categories = ['Trust Score', 'Reliability', 'Freshness', 'Consistency']
    values = [client_data['Trust Score'], client_data['Reliability'], 
              client_data['Freshness'], client_data['Consistency']]
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'Client {selected_client}'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trust score table
    st.markdown("### All Clients Trust Metrics")
    
    st.dataframe(trust_data.style.format({
        'Trust Score': '{:.3f}',
        'Reliability': '{:.3f}',
        'Freshness': '{:.3f}',
        'Consistency': '{:.3f}'
    }).background_gradient(subset=['Trust Score'], cmap='RdYlGn'),
    use_container_width=True)
    
    # Reward history
    st.markdown("### Cumulative Reward History")
    
    # Simulate reward history
    rounds = list(range(1, 51))
    
    fig = go.Figure()
    
    for client_id in [0, 5, 10, 15]:
        cumulative = np.cumsum([rewards[client_id] / 50 + np.random.uniform(-0.5, 0.5) 
                               for _ in rounds])
        fig.add_trace(go.Scatter(
            x=rounds,
            y=cumulative,
            mode='lines',
            name=f'Client {client_id}'
        ))
    
    fig.update_layout(
        height=400,
        xaxis_title='Round',
        yaxis_title='Cumulative Reward'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Free rider detection
    st.markdown("### Free Rider Detection")
    
    low_contributors = [k for k, v in contributions.items() if v < 0.3]
    
    if low_contributors:
        st.warning(f"Detected {len(low_contributors)} potential free riders: Clients {low_contributors}")
    else:
        st.success("No free riders detected")
    
    # Behavior analysis
    st.markdown("### Behavior Type Distribution")
    
    behavior_counts = pd.DataFrame({
        'Behavior': ['Conservative', 'Aggressive', 'Momentum', 'Long-term', 'Contrarian'],
        'Count': [6, 4, 5, 3, 2],
        'Avg Contribution': [0.55, 0.72, 0.68, 0.48, 0.61]
    })
    
    fig = px.bar(behavior_counts, x='Behavior', y='Count',
                color='Avg Contribution', color_continuous_scale='Viridis')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    render_footer()


if __name__ == "__main__":
    render_incentives_page()
