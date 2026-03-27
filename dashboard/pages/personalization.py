"""
Personalization Analytics Page
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


def render_personalization_page():
    """Render personalization analytics page"""
    st.markdown("# 👤 Personalization Analytics")
    
    data = load_demo_data()
    
    st.markdown("### Global vs Personalized Model Performance")
    
    # Simulate global and personalized accuracies
    np.random.seed(42)
    n_clients = 20
    
    global_acc = 0.78
    personalized_accs = {
        i: global_acc + np.random.uniform(-0.05, 0.15)
        for i in range(n_clients)
    }
    
    # Bar comparison
    df = pd.DataFrame({
        'Client': list(personalized_accs.keys()),
        'Personalized': list(personalized_accs.values()),
        'Global': [global_acc] * n_clients
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Client'], y=df['Global'],
        name='Global Model',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=df['Client'], y=df['Personalized'],
        name='Personalized Model',
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        height=400,
        barmode='group',
        yaxis_title='Accuracy',
        xaxis_title='Client ID'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Personalization gain
    st.markdown("### Personalization Gain")
    
    gains = {k: v - global_acc for k, v in personalized_accs.items()}
    avg_gain = np.mean(list(gains.values()))
    improved = sum(1 for g in gains.values() if g > 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Gain", f"{avg_gain:.2%}", 
                 delta_color="normal" if avg_gain > 0 else "inverse")
    
    with col2:
        st.metric("Improved Clients", f"{improved}/{n_clients}")
    
    with col3:
        st.metric("Max Improvement", f"{max(gains.values()):.2%}")
    
    # Gain distribution
    fig = px.histogram(x=list(gains.values()), nbins=15,
                      color_discrete_sequence=['steelblue'],
                      labels={'x': 'Personalization Gain'})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Per-client metrics
    st.markdown("### Per-Client Performance Metrics")
    
    metrics_data = []
    behavior_types = ['Conservative', 'Aggressive', 'Momentum', 'Long-term', 'Contrarian']
    
    for client_id in range(n_clients):
        metrics_data.append({
            'Client ID': client_id,
            'Global Acc': global_acc,
            'Personalized Acc': personalized_accs[client_id],
            'Gain': gains[client_id],
            'F1 Score': personalized_accs[client_id] - np.random.uniform(0.02, 0.05),
            'Precision': personalized_accs[client_id] + np.random.uniform(-0.03, 0.03),
            'Recall': personalized_accs[client_id] + np.random.uniform(-0.04, 0.02),
            'Data Size': np.random.randint(500, 2000),
            'Behavior Type': behavior_types[client_id % len(behavior_types)]
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    st.dataframe(df_metrics.style.format({
        'Global Acc': '{:.2%}',
        'Personalized Acc': '{:.2%}',
        'Gain': '{:.2%}',
        'F1 Score': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}'
    }).background_gradient(subset=['Gain'], cmap='RdYlGn'),
    use_container_width=True)
    
    # Behavior type analysis
    st.markdown("### Performance by Behavior Type")
    
    behavior_perf = df_metrics.groupby('Behavior Type').agg({
        'Personalized Acc': 'mean',
        'Gain': 'mean',
        'Client ID': 'count'
    }).reset_index()
    behavior_perf.columns = ['Behavior Type', 'Avg Accuracy', 'Avg Gain', 'Count']
    
    fig = px.bar(behavior_perf, x='Behavior Type', y='Avg Accuracy',
                color='Avg Gain', color_continuous_scale='RdYlGn',
                text='Count')
    fig.update_traces(texttemplate='%{text} clients')
    fig.update_layout(height=350, yaxis_title='Average Accuracy')
    st.plotly_chart(fig, use_container_width=True)
    
    # Client detail view
    st.markdown("### Client Detail View")
    
    selected_client = st.selectbox("Select Client for Details", range(n_clients))
    
    client_data = df_metrics[df_metrics['Client ID'] == selected_client].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart
        categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        values = [
            client_data['Personalized Acc'],
            client_data['F1 Score'],
            client_data['Precision'],
            client_data['Recall']
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Performance'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=350,
            title=f"Client {selected_client} Performance Profile"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Comparison bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Global', 'Personalized'],
            y=[client_data['Global Acc'], client_data['Personalized Acc']],
            marker_color=['lightblue', 'darkblue'],
            text=[f"{client_data['Global Acc']:.1%}", f"{client_data['Personalized Acc']:.1%}"],
            textposition='auto'
        ))
        fig.update_layout(height=350, yaxis_title='Accuracy',
                         title=f"Client {selected_client}: Global vs Personalized")
        st.plotly_chart(fig, use_container_width=True)
    
    # Learning curves
    st.markdown("### Personalized Learning Curves")
    
    rounds = list(range(1, 51))
    
    fig = go.Figure()
    
    for client_id in [0, 5, 10, 15]:
        # Simulate learning curve
        curve = 0.5 + 0.3 * (1 - np.exp(-np.array(rounds) / 15)) + np.random.uniform(0, 0.05, len(rounds))
        curve += personalized_accs[client_id] - 0.75  # Adjust to final accuracy
        
        fig.add_trace(go.Scatter(
            x=rounds, y=curve,
            mode='lines', name=f'Client {client_id}'
        ))
    
    fig.update_layout(
        height=400,
        xaxis_title='Round',
        yaxis_title='Accuracy',
        legend=dict(x=0.7, y=0.2)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    render_footer()


if __name__ == "__main__":
    render_personalization_page()
