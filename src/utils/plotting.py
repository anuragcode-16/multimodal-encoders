"""
Plotting utilities for FedFIM visualization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PlotGenerator:
    """Generate plots for FedFIM analysis"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path('outputs/plots')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, history: Dict[str, List], 
                             title: str = "Training Progress") -> go.Figure:
        """Plot training loss and accuracy curves"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
        
        rounds = list(range(1, len(history.get('loss', [])) + 1))
        
        # Loss curve
        fig.add_trace(
            go.Scatter(x=rounds, y=history.get('loss', []),
                      mode='lines+markers', name='Loss',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Accuracy curve
        fig.add_trace(
            go.Scatter(x=rounds, y=history.get('accuracy', []),
                      mode='lines+markers', name='Accuracy',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=400,
            xaxis_title='Round',
            xaxis2_title='Round',
            yaxis_title='Loss',
            yaxis2_title='Accuracy'
        )
        
        return fig
    
    def plot_candlestick(self, data: pd.DataFrame, 
                         title: str = "Price Chart") -> go.Figure:
        """Plot candlestick chart"""
        fig = go.Figure(data=[
            go.Candlestick(
                x=data['date'] if 'date' in data.columns else data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC'
            )
        ])
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig
    
    def plot_technical_indicators(self, data: pd.DataFrame) -> go.Figure:
        """Plot technical indicators"""
        fig = make_subplots(rows=4, cols=1, 
                           subplot_titles=('Price & MAs', 'RSI', 'MACD', 'Volume'),
                           vertical_spacing=0.05)
        
        dates = data['date'] if 'date' in data.columns else data.index
        
        # Price with moving averages
        fig.add_trace(
            go.Scatter(x=dates, y=data['close'], mode='lines', name='Close', line=dict(color='blue')),
            row=1, col=1
        )
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=data['sma_20'], mode='lines', name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        if 'sma_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=data['sma_50'], mode='lines', name='SMA 50', line=dict(color='red')),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in data.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=data['rsi'], mode='lines', name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in data.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=data['macd'], mode='lines', name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            if 'macd_signal' in data.columns:
                fig.add_trace(
                    go.Scatter(x=dates, y=data['macd_signal'], mode='lines', name='Signal', line=dict(color='orange')),
                    row=3, col=1
                )
        
        # Volume
        colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] else 'red' 
                  for i in range(len(data))]
        fig.add_trace(
            go.Bar(x=dates, y=data['volume'], name='Volume', marker_color=colors),
            row=4, col=1
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Technical Analysis")
        
        return fig
    
    def plot_sentiment_analysis(self, sentiment_data: pd.DataFrame) -> go.Figure:
        """Plot sentiment analysis"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Sentiment Trend', 'Sentiment Distribution',
                                          'Sentiment vs Returns', 'Sentiment Heatmap'))
        
        dates = sentiment_data['date'] if 'date' in sentiment_data.columns else sentiment_data.index
        
        # Sentiment trend
        fig.add_trace(
            go.Scatter(x=dates, y=sentiment_data.get('sentiment_score', [0]*len(dates)),
                      mode='lines', name='Sentiment', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Sentiment distribution
        if 'sentiment' in sentiment_data.columns:
            sentiment_counts = sentiment_data['sentiment'].value_counts()
            fig.add_trace(
                go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                      name='Count', marker_color=['green', 'red', 'gray']),
                row=1, col=2
            )
        
        # Scatter plot
        if 'returns' in sentiment_data.columns and 'sentiment_score' in sentiment_data.columns:
            fig.add_trace(
                go.Scatter(x=sentiment_data['sentiment_score'], 
                          y=sentiment_data['returns'],
                          mode='markers', name='Sentiment vs Returns',
                          marker=dict(color='blue', opacity=0.5)),
                row=2, col=1
            )
        
        # Heatmap placeholder
        fig.add_trace(
            go.Heatmap(z=[[0, 0.5], [0.5, 1]], showscale=False),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Sentiment Analysis")
        
        return fig
    
    def plot_client_contributions(self, contributions: Dict[int, float],
                                  title: str = "Client Contributions") -> go.Figure:
        """Plot client contribution scores"""
        clients = list(contributions.keys())
        scores = list(contributions.values())
        
        # Color by contribution level
        colors = ['green' if s > 0.5 else 'orange' if s > 0.3 else 'red' for s in scores]
        
        fig = go.Figure(data=[
            go.Bar(x=clients, y=scores, marker_color=colors, name='Contribution')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Client ID',
            yaxis_title='Contribution Score',
            height=400
        )
        
        return fig
    
    def plot_drift_scores(self, drift_history: List[Dict]) -> go.Figure:
        """Plot drift scores over time"""
        if not drift_history:
            return go.Figure()
        
        rounds = [d['round'] for d in drift_history]
        mean_drifts = [d['mean_drift'] for d in drift_history]
        max_drifts = [d['max_drift'] for d in drift_history]
        min_drifts = [d['min_drift'] for d in drift_history]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(x=rounds, y=mean_drifts, mode='lines+markers',
                      name='Mean Drift', line=dict(color='blue', width=2))
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=max_drifts, mode='lines',
                      name='Max Drift', line=dict(color='red', dash='dash'))
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=min_drifts, mode='lines',
                      name='Min Drift', line=dict(color='green', dash='dash'))
        )
        
        fig.update_layout(
            title='Drift Scores Over Training',
            xaxis_title='Round',
            yaxis_title='Drift Score',
            height=400
        )
        
        return fig
    
    def plot_personalization_comparison(self, global_acc: float,
                                         personalized_accs: Dict[int, float]) -> go.Figure:
        """Compare global vs personalized performance"""
        clients = list(personalized_accs.keys())
        p_accs = list(personalized_accs.values())
        g_accs = [global_acc] * len(clients)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(x=clients, y=g_accs, name='Global Model',
                  marker_color='lightblue', opacity=0.7)
        )
        fig.add_trace(
            go.Bar(x=clients, y=p_accs, name='Personalized Model',
                  marker_color='darkblue')
        )
        
        fig.update_layout(
            title='Global vs Personalized Model Performance',
            xaxis_title='Client ID',
            yaxis_title='Accuracy',
            barmode='group',
            height=400
        )
        
        return fig
    
    def plot_aggregation_weights(self, weights_history: List[Dict[int, float]]) -> go.Figure:
        """Plot aggregation weights evolution"""
        if not weights_history:
            return go.Figure()
        
        # Stack plot
        rounds = list(range(1, len(weights_history) + 1))
        client_ids = list(weights_history[0].keys())
        
        fig = go.Figure()
        
        for cid in client_ids:
            weights = [w.get(cid, 0) for w in weights_history]
            fig.add_trace(
                go.Scatter(x=rounds, y=weights, mode='lines',
                          name=f'Client {cid}', stackgroup='one')
            )
        
        fig.update_layout(
            title='Aggregation Weight Distribution Over Rounds',
            xaxis_title='Round',
            yaxis_title='Weight',
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              labels: List[str] = None) -> go.Figure:
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['Up', 'Down', 'Neutral']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig
    
    def plot_financial_performance(self, returns: np.ndarray,
                                   title: str = "Strategy Performance") -> go.Figure:
        """Plot cumulative returns and drawdown"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Cumulative Returns', 'Drawdown'),
                           vertical_spacing=0.1)
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(y=cumulative, mode='lines', name='Cumulative Return',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Drawdown
        fig.add_trace(
            go.Scatter(y=drawdown, mode='lines', name='Drawdown',
                      fill='tozeroy', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=True, title_text=title)
        
        return fig
    
    def plot_federated_comparison(self, methods_results: Dict[str, Dict]) -> go.Figure:
        """Compare different federated methods"""
        fig = go.Figure()
        
        for method, results in methods_results.items():
            rounds = list(range(1, len(results.get('accuracy', [])) + 1))
            fig.add_trace(
                go.Scatter(x=rounds, y=results.get('accuracy', []),
                          mode='lines+markers', name=method,
                          line=dict(width=2))
            )
        
        fig.update_layout(
            title='Federated Methods Comparison',
            xaxis_title='Round',
            yaxis_title='Accuracy',
            height=400,
            legend=dict(x=0.7, y=0.9)
        )
        
        return fig
    
    def plot_wordcloud(self, text_data: List[str], title: str = "Word Cloud") -> go.Figure:
        """Generate word frequency visualization (simplified)"""
        from collections import Counter
        import re
        
        # Tokenize and count
        words = []
        for text in text_data:
            words.extend(re.findall(r'\w+', text.lower()))
        
        word_counts = Counter(words)
        top_words = word_counts.most_common(30)
        
        if not top_words:
            return go.Figure()
        
        words_list = [w[0] for w in top_words]
        counts = [w[1] for w in top_words]
        
        fig = go.Figure(data=[
            go.Bar(x=words_list, y=counts, marker_color='steelblue')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Word',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str):
        """Save plot to file"""
        path = self.output_dir / f"{filename}.html"
        fig.write_html(str(path))
        
        # Also save as image if kaleido available
        try:
            fig.write_image(str(self.output_dir / f"{filename}.png"))
        except:
            pass
        
        return str(path)


# Convenience functions
def plot_training_history(history: Dict, output_path: str = None) -> go.Figure:
    """Quick plotting function for training history"""
    generator = PlotGenerator()
    fig = generator.plot_training_curves(history)
    if output_path:
        generator.save_plot(fig, output_path)
    return fig


def plot_metrics_comparison(metrics: Dict[str, float], title: str = "Metrics") -> go.Figure:
    """Plot metrics as bar chart"""
    fig = go.Figure(data=[
        go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
    ])
    fig.update_layout(title=title, height=400)
    return fig