"""
Advanced charting utilities for FedFIM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class ChartGenerator:
    """
    Generate publication-quality charts for research paper
    """
    
    def __init__(self, output_dir: str = 'outputs/paper_figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper settings
        self.figsize_single = (8, 5)
        self.figsize_double = (12, 5)
        self.dpi = 300
        self.font_size = 12
    
    def plot_training_comparison(self, histories: Dict[str, Dict], 
                                 metric: str = 'accuracy',
                                 title: str = None,
                                 filename: str = None) -> plt.Figure:
        """
        Compare training curves across methods
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
        
        for (method, history), color in zip(histories.items(), colors):
            rounds = range(1, len(history[metric]) + 1)
            ax.plot(rounds, history[metric], label=method, color=color, linewidth=2)
            
            # Add confidence interval if available
            if f'{metric}_std' in history:
                std = history[f'{metric}_std']
                ax.fill_between(rounds, 
                               np.array(history[metric]) - np.array(std),
                               np.array(history[metric]) + np.array(std),
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Federated Round', fontsize=self.font_size)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=self.font_size)
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(self.output_dir / f'{filename}.pdf', dpi=self.dpi, bbox_inches='tight')
            fig.savefig(self.output_dir / f'{filename}.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_personalization_gain(self, global_acc: float,
                                   personalized_accs: Dict[int, float],
                                   behavior_types: Dict[int, str] = None,
                                   title: str = None,
                                   filename: str = None) -> plt.Figure:
        """
        Plot personalization gains per client
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        clients = list(personalized_accs.keys())
        p_accs = [personalized_accs[c] for c in clients]
        g_accs = [global_acc] * len(clients)
        
        x = np.arange(len(clients))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, g_accs, width, label='Global Model', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, p_accs, width, label='Personalized', color='darkblue')
        
        # Highlight improvements
        for i, (g, p) in enumerate(zip(g_accs, p_accs)):
            if p > g:
                ax.annotate(f'+{(p-g)*100:.1f}%', 
                           xy=(i + width/2, p),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', fontsize=8,
                           color='green')
        
        ax.set_xlabel('Client ID', fontsize=self.font_size)
        ax.set_ylabel('Accuracy', fontsize=self.font_size)
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True, alpha=0.3, axis='y')
        
        if title:
            ax.set_title(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(self.output_dir / f'{filename}.pdf', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_drift_impact(self, drift_history: List[Dict],
                          performance_history: List[Dict],
                          title: str = None,
                          filename: str = None) -> plt.Figure:
        """
        Plot drift scores alongside performance
        """
        fig, ax1 = plt.subplots(figsize=self.figsize_single)
        
        rounds = range(1, len(drift_history) + 1)
        drifts = [d.get('mean_drift', 0) for d in drift_history]
        
        color = 'tab:red'
        ax1.set_xlabel('Federated Round', fontsize=self.font_size)
        ax1.set_ylabel('Drift Score', color=color, fontsize=self.font_size)
        ax1.plot(rounds, drifts, color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Drift Threshold')
        
        ax2 = ax1.twinx()
        
        accuracies = [p.get('accuracy', 0) for p in performance_history]
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color, fontsize=self.font_size)
        ax2.plot(rounds, accuracies, color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        if title:
            ax1.set_title(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(self.output_dir / f'{filename}.pdf', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_contribution_heatmap(self, contribution_history: List[Dict[int, float]],
                                   title: str = None,
                                   filename: str = None) -> plt.Figure:
        """
        Plot heatmap of contributions over rounds
        """
        # Convert to matrix
        if not contribution_history:
            return None
        
        client_ids = sorted(list(contribution_history[0].keys()))
        n_rounds = len(contribution_history)
        n_clients = len(client_ids)
        
        matrix = np.zeros((n_rounds, n_clients))
        for round_idx, round_contrib in enumerate(contribution_history):
            for client_id, value in round_contrib.items():
                col_idx = client_ids.index(client_id)
                matrix[round_idx, col_idx] = value
        
        fig, ax = plt.subplots(figsize=self.figsize_double)
        
        im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd')
        
        ax.set_xlabel('Federated Round', fontsize=self.font_size)
        ax.set_ylabel('Client ID', fontsize=self.font_size)
        
        plt.colorbar(im, ax=ax, label='Contribution Score')
        
        if title:
            ax.set_title(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(self.output_dir / f'{filename}.pdf', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_financial_metrics_comparison(self, metrics: Dict[str, Dict[str, float]],
                                          title: str = None,
                                          filename: str = None) -> plt.Figure:
        """
        Plot comparison of financial metrics across methods
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize_double)
        
        metric_names = ['Sharpe Ratio', 'Cumulative Return', 'Max Drawdown']
        metric_keys = ['sharpe_ratio', 'cumulative_return', 'max_drawdown']
        
        for ax, metric_name, metric_key in zip(axes, metric_names, metric_keys):
            methods = list(metrics.keys())
            values = [metrics[m].get(metric_key, 0) for m in methods]
            
            bars = ax.bar(methods, values, color=plt.cm.tab10(np.linspace(0, 1, len(methods))))
            ax.set_title(metric_name, fontsize=self.font_size)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        if title:
            fig.suptitle(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(self.output_dir / f'{filename}.pdf', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_communication_efficiency(self, methods_data: Dict[str, Dict],
                                      title: str = None,
                                      filename: str = None) -> plt.Figure:
        """
        Plot communication efficiency comparison
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        for method, data in methods_data.items():
            rounds = data['rounds']
            cum_cost = np.cumsum(data['cost_per_round'])
            accuracy = data['accuracy']
            
            ax.plot(cum_cost, accuracy, label=method, linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Cumulative Communication Cost (MB)', fontsize=self.font_size)
        ax.set_ylabel('Accuracy', fontsize=self.font_size)
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=self.font_size + 2)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(self.output_dir / f'{filename}.pdf', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_summary_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Create summary table for paper
        """
        rows = []
        
        for method, metrics in results.items():
            row = {
                'Method': method,
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'F1 Score': f"{metrics.get('f1', 0):.4f}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Comm. Cost (MB)': f"{metrics.get('communication_cost', 0):.1f}",
                'Conv. Rounds': metrics.get('convergence_rounds', '-'),
                'Personalization Gain': f"{metrics.get('personalization_gain', 0):.2%}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to LaTeX
        latex_path = self.output_dir / 'summary_table.tex'
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        return df