"""
Evaluation script for all methods
"""
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CHECKPOINTS_DIR, set_random_seed
from src.utils.metrics import ClassificationMetrics, FinancialMetrics, FederatedMetrics
from src.utils.helpers import load_json, save_json
from src.utils.plotting import PlotGenerator


def evaluate_all_methods(results_dir: str = None):
    """
    Evaluate and compare all trained methods
    """
    results_dir = Path(results_dir or CHECKPOINTS_DIR / 'results')
    plotter = PlotGenerator()
    
    # Load results
    results = {}
    
    result_files = {
        'fedfim': 'training_history.json',
        'centralized': 'centralized_results.json',
        'fedavg': 'fedavg_results.json'
    }
    
    for method, filename in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            results[method] = load_json(str(filepath))
    
    if not results:
        print("No results found. Run training first.")
        return
    
    # Comparison metrics
    comparison = {}
    
    for method, data in results.items():
        if 'global_metrics' in data:
            # FedFIM format
            accuracies = [m.get('accuracy', 0) for m in data['global_metrics']]
            comparison[method] = {
                'final_accuracy': accuracies[-1] if accuracies else 0,
                'best_accuracy': max(accuracies) if accuracies else 0,
                'convergence_speed': len([a for a in accuracies if a < 0.5]),  # Rounds to 50% acc
                'accuracy_std': np.std(accuracies) if accuracies else 0
            }
        elif 'test_metrics' in data:
            # Centralized format
            comparison[method] = {
                'final_accuracy': data['test_metrics'].get('accuracy', 0),
                'best_accuracy': data.get('best_val_accuracy', 0),
                'f1_score': data['test_metrics'].get('f1_macro', 0)
            }
        elif 'accuracy' in data:
            # FedAvg format
            comparison[method] = {
                'final_accuracy': data['accuracy'][-1] if data['accuracy'] else 0,
                'best_accuracy': max(data['accuracy']) if data['accuracy'] else 0
            }
    
    # Generate comparison plots
    fig = plotter.plot_federated_comparison(comparison)
    plotter.save_plot(fig, 'methods_comparison')
    
    # Save comparison
    save_json(comparison, str(results_dir / 'methods_comparison.json'))
    
    print("\n=== Methods Comparison ===")
    print(f"{'Method':<15} {'Final Acc':<12} {'Best Acc':<12}")
    print("-" * 40)
    for method, metrics in comparison.items():
        print(f"{method:<15} {metrics['final_accuracy']:.4f}      {metrics['best_accuracy']:.4f}")
    
    return comparison


def evaluate_personalization(server, clients, test_loaders):
    """
    Evaluate personalization gain
    """
    results = {}
    
    # Global model accuracy
    global_acc = server.global_metrics_history[-1]['accuracy'] if server.global_metrics_history else 0
    
    # Personalized accuracy per client
    personalized_accs = {}
    
    for client in clients:
        client_id = client.client_id
        if client_id in test_loaders:
            metrics = client.evaluate()
            personalized_accs[client_id] = metrics['accuracy']
    
    # Compute gain
    gain_metrics = FederatedMetrics.personalization_gain(global_acc, personalized_accs)
    
    results = {
        'global_accuracy': global_acc,
        'personalized_accuracies': personalized_accs,
        'mean_gain': gain_metrics['mean_gain'],
        'improved_clients': gain_metrics['improved_clients']
    }
    
    return results


def evaluate_drift_resilience(server):
    """
    Evaluate drift resilience
    """
    from src.utils.metrics import DriftMetrics
    
    drift_history = server.drift_history
    metrics_history = server.global_metrics_history
    
    if len(metrics_history) < 10:
        return {'drift_resilience': 0}
    
    # Find drift points
    drift_points = []
    for i, drift in enumerate(drift_history):
        if drift['mean_drift'] > 0.3:  # Threshold
            drift_points.append(i)
    
    # Compute resilience metrics
    if drift_points:
        # Compare accuracy before and after drift
        before_acc = np.mean([m['accuracy'] for m in metrics_history[:drift_points[0]]])
        after_acc = np.mean([m['accuracy'] for m in metrics_history[drift_points[-1]:]])
        
        resilience = {
            'num_drift_events': len(drift_points),
            'accuracy_before_drift': before_acc,
            'accuracy_after_drift': after_acc,
            'recovery_rate': after_acc / before_acc if before_acc > 0 else 0
        }
    else:
        resilience = {
            'num_drift_events': 0,
            'recovery_rate': 1.0
        }
    
    return resilience


def generate_paper_plots(results_dir: str = None):
    """
    Generate publication-ready plots
    """
    results_dir = Path(results_dir or CHECKPOINTS_DIR / 'results')
    plotter = PlotGenerator()
    
    # Load FedFIM results
    fedfim_path = results_dir / 'training_history.json'
    if not fedfim_path.exists():
        print("FedFIM results not found")
        return
    
    fedfim_results = load_json(str(fedfim_path))
    
    # Plot 1: Training curves
    history = {
        'loss': [m.get('loss', 0) for m in fedfim_results.get('global_metrics', [])],
        'accuracy': [m.get('accuracy', 0) for m in fedfim_results.get('global_metrics', [])]
    }
    fig1 = plotter.plot_training_curves(history, "FedFIM Training Progress")
    plotter.save_plot(fig1, 'paper_training_curves')
    
    # Plot 2: Client contributions
    if fedfim_results.get('incentive_rewards'):
        last_rewards = fedfim_results['incentive_rewards'][-1]
        fig2 = plotter.plot_client_contributions(last_rewards, "Client Contribution Scores")
        plotter.save_plot(fig2, 'paper_contributions')
    
    # Plot 3: Drift scores
    if fedfim_results.get('drift_scores'):
        drift_data = [
            {'round': i+1, 'mean_drift': np.mean(list(d.values())) if d else 0,
             'max_drift': max(d.values()) if d else 0, 'min_drift': min(d.values()) if d else 0}
            for i, d in enumerate(fedfim_results['drift_scores'])
        ]
        fig3 = plotter.plot_drift_scores(drift_data)
        plotter.save_plot(fig3, 'paper_drift_scores')
    
    print("Paper plots saved to outputs/plots/")


if __name__ == "__main__":
    evaluate_all_methods()
    generate_paper_plots()