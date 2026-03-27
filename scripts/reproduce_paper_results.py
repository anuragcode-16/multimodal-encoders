#!/usr/bin/env python
"""
Script to reproduce all paper results and generate figures.
This runs the full experiment suite for the IEEE/Springer paper.
"""
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import set_random_seed, FEDERATED_CONFIG
from src.utils.logger import get_logger
from src.visualization.chart_utils import ChartGenerator
from src.utils.helpers import save_json


def run_paper_experiments():
    """
    Run the complete experiment suite for the paper.
    """
    logger = get_logger("PaperReproduction")
    logger.info("Starting Paper Experiment Reproduction")
    
    start_time = time.time()
    results = {}
    
    # Configuration for paper experiments
    config = {
        'num_rounds': 50,
        'num_clients': 20,
        'local_epochs': 5,
        'seed': 42
    }
    
    set_random_seed(config['seed'])
    
    # Import training modules
    from src.training.train_fedfim import train_fedfim
    from src.training.train_fedavg import train_fedavg
    from src.training.train_centralized import train_centralized
    from src.training.evaluate import evaluate_all_methods
    
    # ==========================================================
    # Experiment 1: FedFIM (Proposed Method)
    # ==========================================================
    logger.info("=" * 50)
    logger.info("Experiment 1: Training FedFIM")
    logger.info("=" * 50)
    
    try:
        server_fedfim, clients_fedfim, results_fedfim = train_fedfim(
            config_override=config
        )
        results['FedFIM'] = {
            'final_accuracy': results_fedfim['global_metrics'][-1]['accuracy'],
            'best_accuracy': max(m['accuracy'] for m in results_fedfim['global_metrics']),
            'convergence_round': next((i for i, m in enumerate(results_fedfim['global_metrics']) if m['accuracy'] > 0.8), 50)
        }
        logger.info(f"FedFIM Final Accuracy: {results['FedFIM']['final_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"FedFIM training failed: {e}")
        results['FedFIM'] = {'error': str(e)}
    
    # ==========================================================
    # Experiment 2: FedAvg (Baseline)
    # ==========================================================
    logger.info("=" * 50)
    logger.info("Experiment 2: Training FedAvg Baseline")
    logger.info("=" * 50)
    
    try:
        server_fedavg, results_fedavg = train_fedavg()
        results['FedAvg'] = {
            'final_accuracy': results_fedavg['accuracy'][-1] if results_fedavg['accuracy'] else 0,
            'best_accuracy': max(results_fedavg['accuracy']) if results_fedavg['accuracy'] else 0
        }
        logger.info(f"FedAvg Final Accuracy: {results['FedAvg']['final_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"FedAvg training failed: {e}")
        results['FedAvg'] = {'error': str(e)}
    
    # ==========================================================
    # Experiment 3: Centralized (Baseline)
    # ==========================================================
    logger.info("=" * 50)
    logger.info("Experiment 3: Training Centralized Baseline")
    logger.info("=" * 50)
    
    try:
        model_centralized, results_centralized = train_centralized(
            epochs=config['num_rounds']
        )
        results['Centralized'] = {
            'final_accuracy': results_centralized['test_metrics']['accuracy'],
            'best_accuracy': results_centralized.get('best_val_accuracy', 0)
        }
        logger.info(f"Centralized Final Accuracy: {results['Centralized']['final_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Centralized training failed: {e}")
        results['Centralized'] = {'error': str(e)}
    
    # ==========================================================
    # Generate Figures
    # ==========================================================
    logger.info("Generating paper figures...")
    
    chart_gen = ChartGenerator()
    
    # Generate comparison plots
    try:
        # Prepare histories for plotting
        histories = {
            'FedFIM': {
                'accuracy': [m['accuracy'] for m in results_fedfim['global_metrics']],
                'loss': [m['loss'] for m in results_fedfim['global_metrics']]
            },
            'FedAvg': {
                'accuracy': results_fedavg['accuracy'],
                'loss': [1-a for a in results_fedavg['accuracy']]  # Approximate
            },
            'Centralized': {
                'accuracy': results_centralized['history']['val_accuracy'],
                'loss': results_centralized['history']['val_loss']
            }
        }
        
        chart_gen.plot_training_comparison(
            histories, 
            metric='accuracy',
            title='Model Comparison: Validation Accuracy',
            filename='paper_fig_1_accuracy_comparison'
        )
        
    except Exception as e:
        logger.warning(f"Could not generate comparison plot: {e}")
    
    # ==========================================================
    # Create Summary Table
    # ==========================================================
    logger.info("Creating summary table...")
    
    summary_table = chart_gen.create_summary_table(results)
    logger.info("\n" + str(summary_table))
    
    # ==========================================================
    # Save All Results
    # ==========================================================
    output_dir = Path('outputs/paper_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(results, str(output_dir / 'paper_results.json'))
    
    elapsed = time.time() - start_time
    logger.info(f"\nExperiment reproduction completed in {elapsed/60:.2f} minutes")
    logger.info(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    results = run_paper_experiments()
    print("\n" + "="*50)
    print("PAPER RESULTS SUMMARY")
    print("="*50)
    for method, metrics in results.items():
        print(f"\n{method}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")