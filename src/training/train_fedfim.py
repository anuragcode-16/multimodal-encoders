"""
Main training script for FedFIM
"""
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import set_random_seed, FEDERATED_CONFIG, TRAINING_CONFIG, CHECKPOINTS_DIR
from src.data_collection.preprocess import DataPreprocessor, create_data_loaders
from src.models.fedfim import create_fedfim_model
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.incentive import IncentiveMechanism
from src.utils.metrics import ClassificationMetrics, FinancialMetrics, FederatedMetrics
from src.utils.logger import get_logger
from src.utils.helpers import save_json, count_parameters, get_device, ensure_dir
# Import PlotGenerator for automatic plotting
from src.utils.plotting import PlotGenerator


def train_fedfim(config_override: dict = None):
    """
    Main training function for FedFIM
    """
    # Setup
    logger = get_logger("FedFIM_Training")
    logger.info("Starting FedFIM training")
    
    # Configuration
    config = FEDERATED_CONFIG
    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)
    
    set_random_seed(TRAINING_CONFIG.random_seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Prepare data
    logger.info("Preparing data...")
    preprocessor = DataPreprocessor()
    full_data, client_profiles = preprocessor.prepare_full_dataset()
    client_datasets = preprocessor.create_client_splits(full_data, client_profiles, non_iid=True)
    train_datasets, val_datasets, test_datasets = preprocessor.split_train_val_test(client_datasets)
    
    # Create data loaders
    train_loaders = create_data_loaders(train_datasets, batch_size=config.local_batch_size)
    val_loaders = create_data_loaders(val_datasets, batch_size=config.local_batch_size)
    test_loaders = create_data_loaders(test_datasets, batch_size=config.local_batch_size)
    
    logger.info(f"Created {len(train_loaders)} client datasets")
    
    # Create model
    model = create_fedfim_model(num_clients=config.num_clients, device=device)
    logger.info(f"Model created with {count_parameters(model):,} parameters")
    
    # Create clients
    clients = []
    for client_id, profile in enumerate(client_profiles[:config.num_clients]):
        if client_id in train_loaders and client_id in val_loaders:
            # Create a copy of model for each client
            client_model = create_fedfim_model(num_clients=config.num_clients, device=device)
            client = FederatedClient(
                client_id=client_id,
                model=client_model,
                train_loader=train_loaders[client_id],
                val_loader=val_loaders[client_id],
                config=config,
                device=device
            )
            client.behavior_type = profile.get('behavior_type', 'unknown')
            clients.append(client)
    
    logger.info(f"Created {len(clients)} federated clients")
    
    # Create server
    server = FederatedServer(
        model=model,
        config=config,
        aggregation_type='fedfim',
        device=device
    )
    
    # Incentive mechanism
    incentive = IncentiveMechanism(config)
    
    # Training loop
    logger.info(f"Starting federated training for {config.num_rounds} rounds")
    
    all_results = {
        'rounds': [],
        'global_metrics': [],
        'client_metrics': [],
        'incentive_rewards': [],
        'drift_scores': []
    }
    
    best_accuracy = 0.0
    
    for round_num in tqdm(range(1, config.num_rounds + 1), desc="Federated Rounds"):
        # Train round
        round_metrics = server.train_round(clients, epochs=config.local_epochs)
        
        # Compute incentives
        client_metrics_for_incentive = {
            c['client_id']: c for c in round_metrics['client_metrics']
        }
        rewards = incentive.compute_all_rewards(client_metrics_for_incentive)
        
        # Evaluate global model
        if test_loaders:
            # Combine all test data for global evaluation
            global_metrics = server.evaluate_global_model(
                list(test_loaders.values())[0]  # Use first test loader
            )
        else:
            global_metrics = {'accuracy': 0, 'loss': 0}
        
        # Track drift
        drift_scores = {
            client.client_id: client.drift_score 
            for client in clients if hasattr(client, 'drift_score')
        }
        
        # Store results
        all_results['rounds'].append(round_num)
        all_results['global_metrics'].append(global_metrics)
        all_results['client_metrics'].append(round_metrics['client_metrics'])
        all_results['incentive_rewards'].append(rewards)
        all_results['drift_scores'].append(drift_scores)
        
        # Log progress
        if round_num % 5 == 0:
            logger.info(f"Round {round_num}: Accuracy={global_metrics.get('accuracy', 0):.4f}, "
                       f"Loss={global_metrics.get('loss', 0):.4f}")
        
        # Save best model
        if global_metrics.get('accuracy', 0) > best_accuracy:
            best_accuracy = global_metrics.get('accuracy', 0)
            server.save_checkpoint(str(CHECKPOINTS_DIR / 'best_model.pth'))
    
    # Final evaluation
    logger.info("Training completed. Running final evaluation...")
    
    # Compute final metrics
    final_results = {
        'best_accuracy': best_accuracy,
        'final_accuracy': all_results['global_metrics'][-1].get('accuracy', 0),
        'total_rounds': config.num_rounds,
        'num_clients': len(clients),
        'incentive_summary': incentive.get_summary()
    }
    
    # Save results
    results_dir = CHECKPOINTS_DIR / 'results'
    ensure_dir(str(results_dir))
    
    save_json(all_results, str(results_dir / 'training_history.json'))
    save_json(final_results, str(results_dir / 'final_results.json'))
    
    # ---------------------------------------------------------
    # NEW: Automatically generate plots after training
    # ---------------------------------------------------------
    print("\nGenerating training visualizations...")
    
    plotter = PlotGenerator(output_dir=str(CHECKPOINTS_DIR.parent / 'outputs' / 'plots'))
    
    # 1. Training Curves
    history = {
        'loss': [m.get('loss', 0) for m in all_results['global_metrics']],
        'accuracy': [m.get('accuracy', 0) for m in all_results['global_metrics']]
    }
    fig = plotter.plot_training_curves(history, title="FedFIM Training Progress")
    plotter.save_plot(fig, 'training_progress')
    
    # 2. Client Contributions
    if all_results.get('incentive_rewards'):
        last_rewards = all_results['incentive_rewards'][-1]
        fig2 = plotter.plot_client_contributions(last_rewards)
        plotter.save_plot(fig2, 'client_contributions')
    
    print(f"Plots saved to: {CHECKPOINTS_DIR.parent / 'outputs' / 'plots'}")
    # ---------------------------------------------------------

    logger.info(f"Final accuracy: {final_results['final_accuracy']:.4f}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Results saved to {results_dir}")
    
    return server, clients, all_results


if __name__ == "__main__":
    server, clients, results = train_fedfim()