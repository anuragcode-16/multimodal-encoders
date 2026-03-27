"""
FedAvg baseline training
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FEDERATED_CONFIG, set_random_seed
from src.federated.server import FederatedServer
from src.training.train_fedfim import train_fedfim


def train_fedavg():
    """
    Train using FedAvg (baseline comparison)
    """
    print("Training FedAvg baseline...")
    
    # Use same setup but with FedAvg aggregation
    config_override = {
        'aggregation_type': 'fedavg'
    }
    
    # Import and run modified training
    set_random_seed(42)
    
    # The server will use FedAvg aggregation
    from src.data_collection.preprocess import DataPreprocessor, create_data_loaders
    from src.models.fedfim import create_fedfim_model
    from src.federated.client import FederatedClient
    from src.utils.helpers import get_device
    
    device = get_device()
    config = FEDERATED_CONFIG
    
    # Prepare data
    preprocessor = DataPreprocessor()
    full_data, client_profiles = preprocessor.prepare_full_dataset()
    client_datasets = preprocessor.create_client_splits(full_data, client_profiles)
    train_datasets, val_datasets, test_datasets = preprocessor.split_train_val_test(client_datasets)
    train_loaders = create_data_loaders(train_datasets, batch_size=config.local_batch_size)
    val_loaders = create_data_loaders(val_datasets, batch_size=config.local_batch_size)
    
    # Create model and server with FedAvg
    model = create_fedfim_model(num_clients=config.num_clients, device=device)
    
    # Use FedAvg aggregation
    server = FederatedServer(
        model=model,
        config=config,
        aggregation_type='fedavg',
        device=device
    )
    
    # Create clients
    clients = []
    for client_id in train_loaders:
        client_model = create_fedfim_model(num_clients=config.num_clients, device=device)
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            train_loader=train_loaders[client_id],
            val_loader=val_loaders.get(client_id, train_loaders[client_id]),
            config=config,
            device=device
        )
        clients.append(client)
    
    # Train
    results = {'rounds': [], 'accuracy': []}
    
    from tqdm import tqdm
    for round_num in tqdm(range(1, config.num_rounds + 1), desc="FedAvg Training"):
        round_metrics = server.train_round(clients, epochs=config.local_epochs)
        
        if test_datasets:
            test_loader = list(create_data_loaders(test_datasets).values())[0]
            global_metrics = server.evaluate_global_model(test_loader)
            results['accuracy'].append(global_metrics['accuracy'])
        
        results['rounds'].append(round_num)
    
    # Save results
    from src.utils.helpers import save_json
    save_json(results, 'models/checkpoints/results/fedavg_results.json')
    
    print(f"FedAvg Final Accuracy: {results['accuracy'][-1]:.4f}")
    
    return server, results


if __name__ == "__main__":
    server, results = train_fedavg()