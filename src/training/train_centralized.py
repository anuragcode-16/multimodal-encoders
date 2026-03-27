"""
Centralized training baseline
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import TRAINING_CONFIG, set_random_seed
from src.data_collection.preprocess import DataPreprocessor, create_data_loaders
from src.models.fedfim import create_fedfim_model
from src.utils.metrics import ClassificationMetrics
from src.utils.helpers import get_device, save_json


def train_centralized(epochs: int = 50, batch_size: int = 64):
    """
    Train model in centralized manner (baseline)
    """
    set_random_seed(TRAINING_CONFIG.random_seed)
    device = get_device()
    
    # Prepare data (combine all client data)
    preprocessor = DataPreprocessor()
    full_data, _ = preprocessor.prepare_full_dataset()
    
    # Create single dataset from all data
    from src.data_collection.preprocess import FederatedDataset
    from torch.utils.data import DataLoader, random_split
    
    # Process all data together
    features = preprocessor._extract_features(full_data, {'risk_tolerance': 0.5})
    labels = preprocessor._create_labels(full_data, {'risk_tolerance': 0.5})
    
    full_dataset = FederatedDataset(features, labels, client_id=0, sequence_length=30)
    
    # Split into train/val/test
    n = len(full_dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model = create_fedfim_model(num_clients=1, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG.learning_rate)
    
    # Training loop
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0.0
    
    for epoch in tqdm(range(epochs), desc="Centralized Training"):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            price = batch['price'].to(device)
            sentiment = batch['sentiment'].to(device)
            behavior = batch['behavior'].to(device)
            labels = batch['label'].to(device)
            client_ids = torch.zeros(len(labels), dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            outputs = model(price, sentiment, behavior, client_ids, use_personalization=False)
            loss = criterion(outputs['direction'], labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs['direction'].argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                price = batch['price'].to(device)
                sentiment = batch['sentiment'].to(device)
                behavior = batch['behavior'].to(device)
                labels = batch['label'].to(device)
                client_ids = torch.zeros(len(labels), dtype=torch.long, device=device)
                
                outputs = model(price, sentiment, behavior, client_ids, use_personalization=False)
                loss = criterion(outputs['direction'], labels)
                
                val_loss += loss.item()
                preds = outputs['direction'].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        # Record history
        history['loss'].append(train_loss / len(train_loader))
        history['accuracy'].append(train_correct / train_total)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_accuracy'].append(val_correct / val_total)
        
        # Save best model
        if val_correct / val_total > best_val_acc:
            best_val_acc = val_correct / val_total
            torch.save(model.state_dict(), 'models/checkpoints/centralized_best.pth')
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            price = batch['price'].to(device)
            sentiment = batch['sentiment'].to(device)
            behavior = batch['behavior'].to(device)
            labels = batch['label'].to(device)
            client_ids = torch.zeros(len(labels), dtype=torch.long, device=device)
            
            outputs = model(price, sentiment, behavior, client_ids, use_personalization=False)
            preds = outputs['direction'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = ClassificationMetrics.compute(
        np.array(all_labels), np.array(all_preds)
    )
    
    results = {
        'method': 'centralized',
        'epochs': epochs,
        'history': history,
        'test_metrics': metrics,
        'best_val_accuracy': best_val_acc
    }
    
    save_json(results, 'models/checkpoints/results/centralized_results.json')
    
    return model, results


if __name__ == "__main__":
    model, results = train_centralized()
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")