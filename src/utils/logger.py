"""
Logging utilities for FedFIM
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class FedFIMLogger:
    """Custom logger for FedFIM experiments"""
    
    def __init__(self, name: str = "FedFIM", log_dir: str = "logs", 
                 level: int = logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            self.log_dir / f"experiment_{timestamp}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.metrics_history = []
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics for tracking"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        self.metrics_history.append(entry)
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    def log_round(self, round_num: int, metrics: dict):
        """Log federated round results"""
        self.logger.info(f"Round {round_num} completed:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def save_metrics(self, filename: str = None):
        """Save metrics history to JSON"""
        filename = filename or f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.log_dir / filename
        
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {path}")
        return str(path)


# Global logger instance
logger = FedFIMLogger()


def get_logger(name: str = "FedFIM") -> FedFIMLogger:
    """Get logger instance"""
    return FedFIMLogger(name)