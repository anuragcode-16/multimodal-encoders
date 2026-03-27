# FedFIM: Drift-Aware, Incentive-Compatible, Multimodal Personalized Federated Learning for Financial Intelligence

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-yellow.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)

FedFIM is a comprehensive federated learning framework designed specifically for robust financial modeling. It facilitates privacy-preserving predictive analytics by fusing multiple financial modalities—including historical market data, social sentiment, and user behavior patterns. To handle the volatile and heterogeneous nature of financial markets, FedFIM intrinsically supports temporal drift adaptation alongside client-specific personalization and fair incentive mechanisms.

## 🎯 Key Features

- **Multimodal Fusion**: Integrates market data (OHLCV + technical indicators), sentiment analysis (news, social media), and user behavior profiles
- **Privacy-Preserving FL**: Federated learning with local data, no raw data sharing
- **Drift-Aware Aggregation**: Custom aggregation that adapts to temporal distribution shifts
- **Incentive Mechanism**: Contribution scoring and fair reward distribution
- **Personalization**: Client-specific model adaptation for heterogeneous users
- **Comprehensive Dashboard**: Full Streamlit UI for visualization and analysis

## 🏗️ Architecture Overview

```
FEDFIM ARCHITECTURE
│
├── Data Encoders (Multimodal)
│   ├── Price Encoder (LSTM)        # OHLCV + technical indicators
│   ├── Sentiment Encoder (MLP)     # News & social media embeddings
│   └── Behavior Encoder (MLP)      # User trading patterns & risk
│
├── Fusion Layer
│   └── Attention | FUSION LAYER    # Multi-head attention fusion
│
├── Prediction Heads
│   ├── Direction Head              # Buy/Sell/Hold predictions
│   ├── Risk Head                   # Risk assessment
│   └── Action Head                 # Recommended actions
│
├── Personalization Adapter (Per-Client)
│   └── Client-specific model adaptation
│
└── Drift-Aware Aggregation (Server)
    └── Handles temporal distribution shifts
```

## 📁 Project Structure

```text
fedfim/
├── .github/
│   └── workflows/                  # CI/CD pipelines
├── data/
│   ├── processed/                  # Processed datasets
│   ├── raw/                        # Raw downloaded data
│   └── synthetic/                  # Generated synthetic data
├── models/
│   ├── checkpoints/                # Training checkpoints
│   ├── client_models/              # Client-specific models
│   └── global_model.pth            # Global FL model
├── notebooks/                      # Jupyter notebooks
├── outputs/                        # Generated outputs and figures
├── scripts/                        # Utility scripts
├── src/
│   ├── data_collection/            # Data pipeline
│   ├── features/                   # Feature engineering
│   ├── federated/                  # FL logic
│   ├── models/                     # Neural network models
│   ├── training/                   # Training scripts
│   ├── utils/                      # Utilities
│   ├── visualization/              # Visualization tools
│   └── config.py                   # Configuration settings
├── dashboard/
│   ├── pages/                      # Individual page components
│   └── ui_components.py            # Shared UI components
├── tests/                          # Unit tests
├── .env.example                    # Example environment variables
├── .gitignore                      # Git ignore file
├── app.py                          # Main Streamlit application
├── docker-compose.yml              # Docker Compose configuration
├── Dockerfile                      # Docker configuration
├── LICENSE                         # License file
├── Makefile                        # Automation commands
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── setup.py                        # Package setup
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/anuragcode-16/multimodal-encoders.git
cd fedfim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run Training

```bash
# Train FedFIM
python -m src.training.train_fedfim

# Or use Makefile
make train
```

### Launch Dashboard

```bash
streamlit run app.py
# Or
make dashboard
```

## 🔬 Research Components

### 1. Multimodal Fusion

Three distinct encoders process different data modalities:

- **Price Encoder**: LSTM-based processing of OHLCV and technical indicators
- **Sentiment Encoder**: Text embeddings from financial news and social media
- **Behavior Encoder**: User trading patterns and risk preferences

### 2. Federated Learning Algorithm

FedFIM improves upon FedAvg with:

- **Drift-weighted aggregation**: Lower weight for high-drift clients
- **Contribution-based rewards**: Incentivize quality updates
- **Local adaptation**: Personalized layers per client

### 3. Drift Detection

Multiple drift detection methods:

- **Parameter drift**: L2 distance between updates
- **Distribution drift**: KL divergence, Wasserstein distance
- **Concept drift**: Performance monitoring

### 4. Incentive Mechanism

Contribution scoring based on:

- Performance improvement
- Update consistency
- Data freshness
- Stability under drift

## 📊 Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score, AUC

### Financial Metrics
- Sharpe Ratio, Cumulative Return, Max Drawdown, Volatility

### Federated Metrics
- Communication Cost, Convergence Rounds, Personalization Gain

## 🧪 Running Experiments

```bash
# Run all experiments (FedFIM + baselines)
bash scripts/run_experiments.sh

# Run with custom parameters
ROUNDS=100 CLIENTS=50 bash scripts/run_experiments.sh

# Generate paper figures
make paper-figures
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fedfim:latest .

# Run dashboard
docker run -p 8501:8501 fedfim:latest

# Or use docker-compose
docker-compose up -d
```

## 📊 Dashboard Features

1. **Overview**: Key metrics, predictions, and system status
2. **Market Analytics**: Price charts, technical indicators
3. **Sentiment Analytics**: Sentiment trends, word clouds
4. **Predictions**: Model predictions, confidence scores
5. **Federated Training**: Training curves, client metrics
6. **Personalization**: Global vs personalized performance
7. **Drift Analysis**: Drift scores, regime detection
8. **Incentives**: Contribution scores, reward distribution

## ⚙️ Configuration

Edit `src/config.py` or use environment variables:

```bash
export NUM_CLIENTS=50
export FEDERATED_ROUNDS=100
export LEARNING_RATE=0.001
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
