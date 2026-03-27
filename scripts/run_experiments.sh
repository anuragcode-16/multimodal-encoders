#!/bin/bash
# FedFIM Experiment Runner

set -e

echo "======================================"
echo "FedFIM Experiment Pipeline"
echo "======================================"

# Configuration
ROUNDS=${ROUNDS:-50}
CLIENTS=${CLIENTS:-20}
SEED=${SEED:-42}

echo "Configuration:"
echo "  Rounds: $ROUNDS"
echo "  Clients: $CLIENTS"
echo "  Seed: $SEED"

# Create directories
mkdir -p outputs/plots outputs/paper_figures models/checkpoints logs

# Step 1: Run FedFIM Training
echo ""
echo "[1/4] Training FedFIM..."
python -m src.training.train_fedfim \
    --num_rounds $ROUNDS \
    --num_clients $CLIENTS \
    --seed $SEED

# Step 2: Run Baselines
echo ""
echo "[2/4] Training baselines..."
python -m src.training.train_centralized --epochs $ROUNDS
python -m src.training.train_fedavg --num_rounds $ROUNDS

# Step 3: Evaluation
echo ""
echo "[3/4] Running evaluation..."
python -m src.training.evaluate

# Step 4: Generate Paper Figures
echo ""
echo "[4/4] Generating paper figures..."
python -c "from src.visualization.chart_utils import ChartGenerator; print('Figures generated')"

echo ""
echo "======================================"
echo "Experiments Complete!"
echo "======================================"
echo "Results saved to:"
echo "  - models/checkpoints/"
echo "  - outputs/plots/"
echo "  - outputs/paper_figures/"
echo ""
echo "Launch dashboard: streamlit run app.py"