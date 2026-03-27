.PHONY: help install test train dashboard clean docker-build docker-run

help:
    @echo "FedFIM - Available Commands:"
    @echo "  make install        - Install dependencies"
    @echo "  make test           - Run unit tests"
    @echo "  make train          - Train FedFIM model"
    @echo "  make train-baseline - Train baseline models"
    @echo "  make dashboard      - Launch Streamlit dashboard"
    @echo "  make clean          - Clean generated files"
    @echo "  make docker-build   - Build Docker image"
    @echo "  make docker-run     - Run Docker container"
    @echo "  make paper-figures  - Generate paper figures"

install:
    pip install -r requirements.txt
    pip install -e .

test:
    python -m pytest tests/ -v --cov=src --cov-report=html

train:
    python -m src.training.train_fedfim

train-baseline:
    python -m src.training.train_centralized
    python -m src.training.train_fedavg

dashboard:
    streamlit run app.py

clean:
    rm -rf __pycache__ .pytest_cache .coverage htmlcov
    rm -rf data/cache/*.parquet
    rm -rf outputs/plots/*.html outputs/plots/*.png
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -delete

docker-build:
    docker build -t fedfim:latest .

docker-run:
    docker-compose up -d fedfim-dashboard

docker-stop:
    docker-compose down

paper-figures:
    python -m src.training.evaluate
    jupyter nbconvert --execute notebooks/experiments.ipynb --output-dir outputs/paper_figures

lint:
    black src/ tests/
    flake8 src/ tests/

format:
    black src/ tests/

# Quick start
quickstart: install train dashboard