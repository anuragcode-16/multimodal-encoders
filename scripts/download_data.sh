#!/bin/bash
# Download required data for FedFIM

set -e

echo "Downloading FedFIM data..."

# Create directories
mkdir -p data/raw data/processed data/synthetic

# Check if data exists
if [ -d "data/raw" ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "Data already exists in data/raw/"
    exit 0
fi

echo "Generating synthetic data..."
python -c "
import sys
sys.path.insert(0, '.')
from src.data_collection.market_data import MarketDataCollector
from src.data_collection.sentiment_data import SentimentDataCollector
from src.config import DATA_CONFIG

# Generate market data
market_collector = MarketDataCollector(use_cache=True)
for ticker in DATA_CONFIG.tickers:
    print(f'Generating data for {ticker}...')
    market_collector.fetch_yahoo_finance(ticker, DATA_CONFIG.start_date, DATA_CONFIG.end_date)

# Generate sentiment data
sentiment_collector = SentimentDataCollector(use_cache=True)
for ticker in DATA_CONFIG.tickers:
    print(f'Generating sentiment for {ticker}...')
    sentiment_collector.fetch_reddit_sentiment(ticker, DATA_CONFIG.start_date, DATA_CONFIG.end_date)
    sentiment_collector.fetch_news_sentiment(ticker, DATA_CONFIG.start_date, DATA_CONFIG.end_date)

print('Data generation complete!')
"

echo "Data download complete!"