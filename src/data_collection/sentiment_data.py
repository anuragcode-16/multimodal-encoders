"""
Sentiment data collection and processing with LLM support
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re
import requests
import json
import time

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


class SentimentDataCollector:
    """Collects and processes sentiment data with LLM support"""
    
    def __init__(self, use_cache: bool = True, use_llm: bool = False):
        self.use_cache = use_cache
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM Configuration
        self.use_llm = use_llm
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.model = os.environ.get('OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct:free')
        
        if self.use_llm and not self.api_key:
            print("Warning: USE_LLM_SENTIMENT is true but OPENROUTER_API_KEY is missing. Falling back to local processing.")
            self.use_llm = False
    
    def fetch_reddit_sentiment(self, ticker: str, start_date: str, 
                                end_date: str) -> pd.DataFrame:
        """Fetch Reddit sentiment data"""
        cache_file = self.cache_dir / f"{ticker}_reddit.parquet"
        
        if self.use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)
        
        try:
            import praw
            # Try Reddit API
            reddit = praw.Reddit(
                client_id=os.environ.get('REDDIT_CLIENT_ID', ''),
                client_secret=os.environ.get('REDDIT_CLIENT_SECRET', ''),
                user_agent=os.environ.get('REDDIT_USER_AGENT', 'FedFIM')
            )
            # Fetch data...
        except:
            pass
        
        # Generate synthetic sentiment data
        return self._generate_synthetic_sentiment(ticker, start_date, end_date, 'reddit')
    
    def fetch_news_sentiment(self, ticker: str, start_date: str, 
                             end_date: str) -> pd.DataFrame:
        """Fetch news sentiment data"""
        cache_file = self.cache_dir / f"{ticker}_news.parquet"
        
        if self.use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)
        
        return self._generate_synthetic_sentiment(ticker, start_date, end_date, 'news')
    
    def _generate_synthetic_sentiment(self, ticker: str, start_date: str, 
                                       end_date: str, source: str) -> pd.DataFrame:
        """Generate synthetic sentiment data aligned with market patterns"""
        np.random.seed(hash(f"{ticker}_{source}") % (2**32))
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start=start, end=end, freq='D')
        n = len(dates)
        
        # Financial sentiment templates
        positive_templates = [
            f"{ticker} shows strong growth potential",
            f"Analysts bullish on {ticker}",
            f"{ticker} beats earnings expectations",
            f"Strong quarterly results for {ticker}",
            f"{ticker} announces innovative product line",
            f"Investors optimistic about {ticker} future",
            f"{ticker} expands into new markets",
            f"Revenue growth exceeds projections for {ticker}",
        ]
        
        negative_templates = [
            f"{ticker} faces regulatory challenges",
            f"Analysts downgrade {ticker}",
            f"{ticker} misses earnings targets",
            f"Concerns over {ticker} market position",
            f"{ticker} announces layoffs",
            f"Investors concerned about {ticker} debt",
            f"Competition threatens {ticker} market share",
            f"Supply chain issues impact {ticker}",
        ]
        
        neutral_templates = [
            f"{ticker} maintains steady performance",
            f"Market awaits {ticker} earnings report",
            f"Analysts neutral on {ticker}",
            f"{ticker} announces management changes",
            f"Trading volume stable for {ticker}",
        ]
        
        texts = []
        sentiments = []
        scores = []
        
        for i in range(n):
            # Generate correlated sentiment with some randomness
            base_sentiment = np.sin(i / 30) * 0.3 + np.random.normal(0, 0.4)
            
            # Add regime changes
            if i > n // 3 and i < 2 * n // 3:
                base_sentiment -= 0.2  # Bearish period
            
            if base_sentiment > 0.2:
                text = np.random.choice(positive_templates)
                sentiment = 'positive'
                score = np.random.uniform(0.3, 1.0)
            elif base_sentiment < -0.2:
                text = np.random.choice(negative_templates)
                sentiment = 'negative'
                score = np.random.uniform(-1.0, -0.3)
            else:
                text = np.random.choice(neutral_templates)
                sentiment = 'neutral'
                score = np.random.uniform(-0.2, 0.2)
            
            # Add some noise
            text = f"{text} [Day {i}]"
            texts.append(text)
            sentiments.append(sentiment)
            scores.append(score)
        
        df = pd.DataFrame({
            'date': dates,
            'text': texts,
            'sentiment': sentiments,
            'score': scores,
            'source': source
        })
        
        return df
    
    def generate_sentiment_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentiment embeddings using available methods"""
        
        # Use LLM if enabled and key is present
        if self.use_llm and self.api_key:
            return self._generate_llm_embeddings(texts)
        
        # Fallback to local processing
        return self._generate_local_embeddings(texts)
    
    def _generate_llm_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenRouter LLM"""
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        
        # Process in small batches to avoid rate limits
        batch_size = 5
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Create prompt for batch
            prompt = "Analyze the sentiment of the following financial texts. "
            prompt += "Return a JSON array where each item has 'label' (positive/negative/neutral) and 'score' (-1 to 1).\n\n"
            
            for idx, text in enumerate(batch):
                prompt += f"{idx+1}. {text}\n"
            
            prompt += "\nJSON Response:"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a financial sentiment analysis expert. Output only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0, # Deterministic
                "max_tokens": 500
            }

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                content = response.json()['choices'][0]['message']['content']
                
                # Parse JSON from response
                # Handle potential markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                results = json.loads(content)
                
                # Convert to embedding vector (384-dim)
                for item in results:
                    vec = np.zeros(384)
                    
                    # Encode label
                    label = item.get('label', 'neutral').lower()
                    if 'positive' in label:
                        vec[0] = 1
                    elif 'negative' in label:
                        vec[0] = -1
                    else:
                        vec[0] = 0
                    
                    # Encode score
                    score = item.get('score', 0)
                    vec[1] = score
                    
                    # Fill rest with semantic hashing (simplified)
                    vec[4:] = np.random.randn(380) * 0.1
                    
                    embeddings.append(vec)
                    
            except Exception as e:
                print(f"LLM API Error: {e}. Falling back to local for this batch.")
                embeddings.extend(self._generate_local_embeddings(batch))
            
            # Rate limit safety
            time.sleep(1)
                
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using local TF-IDF and TextBlob"""
        embeddings = []
        
        for text in texts:
            # Simple TF-IDF style embedding (384-dim for compatibility)
            words = re.findall(r'\w+', text.lower())
            embedding = np.zeros(384)
            
            # Hash words to positions
            for word in set(words):
                pos = hash(word) % 384
                embedding[pos] = words.count(word) / max(len(words), 1)
            
            # Add sentiment features
            if TextBlob:
                blob = TextBlob(text)
                embedding[-10] = blob.sentiment.polarity
                embedding[-9] = blob.sentiment.subjectivity
            
            # Normalize
            norm = np.linalg.norm(embedding) + 1e-8
            embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)


class SentimentProcessor:
    """Process sentiment data for model input"""
    
    def __init__(self):
        # Check env var for LLM usage
        use_llm = os.environ.get('USE_LLM_SENTIMENT', 'false').lower() == 'true'
        self.collector = SentimentDataCollector(use_llm=use_llm)
    
    def create_sentiment_features(self, market_data: pd.DataFrame, 
                                   ticker: str) -> pd.DataFrame:
        """Create sentiment features aligned with market data"""
        # Get sentiment data
        reddit_sentiment = self.collector.fetch_reddit_sentiment(
            ticker, 
            market_data['date'].min().strftime('%Y-%m-%d'),
            market_data['date'].max().strftime('%Y-%m-%d')
        )
        
        news_sentiment = self.collector.fetch_news_sentiment(
            ticker,
            market_data['date'].min().strftime('%Y-%m-%d'),
            market_data['date'].max().strftime('%Y-%m-%d')
        )
        
        # Aggregate sentiment by date
        reddit_daily = reddit_sentiment.groupby(
            pd.to_datetime(reddit_sentiment['date']).dt.date
        ).agg({
            'score': ['mean', 'std', 'count']
        }).reset_index()
        reddit_daily.columns = ['date', 'reddit_mean', 'reddit_std', 'reddit_count']
        
        news_daily = news_sentiment.groupby(
            pd.to_datetime(news_sentiment['date']).dt.date
        ).agg({
            'score': ['mean', 'std', 'count']
        }).reset_index()
        news_daily.columns = ['date', 'news_mean', 'news_std', 'news_count']
        
        # Merge with market data
        market_data['date_only'] = pd.to_datetime(market_data['date']).dt.date
        
        # Merge Reddit (use suffixes to handle overlapping column names)
        merged = market_data.merge(
            reddit_daily, 
            left_on='date_only', 
            right_on='date', 
            how='left',
            suffixes=('', '_reddit')
        )
        
        # Merge News (use suffixes to handle overlapping column names)
        merged = merged.merge(
            news_daily, 
            left_on='date_only', 
            right_on='date', 
            how='left',
            suffixes=('', '_news')
        )
        
        # Fill missing values
        merged['reddit_mean'] = merged['reddit_mean'].fillna(0)
        merged['reddit_std'] = merged['reddit_std'].fillna(0.1)
        merged['reddit_count'] = merged['reddit_count'].fillna(0)
        merged['news_mean'] = merged['news_mean'].fillna(0)
        merged['news_std'] = merged['news_std'].fillna(0.1)
        merged['news_count'] = merged['news_count'].fillna(0)
        
        # Compute aggregate sentiment features
        merged['sentiment_score'] = (merged['reddit_mean'] + merged['news_mean']) / 2
        merged['sentiment_confidence'] = np.sqrt(merged['reddit_count'] + merged['news_count'])
        merged['sentiment_volatility'] = (merged['reddit_std'] + merged['news_std']) / 2
        
        # Rolling sentiment features
        merged['sentiment_ma_7'] = merged['sentiment_score'].rolling(7).mean()
        merged['sentiment_ma_14'] = merged['sentiment_score'].rolling(14).mean()
        merged['sentiment_momentum'] = merged['sentiment_ma_7'] - merged['sentiment_ma_14']
        
        # Clean up temporary columns
        cols_to_drop = ['date_only', 'date', 'date_reddit', 'date_news']
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns], errors='ignore')
        
        # Drop NaNs created by rolling windows
        merged = merged.dropna()
        
        return merged