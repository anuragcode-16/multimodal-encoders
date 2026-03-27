"""
Sentiment feature engineering
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import re


class SentimentFeatureEngineer:
    """
    Process and engineer sentiment features
    """
    
    def __init__(self):
        self.financial_keywords = {
            'positive': [
                'bullish', 'growth', 'profit', 'gain', 'surge', 'rally',
                'outperform', 'upgrade', 'buy', 'strong', 'beat', 'exceed',
                'positive', 'optimistic', 'soar', 'jump', 'climb', 'rise',
                'success', 'opportunity', 'potential', 'record', 'high'
            ],
            'negative': [
                'bearish', 'loss', 'decline', 'fall', 'drop', 'crash',
                'underperform', 'downgrade', 'sell', 'weak', 'miss', 'below',
                'negative', 'pessimistic', 'plunge', 'sink', 'tumble', 'slide',
                'risk', 'concern', 'threat', 'uncertainty', 'low', 'bankruptcy'
            ],
            'uncertainty': [
                'volatile', 'uncertain', 'fluctuate', 'mixed', 'unclear',
                'ambiguous', 'caution', 'wait', 'observe', 'hesitant'
            ]
        }
    
    def extract_features_from_text(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extract sentiment features from text data
        """
        features = pd.DataFrame()
        
        # Basic text features
        features['text_length'] = texts.str.len()
        features['word_count'] = texts.str.split().str.len()
        features['exclamation_count'] = texts.str.count('!')
        features['question_count'] = texts.str.count(r'\?')
        features['uppercase_ratio'] = texts.apply(self._uppercase_ratio)
        
        # Keyword counts
        features['positive_word_count'] = texts.apply(
            lambda x: self._count_keywords(x, self.financial_keywords['positive'])
        )
        features['negative_word_count'] = texts.apply(
            lambda x: self._count_keywords(x, self.financial_keywords['negative'])
        )
        features['uncertainty_word_count'] = texts.apply(
            lambda x: self._count_keywords(x, self.financial_keywords['uncertainty'])
        )
        
        # Sentiment ratios
        total_keywords = (features['positive_word_count'] + 
                         features['negative_word_count'] + 
                         features['uncertainty_word_count'] + 1e-10)
        
        features['positive_ratio'] = features['positive_word_count'] / total_keywords
        features['negative_ratio'] = features['negative_word_count'] / total_keywords
        features['sentiment_score'] = (features['positive_word_count'] - 
                                       features['negative_word_count']) / total_keywords
        
        # TextBlob sentiment (if available)
        try:
            from textblob import TextBlob
            features['textblob_polarity'] = texts.apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            features['textblob_subjectivity'] = texts.apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity
            )
        except ImportError:
            features['textblob_polarity'] = 0
            features['textblob_subjectivity'] = 0
        
        return features
    
    def aggregate_sentiment_by_date(self, df: pd.DataFrame, 
                                    date_col: str = 'date',
                                    score_col: str = 'sentiment_score') -> pd.DataFrame:
        """
        Aggregate sentiment scores by date
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        
        agg = df.groupby(date_col).agg({
            score_col: ['mean', 'std', 'count', 'min', 'max'],
            'positive_word_count': 'sum',
            'negative_word_count': 'sum'
        })
        
        agg.columns = ['_'.join(col) for col in agg.columns]
        agg = agg.reset_index()
        
        # Additional aggregated features
        agg['sentiment_range'] = agg[f'{score_col}_max'] - agg[f'{score_col}_min']
        agg['sentiment_confidence'] = agg[f'{score_col}_count'] / agg[f'{score_col}_count'].max()
        
        return agg
    
    def create_rolling_sentiment_features(self, df: pd.DataFrame,
                                          score_col: str = 'sentiment_score_mean',
                                          windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling sentiment features
        """
        df = df.copy()
        
        for window in windows:
            df[f'sentiment_ma_{window}'] = df[score_col].rolling(window).mean()
            df[f'sentiment_std_{window}'] = df[score_col].rolling(window).std()
            df[f'sentiment_momentum_{window}'] = (
                df[f'sentiment_ma_{window}'] - df[f'sentiment_ma_{window}'].shift(window)
            )
        
        # Sentiment acceleration
        df['sentiment_velocity'] = df[score_col].diff()
        df['sentiment_acceleration'] = df['sentiment_velocity'].diff()
        
        # Sentiment regime
        df['sentiment_regime'] = pd.cut(
            df[score_col],
            bins=[-np.inf, -0.3, 0.3, np.inf],
            labels=['bearish', 'neutral', 'bullish']
        )
        
        return df
    
    def _uppercase_ratio(self, text: str) -> float:
        if not text or len(text) == 0:
            return 0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        if not text:
            return 0
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """Return list of sentiment feature columns"""
        return [
            'text_length', 'word_count', 'exclamation_count', 'question_count',
            'uppercase_ratio', 'positive_word_count', 'negative_word_count',
            'uncertainty_word_count', 'positive_ratio', 'negative_ratio',
            'sentiment_score', 'textblob_polarity', 'textblob_subjectivity',
            'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_ma_14', 'sentiment_ma_30',
            'sentiment_velocity', 'sentiment_acceleration'
        ]