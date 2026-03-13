"""
Feature Extraction Module
TF-IDF vectorization for classical ML models
"""

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class FeatureExtractor:
    """TF-IDF feature extraction for text classification"""
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: (min_n, max_n) - use unigrams and bigrams for better phrase signal
            min_df: Ignore terms that appear in fewer than this many documents (reduces noise)
            max_df: Ignore terms that appear in more than this fraction of documents (stopwords)
            sublinear_tf: Use 1+log(tf) instead of tf to dampen frequent terms
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.is_fitted = False
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fit vectorizer on texts and transform to TF-IDF features
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix [n_samples, max_features]
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features.toarray()
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform texts to TF-IDF features (vectorizer must be fitted)
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix [n_samples, max_features]
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        features = self.vectorizer.transform(texts)
        return features.toarray()
    
    def save(self, filepath: str):
        """
        Save vectorizer and config to file using pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(
                {
                    'vectorizer': self.vectorizer,
                    'max_features': self.max_features,
                    'ngram_range': getattr(self, 'ngram_range', (1, 1)),
                },
                f,
            )
        print(f"Vectorizer saved to {filepath}")

    def load(self, filepath: str):
        """
        Load vectorizer from file (supports both old format and new config format).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self.vectorizer = data['vectorizer']
            self.max_features = data.get('max_features', self.vectorizer.max_features)
            self.ngram_range = data.get('ngram_range', (1, 1))
        else:
            self.vectorizer = data
            self.max_features = getattr(self.vectorizer, 'max_features', 5000)
            self.ngram_range = getattr(self.vectorizer, 'ngram_range', (1, 1))
        self.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")
    
