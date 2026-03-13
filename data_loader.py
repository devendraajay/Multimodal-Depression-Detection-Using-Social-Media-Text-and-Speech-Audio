"""
Data Loading and Preprocessing Module
Text-only classification - loads tweet JSON files and cleans text
"""

import json
import os
import re
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

# Download stopwords if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DataLoader:
    """Loads and preprocesses Twitter text data for depression detection"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize DataLoader
        
        Args:
            dataset_path: Path to the Dataset directory
        """
        self.dataset_path = os.path.normpath(dataset_path.strip()) if dataset_path else ""
        self.positive_tweet_path = os.path.join(dataset_path, "labeled", "positive", "data", "tweet")
        self.negative_tweet_path = os.path.join(dataset_path, "labeled", "negative", "data", "tweet")
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean tweet text by removing URLs, mentions, hashtags, emojis, and special characters
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@user)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        # Remove extra whitespace
        text = ' '.join(words)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def load_tweet_file(self, file_path: str) -> str:
        """
        Load a single tweet JSON file and extract text
        
        Args:
            file_path: Path to tweet JSON file
            
        Returns:
            Tweet text or empty string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('text', '')
        except Exception as e:
            return ""
    
    def load_text_data(self, label: int, max_samples: int = None) -> Tuple[List[str], List[int]]:
        """
        Load text data from positive or negative folder
        
        Args:
            label: 1 for positive (depressed), 0 for negative (non-depressed)
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []
        
        if label == 1:
            # Positive (depressed)
            folder_path = self.positive_tweet_path
            label_name = "positive (depressed)"
        else:
            # Negative (non-depressed)
            folder_path = self.negative_tweet_path
            label_name = "negative (non-depressed)"
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return texts, labels
        
        # Get all JSON files
        tweet_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        if max_samples:
            tweet_files = tweet_files[:max_samples]
        
        print(f"Loading {label_name} samples...")
        
        for tweet_file in tqdm(tweet_files, desc=f"Processing {label_name}"):
            tweet_path = os.path.join(folder_path, tweet_file)
            raw_text = self.load_tweet_file(tweet_path)
            
            if not raw_text:
                continue
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            if cleaned_text:  # Only add non-empty texts
                texts.append(cleaned_text)
                labels.append(label)
        
        return texts, labels
    
    def load_all_data(self, max_samples: int = None) -> Tuple[List[str], List[int]]:
        """
        Load all labeled data (positive and negative)
        
        Args:
            max_samples: Maximum number of samples per class (None for all)
            
        Returns:
            Tuple of (texts, labels)
        """
        # Load positive samples
        pos_texts, pos_labels = self.load_text_data(label=1, max_samples=max_samples)
        
        # Load negative samples
        neg_texts, neg_labels = self.load_text_data(label=0, max_samples=max_samples)
        
        # Combine
        all_texts = pos_texts + neg_texts
        all_labels = pos_labels + neg_labels
        
        print(f"\nTotal samples loaded: {len(all_texts)}")
        print(f"Positive (depressed): {sum(pos_labels)}")
        print(f"Negative (non-depressed): {len(neg_labels)}")
        
        return all_texts, all_labels
