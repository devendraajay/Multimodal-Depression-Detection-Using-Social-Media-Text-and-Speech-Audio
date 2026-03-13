"""
Multimodal Depression Detection Model
BERT Text Encoder + Timeline Features + Fusion Layer + Classifier
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


class DepressionDetectionModel(nn.Module):
    """
    Multimodal model for depression detection combining:
    - BERT embeddings for text
    - Timeline behavioral features
    """
    
    def __init__(self, 
                 bert_model_name: str = 'bert-base-uncased',
                 num_timeline_features: int = 4,
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 num_classes: int = 1):
        """
        Initialize the model
        
        Args:
            bert_model_name: Name of the pretrained BERT model
            num_timeline_features: Number of timeline features (posting_frequency, avg_temporal_gap, num_tweets, sentiment_trend)
            hidden_dim: Hidden dimension for fusion layers
            dropout: Dropout rate
            num_classes: Number of output classes (1 for binary classification)
        """
        super(DepressionDetectionModel, self).__init__()
        
        # BERT text encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for bert-base-uncased
        
        # Freeze BERT parameters (optional - can be unfrozen for fine-tuning)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # Timeline feature processing
        self.timeline_fc = nn.Sequential(
            nn.Linear(num_timeline_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # Fusion layer: concatenate BERT embeddings + timeline features
        fusion_input_dim = bert_hidden_size + 32
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
            nn.Sigmoid()  # For binary classification
        )
    
    def forward(self, input_ids, attention_mask, timeline_features):
        """
        Forward pass
        
        Args:
            input_ids: Tokenized input text [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            timeline_features: Timeline features [batch_size, num_timeline_features]
            
        Returns:
            Output probabilities [batch_size, num_classes]
        """
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token embedding (first token)
        # Shape: [batch_size, bert_hidden_size]
        text_embedding = bert_outputs.last_hidden_state[:, 0, :]
        
        # Process timeline features
        # Shape: [batch_size, 32]
        timeline_embedding = self.timeline_fc(timeline_features)
        
        # Fusion: concatenate text and timeline embeddings
        # Shape: [batch_size, bert_hidden_size + 32]
        fused_features = torch.cat([text_embedding, timeline_embedding], dim=1)
        
        # Classification
        # Shape: [batch_size, num_classes]
        output = self.classifier(fused_features)
        
        return output


class DepressionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for depression detection
    """
    
    def __init__(self, texts, timeline_features, labels, tokenizer, max_length=128):
        """
        Initialize dataset
        
        Args:
            texts: List of text strings
            timeline_features: List of timeline feature arrays
            labels: List of labels (0 or 1)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.timeline_features = timeline_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        timeline_feat = self.timeline_features[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'timeline_features': torch.tensor(timeline_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
