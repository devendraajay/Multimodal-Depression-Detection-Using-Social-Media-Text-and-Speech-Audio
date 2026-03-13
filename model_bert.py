"""
Train-from-scratch Transformer model for text classification.
No pretrained model weights are used.
"""

import json
import re
from collections import Counter

import torch
import torch.nn as nn


class SimpleTokenizer:
    """Lightweight word-level tokenizer trained from project data."""

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"

    def __init__(self, vocab=None, lower=True):
        self.lower = lower
        self.vocab = vocab or {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        text = str(text)
        if self.lower:
            text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit(self, texts, max_vocab_size=30000, min_freq=2):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        special_tokens = set(self.vocab.keys())
        sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, freq in sorted_items:
            if freq < min_freq:
                continue
            if token in special_tokens:
                continue
            if len(self.vocab) >= max_vocab_size:
                break
            self.vocab[token] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, text, max_length):
        tokens = self._tokenize(text)
        token_ids = [self.vocab[self.CLS_TOKEN]]
        token_ids.extend([self.vocab.get(t, self.vocab[self.UNK_TOKEN]) for t in tokens])
        token_ids.append(self.vocab[self.SEP_TOKEN])
        
        # Truncate to max_length
        token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)
        if len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids.extend([self.vocab[self.PAD_TOKEN]] * pad_len)
            attention_mask.extend([0] * pad_len)
        
        # Ensure all token IDs are valid (within vocab range)
        vocab_size = len(self.vocab)
        token_ids = [min(max(tid, 0), vocab_size - 1) for tid in token_ids]
        
        return token_ids, attention_mask

    def __call__(self, text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"):
        del truncation, padding
        input_ids, attention_mask = self.encode(text, max_length=max_length)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": [input_ids], "attention_mask": [attention_mask]}

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "lower": self.lower}, f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], lower=data.get("lower", True))


class BERTDepressionModel(nn.Module):
    """
    Transformer encoder model trained from scratch for depression detection.
    The class name is kept for compatibility with existing scripts.
    """

    def __init__(
        self,
        bert_model_name: str = "local-transformer",
        vocab_size: int = 30000,
        max_length: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.3,
    ):
        del bert_model_name
        super(BERTDepressionModel, self).__init__()
        
        # Store configuration
        self.model_config = {
            "vocab_size": vocab_size,
            "max_length": max_length,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        }
        
        # Validate configuration
        assert vocab_size > 0, "vocab_size must be positive"
        assert max_length > 0, "max_length must be positive"
        assert embed_dim > 0, "embed_dim must be positive"
        assert num_heads > 0, "num_heads must be positive"
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def get_config_dict(self):
        return dict(self.model_config)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        
        # Get model configuration
        max_pos = self.model_config.get('max_length', 128)
        vocab_size = self.model_config.get('vocab_size', 30000)
        
        # CRITICAL: Clamp input_ids to valid vocabulary range [0, vocab_size-1]
        # This prevents CUDA device-side assert errors
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # Ensure sequence length doesn't exceed max_length
        if seq_len > max_pos:
            input_ids = input_ids[:, :max_pos]
            attention_mask = attention_mask[:, :max_pos]
            seq_len = max_pos
        
        # Create position indices, clamped to valid range [0, max_length-1]
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, 0, max_pos - 1)

        # Token embeddings (input_ids already clamped)
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings (positions already clamped)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds
        
        # Create padding mask for transformer
        key_padding_mask = (attention_mask == 0)
        
        # Encode with transformer
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Pooling: weighted average using attention mask
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Classification
        output = self.classifier(self.dropout(pooled))
        output = self.sigmoid(output)
        return output


class BERTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for text classification model."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.float32),
        }
