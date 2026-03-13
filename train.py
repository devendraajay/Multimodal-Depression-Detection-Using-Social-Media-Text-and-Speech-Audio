"""
Training Script for Depression Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import argparse

from data_loader import DataLoader as DataLoaderClass
from model import DepressionDetectionModel, DepressionDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        timeline_features = batch['timeline_features'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, timeline_features)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            timeline_features = batch['timeline_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, timeline_features)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions
            predictions = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Depression Detection Model')
    parser.add_argument('--dataset_path', type=str, 
                       default='Dataset_MDDL (1)/Dataset',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (reduce if GPU memory is limited)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for BERT')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to load (None for all)')
    parser.add_argument('--model_save_path', type=str, default='models',
                       help='Path to save trained model')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                       help='BERT model name')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = DataLoaderClass(args.dataset_path)
    df, labels = data_loader.load_labeled_data(max_samples=args.max_samples)
    
    print(f"Loaded {len(df)} samples")
    print(f"Positive samples: {sum(labels == 1)}")
    print(f"Negative samples: {sum(labels == 0)}")
    
    # Prepare timeline features
    timeline_feature_cols = ['posting_frequency', 'avg_temporal_gap', 'num_tweets', 'sentiment_trend']
    timeline_features = df[timeline_feature_cols].values
    
    # Normalize timeline features
    scaler = StandardScaler()
    timeline_features = scaler.fit_transform(timeline_features)
    
    # Save scaler for inference
    import pickle
    with open(os.path.join(args.model_save_path, 'timeline_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # Create dataset
    dataset = DepressionDataset(
        texts=df['text'].tolist(),
        timeline_features=timeline_features.tolist(),
        labels=labels.tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = DepressionDetectionModel(
        bert_model_name=args.bert_model,
        num_timeline_features=len(timeline_feature_cols),
        hidden_dim=256,
        dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting training...")
    best_val_accuracy = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'bert_model_name': args.bert_model,
                'max_length': args.max_length
            }, os.path.join(args.model_save_path, 'best_model.pt'))
            print(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    main()
