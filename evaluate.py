"""
Evaluation Script for Depression Detection Model
Computes Accuracy, Precision, Recall, and F1-score
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import argparse
import os
import pickle

from data_loader import DataLoader as DataLoaderClass
from model import DepressionDetectionModel, DepressionDataset


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            timeline_features = batch['timeline_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, timeline_features)
            probabilities = outputs.squeeze().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_probabilities.extend(probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Depression Detection Model')
    parser.add_argument('--dataset_path', type=str,
                       default='Dataset_MDDL (1)/Dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str,
                       default='models/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for BERT')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'bert-base-uncased')
    max_length = checkpoint.get('max_length', args.max_length)
    
    # Initialize model
    print("Loading model...")
    model = DepressionDetectionModel(
        bert_model_name=bert_model_name,
        num_timeline_features=4,
        hidden_dim=256,
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    print("Loading data...")
    data_loader = DataLoaderClass(args.dataset_path)
    df, labels = data_loader.load_labeled_data(max_samples=args.max_samples)
    
    print(f"Evaluating on {len(df)} samples")
    
    # Prepare timeline features
    timeline_feature_cols = ['posting_frequency', 'avg_temporal_gap', 'num_tweets', 'sentiment_trend']
    timeline_features = df[timeline_feature_cols].values
    
    # Load and apply scaler
    scaler_path = os.path.join(os.path.dirname(args.model_path), 'timeline_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        timeline_features = scaler.transform(timeline_features)
    else:
        print("Warning: Scaler not found. Using unscaled features.")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # Create dataset
    dataset = DepressionDataset(
        texts=df['text'].tolist(),
        timeline_features=timeline_features.tolist(),
        labels=labels.tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, dataloader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(results['labels'], results['predictions'],
                                target_names=['Non-Depressed', 'Depressed']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(results['labels'], results['predictions'])
    print(f"                Predicted")
    print(f"              Non-Dep  Depressed")
    print(f"Actual Non-Dep  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"      Depressed {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Explain low accuracy reasons (for viva)
    print("\n" + "="*50)
    print("NOTES ON POTENTIAL LOW ACCURACY:")
    print("="*50)
    print("1. Class Imbalance: Dataset may have unequal distribution")
    print("2. Noisy Social Media Text: Informal language, typos, slang")
    print("3. Limited Dataset Size: May need more training data")
    print("4. Text-only Modality: No audio/video features available")
    print("5. Depression is Complex: Hard to detect from text alone")
    print("="*50)


if __name__ == '__main__':
    main()
