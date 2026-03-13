"""
Inference Script for Unlabeled Data
Predicts depression probability for unlabeled timeline data
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm

from data_loader import DataLoader as DataLoaderClass
from model import DepressionDetectionModel, DepressionDataset


def predict_unlabeled(model, dataloader, device, threshold=0.5):
    """Predict depression probabilities for unlabeled data"""
    model.eval()
    all_probabilities = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            timeline_features = batch['timeline_features'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, timeline_features)
            probabilities = outputs.squeeze().cpu().numpy()
            
            all_probabilities.extend(probabilities)
    
    return np.array(all_probabilities)


def main():
    parser = argparse.ArgumentParser(description='Inference on Unlabeled Data')
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
                       help='Maximum number of samples to predict (None for all)')
    parser.add_argument('--output_path', type=str,
                       default='predictions_unlabeled.csv',
                       help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for depression classification')
    
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
    
    # Load unlabeled data
    print("Loading unlabeled data...")
    data_loader = DataLoaderClass(args.dataset_path)
    df = data_loader.load_unlabeled_data(max_samples=args.max_samples)
    
    print(f"Predicting on {len(df)} unlabeled samples")
    
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
    
    # Create dataset (with dummy labels for compatibility)
    dummy_labels = [0] * len(df)
    dataset = DepressionDataset(
        texts=df['text'].tolist(),
        timeline_features=timeline_features.tolist(),
        labels=dummy_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Predict
    print("Running inference...")
    probabilities = predict_unlabeled(model, dataloader, device, args.threshold)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'text': df['text'].values,
        'depression_probability': probabilities,
        'predicted_label': (probabilities > args.threshold).astype(int),
        'risk_level': pd.cut(probabilities, 
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High'])
    })
    
    # Save predictions
    predictions_df.to_csv(args.output_path, index=False)
    print(f"\nPredictions saved to {args.output_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("PREDICTION STATISTICS")
    print("="*50)
    print(f"Total samples: {len(predictions_df)}")
    print(f"Predicted as Depressed: {sum(predictions_df['predicted_label'] == 1)}")
    print(f"Predicted as Non-Depressed: {sum(predictions_df['predicted_label'] == 0)}")
    print(f"\nRisk Level Distribution:")
    print(predictions_df['risk_level'].value_counts())
    print(f"\nAverage Depression Probability: {probabilities.mean():.4f}")
    print(f"Std Deviation: {probabilities.std():.4f}")
    print("="*50)
    
    # Show high-risk samples
    high_risk = predictions_df[predictions_df['depression_probability'] > 0.7]
    if len(high_risk) > 0:
        print(f"\nHigh-Risk Samples (Probability > 0.7): {len(high_risk)}")
        print("\nTop 10 High-Risk Samples:")
        for idx, row in high_risk.nlargest(10, 'depression_probability').iterrows():
            print(f"\nProbability: {row['depression_probability']:.4f}")
            print(f"Text: {row['text'][:200]}...")


if __name__ == '__main__':
    main()
