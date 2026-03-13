"""
Evaluation Script for All Models
Evaluates and compares all trained models
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import os
import json
import pickle

from data_loader import DataLoader as DataLoaderClass
from feature_extraction import FeatureExtractor
from models_ml import LogisticRegressionScratch, SVMScratch
from model_bert import BERTDepressionModel, BERTDataset, SimpleTokenizer
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_all_models(dataset_path, model_dir='models', max_samples=None, test_size=0.2, random_state=42):
    """Evaluate all trained models"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    data_loader = DataLoaderClass(dataset_path)
    texts, labels = data_loader.load_all_data(max_samples=max_samples)
    
    # Train-test split (same as training)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Load TF-IDF vectorizer
    print("\nLoading TF-IDF vectorizer...")
    feature_extractor = FeatureExtractor()
    feature_extractor.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    X_test_tfidf = feature_extractor.transform(X_test)
    
    results = {}
    
    # Evaluate Logistic Regression (custom) if saved
    lr_path = os.path.join(model_dir, 'lr_model.pkl')
    if os.path.exists(lr_path):
        print("\n" + "="*60)
        print("EVALUATING: Logistic Regression")
        print("="*60)
        lr_model = LogisticRegressionScratch()
        lr_model.load(lr_path)
        results['Logistic_Regression'] = lr_model.evaluate(X_test_tfidf, y_test)
        y_pred_lr = lr_model.predict(X_test_tfidf)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_lr, target_names=['Non-Depressed', 'Depressed']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_lr))
    
    # Evaluate SVM if saved
    svm_path = os.path.join(model_dir, 'svm_model.pkl')
    if os.path.exists(svm_path):
        print("\n" + "="*60)
        print("EVALUATING: SVM")
        print("="*60)
        svm_model = SVMScratch()
        svm_model.load(svm_path)
        results['SVM'] = svm_model.evaluate(X_test_tfidf, y_test)
        y_pred_svm = svm_model.predict(X_test_tfidf)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_svm, target_names=['Non-Depressed', 'Depressed']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_svm))
    
    # Evaluate BERT
    print("\n" + "="*60)
    print("EVALUATING: BERT Model")
    print("="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    tokenizer = SimpleTokenizer.load(os.path.join(model_dir, 'bert_tokenizer.json'))
    
    checkpoint = torch.load(os.path.join(model_dir, 'bert_model.pt'), map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_config = checkpoint.get('model_config', {})
        max_length = int(checkpoint.get('max_length', model_config.get('max_length', 128)))
        bert_model = BERTDepressionModel(**model_config).to(device)
        bert_model.load_state_dict(checkpoint['state_dict'])
    else:
        max_length = 128
        bert_model = BERTDepressionModel(vocab_size=tokenizer.vocab_size).to(device)
        bert_model.load_state_dict(checkpoint)
    test_dataset = BERTDataset(X_test, y_test, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    bert_model.eval()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = bert_model(input_ids, attention_mask)
            predictions = (outputs.squeeze() > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    results['BERT'] = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, zero_division=0)
    }
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Non-Depressed', 'Depressed']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<35} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate All Models')
    parser.add_argument('--dataset_path', type=str,
                       default='Dataset_MDDL (1)/Dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per class')
    
    args = parser.parse_args()
    
    results = evaluate_all_models(
        args.dataset_path,
        args.model_dir,
        args.max_samples
    )
    
    # Save results
    results_path = os.path.join(args.model_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation results saved to {results_path}")


if __name__ == '__main__':
    main()
