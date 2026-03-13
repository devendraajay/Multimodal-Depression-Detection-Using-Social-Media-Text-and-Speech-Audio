"""
Depression Detection Prediction Script
Loads saved models (BERT, Logistic Regression, SVM) and makes predictions on new text.
"""

import os
import sys
import torch
import numpy as np
import argparse
import json

from feature_extraction import FeatureExtractor
from models_ml import LogisticRegressionScratch, SVMScratch, probability_uncertainty
from model_bert import BERTDepressionModel, SimpleTokenizer


def get_device():
    """Get available device (GPU or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda'), f"GPU ({torch.cuda.get_device_name(0)})"
    return torch.device('cpu'), "CPU"


def load_models(model_dir='models'):
    """Load all three models"""
    models = {}
    device, device_label = get_device()
    
    # Load TF-IDF vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        vectorizer = FeatureExtractor()
        vectorizer.load(vectorizer_path)
        models['vectorizer'] = vectorizer
        print("TF-IDF vectorizer loaded")
    else:
        print(f"WARNING: TF-IDF vectorizer not found at {vectorizer_path}")
        models['vectorizer'] = None
    
    # Load BERT model
    bert_path = os.path.join(model_dir, 'bert_model.pt')
    tokenizer_path = os.path.join(model_dir, 'bert_tokenizer.json')
    if os.path.exists(bert_path):
        try:
            tokenizer = SimpleTokenizer.load(tokenizer_path)
            checkpoint = torch.load(bert_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model_config = checkpoint.get('model_config', {})
                bert_model = BERTDepressionModel(**model_config).to(device)
                bert_model.load_state_dict(checkpoint['state_dict'])
                models['bert_max_length'] = int(checkpoint.get('max_length', model_config.get('max_length', 128)))
            else:
                bert_model = BERTDepressionModel(vocab_size=tokenizer.vocab_size).to(device)
                bert_model.load_state_dict(checkpoint)
                models['bert_max_length'] = 128
            bert_model.eval()
            models['bert'] = bert_model
            models['bert_tokenizer'] = tokenizer
            models['bert_device'] = device
            print(f"BERT model loaded (running on {device_label})")
        except Exception as e:
            print(f"WARNING: Error loading BERT model: {e}")
            models['bert'] = None
    else:
        print(f"WARNING: BERT model not found at {bert_path}")
        models['bert'] = None
    
    # Load Logistic Regression (custom)
    lr_path = os.path.join(model_dir, 'lr_model.pkl')
    if os.path.exists(lr_path):
        try:
            lr_model = LogisticRegressionScratch()
            lr_model.load(lr_path)
            models['lr'] = lr_model
            print("Logistic Regression loaded")
        except Exception as e:
            print(f"WARNING: Error loading LR: {e}")
            models['lr'] = None
    else:
        print("WARNING: Logistic Regression model not found")
        models['lr'] = None
    
    # Load SVM model
    svm_path = os.path.join(model_dir, 'svm_model.pkl')
    if os.path.exists(svm_path):
        try:
            svm_model = SVMScratch()
            svm_model.load(svm_path)
            models['svm'] = svm_model
            print("SVM model loaded")
        except Exception as e:
            print(f"WARNING: Error loading SVM: {e}")
            models['svm'] = None
    else:
        print(f"WARNING: SVM model not found at {svm_path}")
        models['svm'] = None
    
    return models


def predict_bert(model, tokenizer, device, text, max_length=128):
    """Predict depression probability using BERT"""
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    probability = output.squeeze().cpu().item()
    uncertainty = probability_uncertainty(probability)
    return float(probability), float(uncertainty)


def predict_ml(vectorizer, model, text):
    """Predict depression probability using ML model (LR or SVM)"""
    X = vectorizer.transform([text])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            probability = float(proba[0, 1])
        else:
            probability = float(proba[0])
        uncertainty = probability_uncertainty(probability)
        return probability, uncertainty
    # Fallback to predict
    pred = model.predict(X)[0]
    probability = float(pred)
    uncertainty = probability_uncertainty(probability)
    return probability, uncertainty


def predict_all_models(text, models):
    """Predict using all available models"""
    results = {}
    
    # BERT prediction
    if models.get('bert') is not None:
        try:
            bert_prob, bert_uncertainty = predict_bert(
                models['bert'],
                models['bert_tokenizer'],
                models['bert_device'],
                text,
                max_length=models.get('bert_max_length', 128)
            )
            results['BERT'] = {
                'probability': bert_prob,
                'uncertainty': bert_uncertainty,
                'confidence': 1.0 - bert_uncertainty,
                'prediction': get_depression_label(bert_prob)
            }
        except Exception as e:
            print(f"Error in BERT prediction: {e}")
            results['BERT'] = None
    
    # Logistic Regression prediction
    if models.get('lr') is not None and models.get('vectorizer') is not None:
        try:
            lr_prob, lr_uncertainty = predict_ml(models['vectorizer'], models['lr'], text)
            results['Logistic_Regression'] = {
                'probability': lr_prob,
                'uncertainty': lr_uncertainty,
                'confidence': 1.0 - lr_uncertainty,
                'prediction': get_depression_label(lr_prob)
            }
        except Exception as e:
            print(f"Error in LR prediction: {e}")
            results['Logistic_Regression'] = None
    
    # SVM prediction
    if models.get('svm') is not None and models.get('vectorizer') is not None:
        try:
            svm_prob, svm_uncertainty = predict_ml(models['vectorizer'], models['svm'], text)
            results['SVM'] = {
                'probability': svm_prob,
                'uncertainty': svm_uncertainty,
                'confidence': 1.0 - svm_uncertainty,
                'prediction': get_depression_label(svm_prob)
            }
        except Exception as e:
            print(f"Error in SVM prediction: {e}")
            results['SVM'] = None
    
    # Calculate combined average
    probs = [r['probability'] for r in results.values() if r is not None]
    if probs:
        combined_prob = np.mean(probs)
        combined_uncertainty = probability_uncertainty(combined_prob)
        results['Combined'] = {
            'probability': combined_prob,
            'uncertainty': combined_uncertainty,
            'confidence': 1.0 - combined_uncertainty,
            'model_disagreement': float(np.std(probs)),
            'prediction': get_depression_label(combined_prob)
        }
    
    return results


def get_depression_label(probability):
    """Binary: Depressed or Not depressed (threshold 0.5)."""
    return "Depressed" if probability >= 0.5 else "Not depressed"


def main():
    parser = argparse.ArgumentParser(description='Predict depression from text using saved models')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text (one per line)')
    parser.add_argument('--json', type=str, help='JSON file with text data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing saved models')
    parser.add_argument('--output', type=str, help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Load models
    print("="*60)
    print("Loading Models...")
    print("="*60)
    models = load_models(args.model_dir)
    
    # Check if at least one model is loaded
    if models.get('bert') is None and models.get('lr') is None and models.get('svm') is None:
        print("\nERROR: No models could be loaded!")
        print("Please train models first using:")
        print("  python train_models.py --dataset_path \"Dataset_MDDL (1)/Dataset\"")
        return
    
    # Get input text
    texts_to_analyze = []
    
    if args.text:
        texts_to_analyze.append(args.text)
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts_to_analyze = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.json:
        try:
            with open(args.json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract text from JSON (similar to app.py logic)
                if isinstance(data, dict):
                    if 'text' in data:
                        texts_to_analyze.append(data['text'])
                    elif 'tweets' in data:
                        for tweet in data['tweets']:
                            if isinstance(tweet, dict) and 'text' in tweet:
                                texts_to_analyze.append(tweet['text'])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            texts_to_analyze.append(item['text'])
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("Interactive Mode - Enter text to analyze")
        print("(Type 'quit' to exit)")
        print("="*60)
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() == 'quit':
                break
            if text:
                texts_to_analyze.append(text)
                break
    
    if not texts_to_analyze:
        print("No text provided. Use --text, --file, or --json arguments.")
        return
    
    # Make predictions
    print("\n" + "="*60)
    print("Making Predictions...")
    print("="*60)
    
    all_results = []
    for idx, text in enumerate(texts_to_analyze, 1):
        print(f"\n--- Text {idx}/{len(texts_to_analyze)} ---")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        results = predict_all_models(text, models)
        all_results.append({
            'text': text,
            'predictions': results
        })
        
        # Display results
        print("\nPredictions:")
        print("-" * 60)
        for model_name, result in results.items():
            if result is not None:
                print(f"{model_name:20s}: {result['prediction']} (confidence: {result['probability']:.2%})")
                print(f"{'':20s}  uncertainty={result['uncertainty']:.4f}, confidence={result['confidence']:.4f}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")
    
    # Summary
    if len(texts_to_analyze) > 1:
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        
        for model_name in ['BERT', 'Logistic_Regression', 'SVM', 'Combined']:
            probs = []
            for result_dict in all_results:
                if model_name in result_dict['predictions'] and result_dict['predictions'][model_name] is not None:
                    probs.append(result_dict['predictions'][model_name]['probability'])
            
            if probs:
                print(f"{model_name:20s}: Mean={np.mean(probs):.4f}, Std={np.std(probs):.4f}, "
                      f"Min={np.min(probs):.4f}, Max={np.max(probs):.4f}")


if __name__ == '__main__':
    main()
