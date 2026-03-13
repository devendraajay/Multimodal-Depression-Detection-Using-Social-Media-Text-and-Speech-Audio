"""
Test script for quick model testing
Tests all models with sample texts to verify they're working correctly
"""

import os
import re
import torch
import numpy as np
from feature_extraction import FeatureExtractor
from models_ml import LogisticRegressionScratch, SVMScratch, probability_uncertainty
from model_bert import BERTDepressionModel, SimpleTokenizer

# NLTK stopwords - must match training (data_loader.clean_text)
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    _STOP_WORDS = set(stopwords.words('english'))
except Exception:
    _STOP_WORDS = set()


def clean_text_for_ml(text):
    """Same cleaning as during training so LR/SVM get same feature distribution."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in _STOP_WORDS]
    text = ' '.join(words)
    text = ' '.join(text.split())
    return text.strip()


MODEL_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_model.pt')
BERT_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'bert_tokenizer.json')
BERT_MAX_LENGTH = 128


def get_device():
    """Get device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda'), f"GPU ({torch.cuda.get_device_name(0)})"
    return torch.device('cpu'), "CPU"


def load_bert_model():
    """Load BERT model and tokenizer"""
    device, device_label = get_device()
    
    try:
        tokenizer = SimpleTokenizer.load(BERT_TOKENIZER_PATH)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer: {e}")
    
    try:
        checkpoint = torch.load(BERT_MODEL_PATH, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load model checkpoint: {e}")
    
    # Load model with proper configuration
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_config = checkpoint.get('model_config', {})
        
        # CRITICAL: Use the vocab_size from the saved checkpoint, not tokenizer
        saved_vocab_size = checkpoint.get('vocab_size', None)
        if saved_vocab_size is None:
            saved_vocab_size = model_config.get('vocab_size', None)
        
        if saved_vocab_size is None:
            # Check the state_dict for token_embedding shape
            if 'token_embedding.weight' in checkpoint['state_dict']:
                saved_vocab_size = checkpoint['state_dict']['token_embedding.weight'].shape[0]
            else:
                raise ValueError("Cannot determine vocab_size from checkpoint")
        
        # Use saved vocab_size to ensure exact match
        model_config['vocab_size'] = saved_vocab_size
        
        # Ensure max_length is set
        if 'max_length' not in model_config:
            saved_max_length = checkpoint.get('max_length', BERT_MAX_LENGTH)
            model_config['max_length'] = saved_max_length
        
        print(f"Loading BERT model with vocab_size={saved_vocab_size}, tokenizer vocab_size={tokenizer.vocab_size}")
        
        # Create model with exact saved configuration
        model = BERTDepressionModel(**model_config).to(device)
        
        # Load state dict
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Model loaded successfully with strict=True")
        except RuntimeError as e:
            print(f"Warning: Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # Fallback: try to infer vocab_size from checkpoint
        if isinstance(checkpoint, dict):
            if 'token_embedding.weight' in checkpoint:
                saved_vocab_size = checkpoint['token_embedding.weight'].shape[0]
            else:
                saved_vocab_size = tokenizer.vocab_size
        else:
            saved_vocab_size = tokenizer.vocab_size
        
        model = BERTDepressionModel(
            vocab_size=saved_vocab_size,
            max_length=BERT_MAX_LENGTH
        ).to(device)
        
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model, tokenizer, device, device_label


def predict_bert(model, tokenizer, device, text):
    """Predict depression probability using BERT"""
    try:
        # Get model configuration - use model's vocab_size
        model_max_length = model.model_config.get("max_length", BERT_MAX_LENGTH)
        model_vocab_size = model.model_config.get('vocab_size')
        
        if model_vocab_size is None:
            raise ValueError("Model vocab_size not found")
        
        # Tokenize text
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=model_max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # CRITICAL: Map tokenizer token IDs to model vocab_size range
        tokenizer_vocab_size = tokenizer.vocab_size
        
        if tokenizer_vocab_size > model_vocab_size:
            # Tokenizer has more tokens - map out-of-range tokens to UNK
            unk_token_id = tokenizer.vocab.get(tokenizer.UNK_TOKEN, 1)
            input_ids = torch.where(
                input_ids >= model_vocab_size,
                torch.tensor(unk_token_id if unk_token_id < model_vocab_size else 1),
                input_ids
            )
            input_ids = torch.clamp(input_ids, 0, model_vocab_size - 1)
        else:
            # Tokenizer vocab is smaller or equal - just clamp to be safe
            input_ids = torch.clamp(input_ids, 0, model_vocab_size - 1)
        
        # Move to device after processing
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Ensure shapes match
        if attention_mask.shape != input_ids.shape:
            attention_mask = attention_mask[:, :input_ids.shape[1]]
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        # Extract probability safely
        if output.dim() > 1:
            probability = float(np.clip(output.squeeze().cpu().item(), 0.0, 1.0))
        else:
            probability = float(np.clip(output.cpu().item(), 0.0, 1.0))
        
        return probability
    except RuntimeError as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            # Fallback to CPU if CUDA error
            print(f"Warning: CUDA error detected, falling back to CPU: {e}")
            input_ids_cpu = input_ids.cpu() if 'input_ids' in locals() else None
            attention_mask_cpu = attention_mask.cpu() if 'attention_mask' in locals() else None
            if input_ids_cpu is not None:
                model_cpu = model.cpu()
                model_cpu.eval()
                with torch.no_grad():
                    output = model_cpu(input_ids_cpu, attention_mask_cpu)
                probability = float(np.clip(output.squeeze().item(), 0.0, 1.0))
                return probability
        raise RuntimeError(f"BERT prediction error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"BERT prediction error: {str(e)}")


def predict_ml(vectorizer, model, text):
    """Predict depression probability using ML model (LR/SVM)."""
    try:
        # CRITICAL: Clean text same as training so features match
        text_cleaned = clean_text_for_ml(text)
        if not text_cleaned:
            text_cleaned = text.strip().lower() or text
        # Transform cleaned text to TF-IDF features
        X = vectorizer.transform([text_cleaned])
        
        # Ensure X is numpy array (FeatureExtractor already returns array, but be safe)
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X)
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            proba = np.asarray(proba)
            
            if proba.ndim == 2:
                if proba.shape[1] >= 2:
                    return float(np.clip(proba[0, 1], 0.0, 1.0))
                else:
                    return float(np.clip(proba[0, 0], 0.0, 1.0))
            elif proba.ndim == 1:
                if len(proba) >= 2:
                    return float(np.clip(proba[1], 0.0, 1.0))
                else:
                    return float(np.clip(proba[0], 0.0, 1.0))
            else:
                return float(np.clip(proba[0], 0.0, 1.0))
        else:
            # Fallback: use predict
            pred = model.predict(X)
            if isinstance(pred, np.ndarray):
                pred_value = int(pred[0]) if len(pred) > 0 else 0
            else:
                pred_value = int(pred)
            probability = 0.8 if pred_value == 1 else 0.2
            return probability
    except Exception as e:
        raise RuntimeError(f"ML prediction error: {str(e)}")


def test_all_models():
    """Test all available models with sample texts"""
    
    print("="*70)
    print("MODEL TESTING SCRIPT")
    print("="*70)
    
    # Sample test texts
    test_texts = [
        ("Depressed example", "I feel so empty and hopeless. Nothing brings me joy anymore. I can't sleep and I don't want to get out of bed."),
        ("Non-depressed example", "Had a great day today! Went for a walk, met friends, and enjoyed the sunshine. Feeling grateful!"),
        ("Neutral example", "Just finished my work. Time to relax and watch some TV."),
    ]
    
    # Check which models are available
    has_bert = os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_TOKENIZER_PATH)
    has_vectorizer = os.path.exists(VECTORIZER_PATH)
    has_lr = os.path.exists(os.path.join(MODEL_DIR, 'lr_model.pkl'))
    has_svm = os.path.exists(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    
    print("\nAvailable Models:")
    print(f"  BERT: {'[OK]' if has_bert else '[NOT FOUND]'}")
    print(f"  Logistic Regression: {'[OK]' if has_lr else '[NOT FOUND]'}")
    print(f"  SVM: {'[OK]' if has_svm else '[NOT FOUND]'}")
    print(f"  Vectorizer: {'[OK]' if has_vectorizer else '[NOT FOUND]'}")
    
    if not has_bert and not has_vectorizer:
        print("\n[ERROR] No models found! Please train models first:")
        print('   python train_models.py --dataset_path "Dataset_MDDL (1)/Dataset"')
        return
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    bert_model, tokenizer, device, device_label = None, None, None, None
    vectorizer = None
    lr_model = None
    svm_model = None
    
    if has_bert:
        try:
            print("Loading BERT model...")
            bert_model, tokenizer, device, device_label = load_bert_model()
            print(f"[OK] BERT loaded on {device_label}")
        except Exception as e:
            print(f"[ERROR] Failed to load BERT: {e}")
    
    if has_vectorizer:
        try:
            print("Loading TF-IDF vectorizer...")
            vectorizer = FeatureExtractor()
            vectorizer.load(VECTORIZER_PATH)
            print("[OK] Vectorizer loaded")
            
            if has_lr:
                try:
                    print("Loading Logistic Regression...")
                    lr_model = LogisticRegressionScratch()
                    lr_model.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
                    print("[OK] Logistic Regression loaded")
                except Exception as e:
                    print(f"[ERROR] Failed to load LR: {e}")
            
            if has_svm:
                try:
                    print("Loading SVM...")
                    svm_model = SVMScratch()
                    svm_model.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
                    print("[OK] SVM loaded")
                except Exception as e:
                    print(f"[ERROR] Failed to load SVM: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to load vectorizer: {e}")
            return
    
    # Test predictions
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    for label, text in test_texts:
        print(f"\nTest: {label}")
        print(f"   Text: {text[:80]}...")
        print("-" * 70)
        
        results = {}
        
        # BERT prediction
        if bert_model is not None:
            try:
                bert_prob = predict_bert(bert_model, tokenizer, device, text)
                results['BERT'] = bert_prob
                label = "Depressed" if bert_prob >= 0.5 else "Not depressed"
                print(f"   BERT:           {label} ({bert_prob:.1%})")
            except Exception as e:
                print(f"   BERT:           [ERROR] {e}")
        
        # Logistic Regression prediction
        if lr_model is not None and vectorizer is not None:
            try:
                lr_prob = predict_ml(vectorizer, lr_model, text)
                results['Logistic Regression'] = lr_prob
                label = "Depressed" if lr_prob >= 0.5 else "Not depressed"
                print(f"   Logistic Reg:   {label} ({lr_prob:.1%})")
            except Exception as e:
                print(f"   Logistic Reg:   [ERROR] {e}")
        
        # SVM prediction
        if svm_model is not None and vectorizer is not None:
            try:
                svm_prob = predict_ml(vectorizer, svm_model, text)
                results['SVM'] = svm_prob
                label = "Depressed" if svm_prob >= 0.5 else "Not depressed"
                print(f"   SVM:            {label} ({svm_prob:.1%})")
            except Exception as e:
                print(f"   SVM:            [ERROR] {e}")
        
        # Combined prediction
        if results:
            avg_prob = np.mean(list(results.values()))
            label = "Depressed" if avg_prob >= 0.5 else "Not depressed"
            print(f"\n   Combined:       {label} ({avg_prob:.1%})")
            print(f"   Models used:     {', '.join(results.keys())}")
    
    print("\n" + "="*70)
    print("[OK] Testing completed!")
    print("="*70)


if __name__ == '__main__':
    test_all_models()
