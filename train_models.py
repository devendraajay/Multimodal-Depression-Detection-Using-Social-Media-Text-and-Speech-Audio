"""
Training Script for All Models
Trains and compares: Logistic Regression (custom), SVM, BERT, and LSTM (text).
Uses cross-validation for evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import json
from tqdm import tqdm

from data_loader import DataLoader as DataLoaderClass
from feature_extraction import FeatureExtractor
from models_ml import LogisticRegressionScratch, SVMScratch, probability_uncertainty
from model_bert import BERTDepressionModel, BERTDataset, SimpleTokenizer

# Keras/TensorFlow for text LSTM (same as nlp(1).ipynb)
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


def train_bert_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5, pos_weight=None):
    """Train BERT model with improved training to prevent constant predictions"""
    # Use weighted loss if class imbalance exists
    if pos_weight is not None:
        criterion = nn.BCELoss(weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 3
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Track predictions to monitor for constant predictions
            with torch.no_grad():
                preds = outputs.squeeze().cpu().numpy()
                train_preds.extend(preds.tolist() if preds.ndim > 0 else [preds])
                train_labels.extend(labels.cpu().numpy().tolist())
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                preds = outputs.squeeze().cpu().numpy()
                val_preds.extend(preds.tolist() if preds.ndim > 0 else [preds])
                val_labels.extend(labels.cpu().numpy().tolist())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Check for constant predictions
        train_pred_mean = np.mean(train_preds)
        train_pred_std = np.std(train_preds)
        val_pred_mean = np.mean(val_preds)
        val_pred_std = np.std(val_preds)
        
        # Calculate accuracy
        train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        
        print(f"  Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"    Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"    Train Pred Mean: {train_pred_mean:.4f} ± {train_pred_std:.4f}, Val Pred Mean: {val_pred_mean:.4f} ± {val_pred_std:.4f}")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"    Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Early stopping if validation loss doesn't improve
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Warn if predictions are too constant
        if train_pred_std < 0.01:
            print(f"  WARNING: Very low prediction variance ({train_pred_std:.6f}). Model may be predicting constant values.")
    
    return model


def evaluate_bert_model(model, test_loader, device):
    """Evaluate BERT model; returns dict with accuracy, precision, recall, f1_score."""
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            predictions = (outputs.squeeze() > 0.5).float()
            pred_np = predictions.cpu().numpy()
            label_np = labels.cpu().numpy()
            if pred_np.ndim == 0:
                all_predictions.append(float(pred_np))
            else:
                all_predictions.extend(pred_np.tolist())
            if label_np.ndim == 0:
                all_labels.append(float(label_np))
            else:
                all_labels.extend(label_np.tolist())
    return {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, zero_division=0),
    }


def bert_probabilities(model, test_loader, device):
    """Return raw class-1 probabilities for uncertainty reporting."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask).squeeze().cpu().numpy()
            if np.ndim(outputs) == 0:
                probs.append(float(outputs))
            else:
                probs.extend(outputs.tolist())
    return np.array(probs, dtype=float)


def run_cv_ml(X, y, model_factory, model_name, cv_folds=5, random_state=42):
    """
    Run stratified k-fold CV for a classical ML model.
    model_factory() returns a fresh model with .train(X_train, y_train) and .evaluate(X_val, y_val).
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y = np.array(y)
    metrics_list = []
    
    print(f"  Running {cv_folds}-fold CV for {model_name}...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create fresh model for this fold
        model = model_factory()
        
        # Train model - handle different training signatures
        try:
            # Try with verbose parameter
            model.train(X_train_fold, y_train_fold, verbose=False)
        except TypeError:
            try:
                # Try without verbose parameter
                model.train(X_train_fold, y_train_fold)
            except Exception as e:
                print(f"    Fold {fold+1} training error: {e}")
                continue
        
        # Evaluate on validation set
        try:
            m = model.evaluate(X_val_fold, y_val_fold)
            metrics_list.append(m)
            print(f"    Fold {fold+1}/{cv_folds}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}")
        except Exception as e:
            print(f"    Fold {fold+1} evaluation error: {e}")
            continue
    
    if len(metrics_list) == 0:
        print(f"  WARNING: No successful CV folds for {model_name}")
        return {
            'accuracy': (0.0, 0.0),
            'precision': (0.0, 0.0),
            'recall': (0.0, 0.0),
            'f1_score': (0.0, 0.0),
        }
    
    acc = [m['accuracy'] for m in metrics_list]
    prec = [m['precision'] for m in metrics_list]
    rec = [m['recall'] for m in metrics_list]
    f1 = [m['f1_score'] for m in metrics_list]
    
    print(f"  CV Results: Acc={np.mean(acc):.4f}±{np.std(acc):.4f}, F1={np.mean(f1):.4f}±{np.std(f1):.4f}")
    
    return {
        'accuracy': (np.mean(acc), np.std(acc)),
        'precision': (np.mean(prec), np.std(prec)),
        'recall': (np.mean(rec), np.std(rec)),
        'f1_score': (np.mean(f1), np.std(f1)),
    }


def run_cv_bert(texts, labels, max_length, device, cv_folds=5, epochs=2, batch_size=16, random_state=42, pos_weight=None):
    """Run stratified k-fold CV for BERT."""
    texts, labels = np.array(texts), np.array(labels)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
        train_labels, val_labels = labels[train_idx].tolist(), labels[val_idx].tolist()
        fold_tokenizer = SimpleTokenizer().fit(train_texts)
        train_ds = BERTDataset(train_texts, train_labels, fold_tokenizer, max_length=max_length)
        val_ds = BERTDataset(val_texts, val_labels, fold_tokenizer, max_length=max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        model = BERTDepressionModel(vocab_size=fold_tokenizer.vocab_size, max_length=max_length).to(device)
        train_bert_model(model, train_loader, val_loader, device, epochs=epochs, pos_weight=pos_weight)
        m = evaluate_bert_model(model, val_loader, device)
        metrics_list.append(m)
    acc = [m['accuracy'] for m in metrics_list]
    prec = [m['precision'] for m in metrics_list]
    rec = [m['recall'] for m in metrics_list]
    f1 = [m['f1_score'] for m in metrics_list]
    return {
        'accuracy': (np.mean(acc), np.std(acc)),
        'precision': (np.mean(prec), np.std(prec)),
        'recall': (np.mean(rec), np.std(rec)),
        'f1_score': (np.mean(f1), np.std(f1)),
    }


def run_cv_lstm(texts, labels, max_words=5000, maxlen=128, cv_folds=5, epochs=10, batch_size=32, random_state=42):
    """
    Run stratified k-fold CV for the text LSTM.
    Mirrors the final LSTM architecture in STEP 5.
    """
    if not HAS_KERAS:
        print("Skipping LSTM CV: tensorflow/keras not available.")
        return {
            'accuracy': (0.0, 0.0),
            'precision': (0.0, 0.0),
            'recall': (0.0, 0.0),
            'f1_score': (0.0, 0.0),
        }

    texts, labels = np.array(texts), np.array(labels)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n--- LSTM CV Fold {fold+1}/{cv_folds} ---")
        train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
        train_labels, val_labels = labels[train_idx].tolist(), labels[val_idx].tolist()

        # Tokenizer and sequences
        lstm_tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        lstm_tokenizer.fit_on_texts(train_texts)
        X_train_seq = pad_sequences(
            lstm_tokenizer.texts_to_sequences(train_texts),
            maxlen=maxlen, padding="post", truncating="post"
        )
        X_val_seq = pad_sequences(
            lstm_tokenizer.texts_to_sequences(val_texts),
            maxlen=maxlen, padding="post", truncating="post"
        )

        vocab_size = min(max_words, len(lstm_tokenizer.word_index) + 1)
        lstm_model = Sequential([
            Embedding(vocab_size, 64),
            LSTM(64),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ])
        lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        lstm_model.fit(
            X_train_seq, np.array(train_labels),
            validation_data=(X_val_seq, np.array(val_labels)),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        # Evaluate on validation fold
        val_probs = lstm_model.predict(X_val_seq, verbose=0).squeeze()
        val_preds = (val_probs > 0.5).astype(int)

        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"    Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}")
        metrics_list.append({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
        })

    if not metrics_list:
        print("WARNING: No successful CV folds for LSTM.")
        return {
            'accuracy': (0.0, 0.0),
            'precision': (0.0, 0.0),
            'recall': (0.0, 0.0),
            'f1_score': (0.0, 0.0),
        }

    acc_vals = [m['accuracy'] for m in metrics_list]
    prec_vals = [m['precision'] for m in metrics_list]
    rec_vals = [m['recall'] for m in metrics_list]
    f1_vals = [m['f1_score'] for m in metrics_list]
    return {
        'accuracy': (np.mean(acc_vals), np.std(acc_vals)),
        'precision': (np.mean(prec_vals), np.std(prec_vals)),
        'recall': (np.mean(rec_vals), np.std(rec_vals)),
        'f1_score': (np.mean(f1_vals), np.std(f1_vals)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train and Compare Multiple Models (with CV)')
    parser.add_argument('--dataset_path', type=str, default='Dataset_MDDL (1)/Dataset', help='Path to dataset directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples per class (None for all)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_features', type=int, default=5000, help='Maximum TF-IDF features')
    parser.add_argument('--ngram_max', type=int, default=2, help='Max n-gram for TF-IDF (1=unigrams only, 2=uni+bigrams)')
    parser.add_argument('--min_df', type=int, default=2, help='Min documents for TF-IDF term (reduces noise)')
    parser.add_argument('--max_df', type=float, default=0.95, help='Max document frequency for TF-IDF (e.g. 0.95)')
    parser.add_argument('--no_sublinear_tf', action='store_true', help='Disable sublinear TF (use raw counts)')
    parser.add_argument('--bert_epochs', type=int, default=5, help='Number of epochs for BERT training')
    parser.add_argument('--bert_batch_size', type=int, default=16, help='Batch size for BERT')
    parser.add_argument('--bert_max_length', type=int, default=128, help='Maximum sequence length for BERT')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models and results')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu', 'auto'),
                        help='Device for BERT: cuda (GPU), cpu, or auto (GPU if available)')
    args = parser.parse_args()
    args.dataset_path = args.dataset_path.strip() if args.dataset_path else args.dataset_path
    args.output_dir = os.path.normpath(os.path.abspath(args.output_dir.strip() if args.output_dir else 'models'))

    os.makedirs(args.output_dir, exist_ok=True)
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: GPU (cuda) - {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using device: CPU (CUDA not available; install CUDA + PyTorch GPU build to use GPU)")
    elif args.device == 'cpu':
        device = torch.device('cpu')
        print("Using device: CPU")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"Using device: GPU (auto) - {torch.cuda.get_device_name(0)}")
        else:
            print("Using device: CPU (auto)")

    # Load data
    print("="*60)
    print("STEP 1: DATA LOADING")
    print("="*60)
    data_loader = DataLoaderClass(args.dataset_path)
    texts, labels = data_loader.load_all_data(max_samples=args.max_samples)
    print(f"Total samples: {len(texts)}")
    
    # Calculate class weights for balancing
    labels_array = np.array(labels)
    pos_count = np.sum(labels_array == 1)
    neg_count = np.sum(labels_array == 0)
    total = len(labels_array)
    
    if pos_count > 0 and neg_count > 0:
        # Calculate class weights: weight for positive class (depressed)
        pos_weight_value = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"\nClass distribution:")
        print(f"  Positive (depressed): {pos_count} ({pos_count/total:.2%})")
        print(f"  Negative (non-depressed): {neg_count} ({neg_count/total:.2%})")
        print(f"  Class weight for positive class: {pos_weight_value:.4f}")
    else:
        pos_weight_value = 1.0
        print("Warning: Cannot calculate class weights - using default")

    if len(texts) == 0:
        print("\nError: No data loaded. Please check:")
        print(f"  1. Dataset path: {os.path.abspath(args.dataset_path)}")
        print("  2. Expected folder structure:")
        print("       <dataset_path>/labeled/positive/data/tweet/*.json")
        print("       <dataset_path>/labeled/negative/data/tweet/*.json")
        print("  3. Remove any leading/trailing spaces from the path (e.g. use \"Dataset_MDDL (1)/Dataset\" not \" Dataset_MDDL (1)/Dataset\")")
        return

    # Train-test split (stratified)
    print("\n" + "="*60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Feature extraction (TF-IDF) on full training set
    print("\n" + "="*60)
    print("STEP 3: FEATURE EXTRACTION (TF-IDF)")
    print("="*60)
    ngram_range = (1, max(1, args.ngram_max))
    feature_extractor = FeatureExtractor(
        max_features=args.max_features,
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=not args.no_sublinear_tf,
    )
    print(f"  ngram_range={ngram_range}, min_df={args.min_df}, max_df={args.max_df}, sublinear_tf={not args.no_sublinear_tf}")
    X_train_tfidf = feature_extractor.fit_transform(X_train)
    X_test_tfidf = feature_extractor.transform(X_test)
    feature_extractor.save(os.path.join(args.output_dir, 'tfidf_vectorizer.pkl'))

    # Cross-validation evaluation (on training set only)
    print("\n" + "="*60)
    print(f"STEP 4: CROSS-VALIDATION ({args.cv_folds}-fold) ON TRAINING SET")
    print("="*60)

    cv_results = {}

    # CV: Logistic Regression (custom) - with class weights
    print("\n--- Logistic Regression CV ---")
    cv_results['Logistic_Regression'] = run_cv_ml(
        X_train_tfidf, list(y_train),
        model_factory=lambda: LogisticRegressionScratch(learning_rate=0.01, max_iter=1500, random_state=args.random_state, class_weight=pos_weight_value),
        model_name='LR',
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )
    # Custom LR uses .train(X, y) with y as list; evaluate expects array.

    # CV: SVM - with class weights
    print("\n--- SVM CV ---")
    cv_results['SVM'] = run_cv_ml(
        X_train_tfidf, list(y_train),
        model_factory=lambda: SVMScratch(learning_rate=0.01, lambda_param=0.01, max_iter=1500, random_state=args.random_state, class_weight=pos_weight_value),
        model_name='SVM',
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )

    # CV: BERT - with class weights
    print("\n--- BERT CV ---")
    cv_results['BERT'] = run_cv_bert(
        X_train, y_train, args.bert_max_length, device,
        cv_folds=args.cv_folds,
        epochs=args.bert_epochs,
        batch_size=args.bert_batch_size,
        random_state=args.random_state,
        pos_weight=pos_weight_value
    )

    # CV: LSTM (text)
    if HAS_KERAS:
        print("\n--- LSTM CV ---")
        cv_results['LSTM'] = run_cv_lstm(
            X_train, y_train,
            max_words=5000,
            maxlen=128,
            cv_folds=args.cv_folds,
            epochs=10,
            batch_size=32,
            random_state=args.random_state,
        )

    # Print CV summary
    print("\n" + "="*60)
    print("CV RESULTS (Mean ± Std)")
    print("="*60)
    print(f"{'Model':<35} {'Accuracy':<18} {'Precision':<18} {'Recall':<18} {'F1-Score':<18}")
    print("-"*60)
    for model_name, m in cv_results.items():
        acc = m['accuracy']
        prec = m['precision']
        rec = m['recall']
        f1 = m['f1_score']
        print(f"{model_name:<35} {acc[0]:.4f}±{acc[1]:.4f}    {prec[0]:.4f}±{prec[1]:.4f}    {rec[0]:.4f}±{rec[1]:.4f}    {f1[0]:.4f}±{f1[1]:.4f}")

    # Train final models on full training set and evaluate on test set
    print("\n" + "="*60)
    print("STEP 5: TRAIN FINAL MODELS (full training set) & TEST SET EVALUATION")
    print("="*60)
    results = {}

    # Final: Logistic Regression (custom) - with class weights
    print("\n--- Logistic Regression (Custom) ---")
    print(f"Class weight for positive class: {pos_weight_value:.4f}")
    
    lr_model = LogisticRegressionScratch(learning_rate=0.01, max_iter=1500, random_state=args.random_state, class_weight=pos_weight_value)
    print("Training Logistic Regression (Custom)...")
    lr_model.train(X_train_tfidf, np.array(y_train), verbose=True)
    
    print("Evaluating on test set...")
    results['Logistic_Regression'] = lr_model.evaluate(X_test_tfidf, np.array(y_test))
    print(f"Test Results - Acc: {results['Logistic_Regression']['accuracy']:.4f}, "
          f"F1: {results['Logistic_Regression']['f1_score']:.4f}")
    
    lr_custom_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]
    results['Logistic_Regression']['uncertainty_mean'] = float(
        np.mean([probability_uncertainty(p) for p in lr_custom_probs])
    )
    lr_model.save(os.path.abspath(os.path.join(args.output_dir, 'lr_model.pkl')))
    print(f"Model saved to {os.path.join(args.output_dir, 'lr_model.pkl')}")

    # Final: SVM - with class weights
    print("\n--- SVM ---")
    print(f"Class weight for positive class: {pos_weight_value:.4f}")
    
    svm_model = SVMScratch(learning_rate=0.01, lambda_param=0.01, max_iter=1500, random_state=args.random_state, class_weight=pos_weight_value)
    print("Training SVM...")
    svm_model.train(X_train_tfidf, np.array(y_train), verbose=True)
    
    print("Evaluating on test set...")
    results['SVM'] = svm_model.evaluate(X_test_tfidf, np.array(y_test))
    print(f"Test Results - Acc: {results['SVM']['accuracy']:.4f}, "
          f"F1: {results['SVM']['f1_score']:.4f}")
    
    svm_probs = svm_model.predict_proba(X_test_tfidf)[:, 1]
    results['SVM']['uncertainty_mean'] = float(
        np.mean([probability_uncertainty(p) for p in svm_probs])
    )
    svm_model.save(os.path.abspath(os.path.join(args.output_dir, 'svm_model.pkl')))
    print(f"Model saved to {os.path.join(args.output_dir, 'svm_model.pkl')}")

    # Final: BERT
    print("\n--- BERT ---")
    tokenizer = SimpleTokenizer().fit(X_train)
    tokenizer.save(os.path.join(args.output_dir, 'bert_tokenizer.json'))
    
    # CRITICAL: Use tokenizer vocab_size for model initialization
    # This ensures exact match between tokenizer and model
    tokenizer_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {tokenizer_vocab_size}")

    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_length=args.bert_max_length)
    test_dataset = BERTDataset(X_test, y_test, tokenizer, max_length=args.bert_max_length)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=args.bert_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bert_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bert_batch_size, shuffle=False)
    
    # Create model with EXACT tokenizer vocab_size
    bert_model = BERTDepressionModel(
        vocab_size=tokenizer_vocab_size,  # Use exact tokenizer vocab_size
        max_length=args.bert_max_length
    ).to(device)
    
    print(f"BERT model initialized with vocab_size: {bert_model.model_config['vocab_size']}")
    bert_model = train_bert_model(bert_model, train_loader, val_loader, device, epochs=args.bert_epochs, pos_weight=pos_weight_value)
    results['BERT'] = evaluate_bert_model(bert_model, test_loader, device)
    bert_probs = bert_probabilities(bert_model, test_loader, device)
    results['BERT']['uncertainty_mean'] = float(np.mean([probability_uncertainty(p) for p in bert_probs]))
    # Save BERT model with all necessary configuration
    # CRITICAL: Save vocab_size explicitly at top level for easy access
    model_vocab_size = bert_model.model_config['vocab_size']
    save_dict = {
        'state_dict': bert_model.state_dict(),
        'model_config': bert_model.get_config_dict(),
        'max_length': args.bert_max_length,
        'vocab_size': model_vocab_size,  # Save at top level for easy access
        'tokenizer_vocab_size': tokenizer.vocab_size  # Also save tokenizer vocab for reference
    }
    
    # Verify vocab_size matches
    if model_vocab_size != tokenizer.vocab_size:
        print(f"WARNING: Model vocab_size ({model_vocab_size}) != Tokenizer vocab_size ({tokenizer.vocab_size})")
        print(f"Using model vocab_size ({model_vocab_size}) for saving")
    
    torch.save(save_dict, os.path.join(args.output_dir, 'bert_model.pt'))
    print(f"BERT model saved with config: vocab_size={model_vocab_size}, max_length={args.bert_max_length}")

    # --- Text LSTM (same as nlp(1).ipynb: Tokenizer -> pad_sequences -> Embedding -> LSTM -> Dense(sigmoid)) ---
    if HAS_KERAS:
        print("\n--- Text LSTM ---")
        LSTM_MAX_WORDS = 5000
        LSTM_MAXLEN = 128
        lstm_tokenizer = Tokenizer(num_words=LSTM_MAX_WORDS, oov_token="<OOV>")
        lstm_tokenizer.fit_on_texts(X_train)
        X_train_seq = pad_sequences(
            lstm_tokenizer.texts_to_sequences(X_train),
            maxlen=LSTM_MAXLEN, padding="post", truncating="post"
        )
        X_test_seq = pad_sequences(
            lstm_tokenizer.texts_to_sequences(X_test),
            maxlen=LSTM_MAXLEN, padding="post", truncating="post"
        )
        vocab_size = min(LSTM_MAX_WORDS, len(lstm_tokenizer.word_index) + 1)
        lstm_model = Sequential([
            Embedding(vocab_size, 64),
            LSTM(64),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ])
        lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        lstm_model.fit(
            X_train_seq, np.array(y_train),
            validation_split=0.1,
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1,
        )
        lstm_probs = lstm_model.predict(X_test_seq, verbose=0)
        lstm_probs = np.squeeze(lstm_probs)
        lstm_preds = (lstm_probs > 0.5).astype(int)
        results["LSTM"] = {
            "accuracy": accuracy_score(y_test, lstm_preds),
            "precision": precision_score(y_test, lstm_preds, zero_division=0),
            "recall": recall_score(y_test, lstm_preds, zero_division=0),
            "f1_score": f1_score(y_test, lstm_preds, zero_division=0),
            "uncertainty_mean": float(np.mean([probability_uncertainty(p) for p in lstm_probs])),
        }
        print(f"Test Results (LSTM) - Acc: {results['LSTM']['accuracy']:.4f}, F1: {results['LSTM']['f1_score']:.4f}")
        lstm_model.save(os.path.join(args.output_dir, "text_lstm.keras"))
        with open(os.path.join(args.output_dir, "text_tokenizer.json"), "w", encoding="utf-8") as f:
            f.write(lstm_tokenizer.to_json())
        with open(os.path.join(args.output_dir, "text_lstm_config.json"), "w", encoding="utf-8") as f:
            json.dump({"maxlen": LSTM_MAXLEN, "num_words": LSTM_MAX_WORDS}, f)
        print(f"LSTM model and tokenizer saved to {args.output_dir}")
    else:
        print("\nSkipping Text LSTM (tensorflow not available). Install with: pip install tensorflow")

    # Test set comparison
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<35} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")

    # Save results (CV and test)
    out = {
        'cv_results': {k: {mk: [float(mv[0]), float(mv[1])] for mk, mv in v.items()} for k, v in cv_results.items()},
        'test_results': results,
    }
    results_path = os.path.join(args.output_dir, 'model_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"\nResults saved to {results_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
