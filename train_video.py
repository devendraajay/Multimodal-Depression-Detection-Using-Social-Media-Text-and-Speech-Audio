"""
Video pipeline for depression detection (same as nlp(1).ipynb on extracted audio).
1. Load video paths by class (normal / depressed)
2. Extract audio from each video (ffmpeg or moviepy)
3. Transcribe with Whisper (16 kHz)
4. Clean text, train all models (LSTM, Logistic Regression, SVM, BERT); save to models/
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from config import AUDIO_VIDEO_DATASET_PATH
except ImportError:
    AUDIO_VIDEO_DATASET_PATH = "Multimodel_Dataset"

from data_loader_video import load_video_paths_and_labels
from video_utils import extract_audio
from model_bert import BERTDepressionModel, BERTDataset, SimpleTokenizer


def clean_text(text: str) -> str:
    """Same as nlp(1).ipynb."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_whisper_on_videos(video_paths: list, labels: list, save_csv: str, device: str = "cuda"):
    """Extract audio from each video, run Whisper, save CSV (filename, text, label)."""
    import librosa
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
    model.config.forced_decoder_ids = None

    records = []
    for path, label in tqdm(list(zip(video_paths, labels)), desc="Video->Audio->Whisper"):
        wav_path = None
        try:
            wav_path = extract_audio(path, sample_rate=16000)
            if wav_path is None:
                print(f"Could not extract audio: {path}")
                continue
            audio, _ = librosa.load(wav_path, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            records.append({"filename": os.path.basename(path), "text": text.strip(), "label": label})
        except Exception as e:
            print(f"Skip {path}: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    if not records:
        df = pd.DataFrame(columns=["filename", "text", "label"])
    else:
        df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved {len(records)} transcripts to {save_csv}")
    return df


def train_video_models(transcripts_csv: str, output_dir: str):
    """Same as train_audio_models but save with video_ prefix."""
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    if not os.path.isfile(transcripts_csv) or os.path.getsize(transcripts_csv) == 0:
        print("No transcripts CSV found or file is empty. Skip video model training.")
        return
    try:
        df = pd.read_csv(transcripts_csv)
    except pd.errors.EmptyDataError:
        print("Transcripts CSV is empty. No video transcripts were generated.")
        print("Install ffmpeg (https://ffmpeg.org/download.html) and add to PATH for best video audio extraction.")
        return
    if df.empty or len(df) < 10:
        print("Too few video transcripts to train (need at least 10). Skip video model training.")
        print("If all extractions failed, install ffmpeg and add it to PATH.")
        return
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    texts = df["clean_text"].tolist()
    labels = np.array(df["label"].values)

    mask = [bool(t.strip()) for t in texts]
    texts = [t for t, m in zip(texts, mask) if m]
    labels = labels[mask]
    if len(texts) < 10:
        print("Too few samples after cleaning.")
        return
    
    # Check if we have at least 2 classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"Only {len(unique_labels)} class(es) found in video transcripts (need at least 2 for classification).")
        print(f"Classes present: {unique_labels.tolist()}")
        print("Skipping video model training.")
        return

    # Try stratified split, fall back to regular split if stratification fails
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
    except ValueError:
        # Stratification failed (likely only one class per split), use regular split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
    
    # Final check: ensure both train and test sets have at least 2 classes
    if len(np.unique(y_train)) < 2:
        print(f"Training set has only {len(np.unique(y_train))} class(es). Cannot train classification models.")
        print("Skipping video model training.")
        return
    os.makedirs(output_dir, exist_ok=True)

    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        MAX_WORDS = 1000
        MAXLEN = 64
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        X_train_seq = pad_sequences(
            tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN, padding="post", truncating="post"
        )
        X_test_seq = pad_sequences(
            tokenizer.texts_to_sequences(X_test), maxlen=MAXLEN, padding="post", truncating="post"
        )
        vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
        model = Sequential([
            Embedding(vocab_size, 64),
            LSTM(64),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(
            X_train_seq, y_train,
            validation_split=0.1,
            epochs=20,
            batch_size=16,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            verbose=1,
        )
        # Check if test set has at least 2 classes for evaluation
        if len(np.unique(y_test)) < 2:
            print("Video LSTM: Test set has only one class. Cannot evaluate properly.")
            print("Saving model anyway, but evaluation metrics may be unreliable.")
        preds = (model.predict(X_test_seq, verbose=0).squeeze() > 0.5).astype(int)
        print("Video LSTM test accuracy:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds, zero_division=0))
        model.save(os.path.join(output_dir, "video_lstm.keras"))
        with open(os.path.join(output_dir, "video_tokenizer.json"), "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json())
        with open(os.path.join(output_dir, "video_lstm_config.json"), "w", encoding="utf-8") as f:
            json.dump({"maxlen": MAXLEN, "num_words": MAX_WORDS}, f)
    except ImportError as e:
        print("Keras not available, skip LSTM:", e)

    # Check again before fitting SVM (in case something went wrong)
    if len(np.unique(y_train)) < 2:
        print("SVM: Training set has only one class. Skipping SVM training.")
        print(f"Video models (LSTM only) saved to {output_dir}")
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X_train_tf = vectorizer.fit_transform(X_train)
        X_test_tf = vectorizer.transform(X_test)
        scaler = StandardScaler()
        X_train_tf = scaler.fit_transform(X_train_tf.toarray())
        X_test_tf = scaler.transform(X_test_tf.toarray())
        
        svm = SVC(kernel="linear", probability=True)
        svm.fit(X_train_tf, y_train)
        preds = svm.predict(X_test_tf)
        print("Video SVM test accuracy:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds, zero_division=0))
        if len(np.unique(y_test)) >= 2:
            print(classification_report(y_test, preds, target_names=["normal", "depressed"]))
        else:
            print("SVM: Test set has only one class. Classification report skipped.")

        import pickle
        with open(os.path.join(output_dir, "video_tfidf.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
        with open(os.path.join(output_dir, "video_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(output_dir, "video_svm.pkl"), "wb") as f:
            pickle.dump(svm, f)
        
        # Train Logistic Regression
        from sklearn.linear_model import LogisticRegression
        if len(np.unique(y_train)) >= 2:
            lr = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            lr.fit(X_train_tf, y_train)
            lr_preds = lr.predict(X_test_tf)
            print("Video Logistic Regression test accuracy:", accuracy_score(y_test, lr_preds), "F1:", f1_score(y_test, lr_preds, zero_division=0))
            with open(os.path.join(output_dir, "video_lr.pkl"), "wb") as f:
                pickle.dump(lr, f)
        else:
            print("Video LR: Training set has only one class. Skipping LR training.")
        
        print(f"Video models (LSTM, LR, SVM) saved to {output_dir}")
    
    # Train BERT (separate from TF-IDF models)
    try:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        
        if len(np.unique(y_train)) >= 2:
            print("\nTraining Video BERT model...")
            # Create tokenizer
            tokenizer = SimpleTokenizer()
            tokenizer.fit(X_train, max_vocab_size=30000, min_freq=2)
            
            # Create datasets
            train_dataset = BERTDataset(X_train, y_train, tokenizer, max_length=128)
            test_dataset = BERTDataset(X_test, y_test, tokenizer, max_length=128)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Create model
            model = BERTDepressionModel(
                vocab_size=tokenizer.vocab_size,
                max_length=128
            ).to(device)
            
            # Calculate class weights
            pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
            
            # Train
            criterion = nn.BCELoss(weight=torch.tensor([pos_weight], device=device))
            optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            
            model.train()
            epochs = 3
            for epoch in range(epochs):
                total_loss = 0
                for batch in tqdm(train_loader, desc=f"Video BERT Epoch {epoch+1}/{epochs}", leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device).float()
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
            
            # Evaluate
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids, attention_mask)
                    preds = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
                    all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds])
                    all_labels.extend(labels.cpu().numpy().tolist())
            
            bert_acc = accuracy_score(all_labels, all_preds)
            bert_f1 = f1_score(all_labels, all_preds, zero_division=0)
            print(f"Video BERT - Accuracy: {bert_acc:.4f}, F1: {bert_f1:.4f}")
            
            # Save BERT model
            import tempfile
            import shutil
            bert_path = os.path.abspath(os.path.join(output_dir, "video_bert.pt"))
            bert_tokenizer_path = os.path.abspath(os.path.join(output_dir, "video_bert_tokenizer.json"))
            os.makedirs(os.path.dirname(bert_path) or ".", exist_ok=True)
            checkpoint = {
                'state_dict': model.state_dict(),
                'model_config': model.model_config,
                'vocab_size': tokenizer.vocab_size,
                'max_length': 128
            }
            try:
                save_dir = os.path.dirname(bert_path)
                if not save_dir:
                    save_dir = "."
                fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=save_dir)
                try:
                    os.close(fd)
                    torch.save(checkpoint, tmp_path)
                    if os.path.exists(bert_path):
                        os.remove(bert_path)
                    shutil.move(tmp_path, bert_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    raise
            except Exception:
                if os.path.exists(bert_path):
                    os.remove(bert_path)
                torch.save(checkpoint, bert_path)
            tokenizer.save(bert_tokenizer_path)
            print(f"Video BERT model saved to {output_dir}")
        else:
            print("Video BERT: Training set has only one class. Skipping BERT training.")
    except Exception as e:
        print(f"Video BERT training failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Train video pipeline (extract audio -> Whisper -> LSTM/LR/SVM/BERT)")
    parser.add_argument("--dataset_path", type=str, default=AUDIO_VIDEO_DATASET_PATH)
    parser.add_argument("--video_subdir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--transcripts_only", action="store_true")
    parser.add_argument("--from_csv", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.from_csv:
        train_video_models(args.from_csv, output_dir)
        return

    dataset_path = args.dataset_path
    if args.video_subdir:
        dataset_path = os.path.join(dataset_path, args.video_subdir)
    paths, labels = load_video_paths_and_labels(dataset_path)
    if not paths:
        print("No video files found. Check path and folder layout (e.g. <path>/normal/*.mp4, <path>/depressed/*.mp4)")
        return

    transcripts_csv = os.path.join(output_dir, "video_transcripts.csv")
    df = run_whisper_on_videos(paths, labels, transcripts_csv, device=args.device)

    if len(df) == 0:
        print("No video transcripts were generated (audio extraction failed for all files).")
        print("Install ffmpeg for best compatibility: https://ffmpeg.org/download.html (add to PATH).")
        print("Skipping video model training.")
        return

    if not args.transcripts_only:
        train_video_models(transcripts_csv, output_dir)


if __name__ == "__main__":
    main()
