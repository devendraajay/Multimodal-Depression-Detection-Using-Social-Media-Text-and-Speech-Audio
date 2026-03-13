"""
Unified Multi-Modal Training Script
Trains text models (LSTM, Logistic Regression, SVM, BERT) and transcribes audio/video.

Pipeline:
- Text: Train models directly on tweet data
- Audio: Transcribe with Whisper → Use text models (same pipeline as text)
- Video: Extract audio → Transcribe with Whisper → Use text models (same pipeline as text)

Audio and video transcripts use the SAME text models - no separate models needed.
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

try:
    from config import AUDIO_VIDEO_DATASET_PATH, DATASET_PATH
except ImportError:
    AUDIO_VIDEO_DATASET_PATH = "Multimodel_Dataset"
    DATASET_PATH = "Dataset_MDDL (1)/Dataset"

from data_loader_audio import load_audio_paths_and_labels
from data_loader_video import load_video_paths_and_labels
from data_loader import DataLoader as TextDataLoader
from models_ml import LogisticRegressionScratch, SVMScratch
from model_bert import BERTDepressionModel, BERTDataset, SimpleTokenizer

# Keras/TensorFlow for LSTM
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("Warning: Keras not available. LSTM models will be skipped.")


def clean_text(text: str) -> str:
    """Clean text: lower, remove non-alpha, normalize spaces."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def train_lstm_model(X_train, y_train, X_test, y_test, output_path, tokenizer_path, config_path, prefix=""):
    """Train LSTM model and save."""
    if not HAS_KERAS:
        print(f"Skipping {prefix}LSTM: Keras not available")
        return None
    
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
    
    preds = (model.predict(X_test_seq, verbose=0).squeeze() > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{prefix}LSTM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    model.save(output_path)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"maxlen": MAXLEN, "num_words": MAX_WORDS}, f)
    
    return model


def train_lr_model(X_train, y_train, X_test, y_test, output_path, prefix=""):
    """Train Logistic Regression model and save."""
    from sklearn.linear_model import LogisticRegression
    
    # Check if we have at least 2 classes
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        print(f"Skipping {prefix}Logistic Regression: Only one class in training data ({unique_labels[0]})")
        return None
    
    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{prefix}Logistic Regression - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    
    return model


def train_svm_model(X_train, y_train, X_test, y_test, output_path, prefix=""):
    """Train SVM model and save."""
    # Check if we have at least 2 classes
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        print(f"Skipping {prefix}SVM: Only one class in training data ({unique_labels[0]})")
        return None
    
    model = SVC(kernel="linear", probability=True, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{prefix}SVM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    
    return model


def train_bert_model(X_train, y_train, X_test, y_test, output_path, tokenizer_path, device, prefix="", epochs=5, label_smoothing=0.1):
    """
    Train BERT model and save.
    label_smoothing: softens 0/1 labels to prevent overconfident (0 or 100%) predictions.
    """
    # Check if we have at least 2 classes
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        print(f"Skipping {prefix}BERT: Only one class in training data ({unique_labels[0]})")
        return None
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(X_train, max_vocab_size=30000, min_freq=2)
    
    # Create datasets
    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_length=128)
    test_dataset = BERTDataset(X_test, y_test, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create model with higher dropout (0.4) to reduce overconfidence
    model = BERTDepressionModel(
        vocab_size=tokenizer.vocab_size,
        max_length=128,
        dropout=0.4
    ).to(device)
    
    # BCE loss - no pos_weight to avoid extreme outputs
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    
    best_val_loss = float("inf")
    best_state = None
    patience = 2
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{prefix}BERT Epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()
            # Label smoothing: 0 -> 0.1, 1 -> 0.9 to prevent extreme predictions
            labels_smooth = labels * (1.0 - label_smoothing) + (label_smoothing / 2)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation loss for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).float()
                labels_smooth = labels * (1.0 - label_smoothing) + (label_smoothing / 2)
                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs.squeeze(), labels_smooth).item()
        val_loss /= len(test_loader)
        model.train()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{prefix}BERT early stop at epoch {epoch+1}")
                break
    
    # Use best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    
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
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"{prefix}BERT - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Save: use temp file then rename to avoid Windows "cannot be opened" errors
    import tempfile
    import shutil
    output_path_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path_abs) or ".", exist_ok=True)
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_config': model.model_config,
        'vocab_size': tokenizer.vocab_size,
        'max_length': 128
    }
    try:
        save_dir = os.path.dirname(output_path_abs)
        if not save_dir:
            save_dir = "."
        fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=save_dir)
        try:
            os.close(fd)
            torch.save(checkpoint, tmp_path)
            if os.path.exists(output_path_abs):
                os.remove(output_path_abs)
            shutil.move(tmp_path, output_path_abs)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    except Exception:
        # Fallback: direct save
        if os.path.exists(output_path_abs):
            os.remove(output_path_abs)
        torch.save(checkpoint, output_path_abs)
    tokenizer.save(os.path.abspath(tokenizer_path))
    
    return model


def train_modality_models(texts, labels, output_dir, prefix="", device="cuda"):
    """Train all models (LSTM, LR, SVM, BERT) for a given modality."""
    print(f"\n{'='*60}")
    print(f"Training {prefix}models")
    print(f"{'='*60}")
    
    # Clean texts
    texts_clean = [clean_text(t) for t in texts]
    
    # Filter empty
    mask = [bool(t.strip()) for t in texts_clean]
    texts_clean = [t for t, m in zip(texts_clean, mask) if m]
    labels_filtered = np.array(labels)[mask]
    
    if len(texts_clean) < 10:
        print(f"Too few samples for {prefix}models. Need at least 10.")
        return
    
    # Check if we have at least 2 classes
    unique_labels = np.unique(labels_filtered)
    if len(unique_labels) < 2:
        print(f"Warning: {prefix}dataset contains only one class ({unique_labels[0]}). Skipping training.")
        print(f"  Total samples: {len(labels_filtered)}, Class distribution: {dict(zip(*np.unique(labels_filtered, return_counts=True)))}")
        return
    
    # Split - use stratify only if we have enough samples per class
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts_clean, labels_filtered, test_size=0.2, stratify=labels_filtered, random_state=42
        )
    except ValueError:
        # If stratification fails (e.g., one class has too few samples), use random split
        print(f"Warning: Stratified split failed for {prefix}. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            texts_clean, labels_filtered, test_size=0.2, random_state=42
        )
    
    # Verify we have both classes in training set
    unique_train_labels = np.unique(y_train)
    if len(unique_train_labels) < 2:
        print(f"Warning: Training set for {prefix}contains only one class ({unique_train_labels[0]}).")
        print(f"  Training samples: {len(y_train)}, Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  Skipping {prefix}model training.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # TF-IDF features for LR and SVM
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)
    scaler = StandardScaler()
    X_train_tf_scaled = scaler.fit_transform(X_train_tf.toarray())
    X_test_tf_scaled = scaler.transform(X_test_tf.toarray())
    
    # Save vectorizer and scaler
    import pickle
    with open(os.path.join(output_dir, f"{prefix}tfidf.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(output_dir, f"{prefix}scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    # Train LSTM
    if HAS_KERAS:
        train_lstm_model(
            X_train, y_train, X_test, y_test,
            os.path.join(output_dir, f"{prefix}lstm.keras"),
            os.path.join(output_dir, f"{prefix}tokenizer.json"),
            os.path.join(output_dir, f"{prefix}lstm_config.json"),
            prefix=prefix
        )
    
    # Train LR
    lr_result = train_lr_model(
        X_train_tf_scaled, y_train, X_test_tf_scaled, y_test,
        os.path.join(output_dir, f"{prefix}lr.pkl"),
        prefix=prefix
    )
    
    # Train SVM
    svm_result = train_svm_model(
        X_train_tf_scaled, y_train, X_test_tf_scaled, y_test,
        os.path.join(output_dir, f"{prefix}svm.pkl"),
        prefix=prefix
    )
    
    # Train BERT
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    bert_result = train_bert_model(
        X_train, y_train, X_test, y_test,
        os.path.join(output_dir, f"{prefix}bert.pt"),
        os.path.join(output_dir, f"{prefix}bert_tokenizer.json"),
        device,
        prefix=prefix
    )
    
    print(f"{prefix}models saved to {output_dir}")


def _whisper_generate(model, processor, input_features, device):
    """Run Whisper generate with English transcription and minimal warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return model.generate(
                input_features,
                language="en",
                task="transcribe",
            )
        except TypeError:
            return model.generate(input_features)


def _transcribe_audio_chunked(audio_16k, model, processor, device, chunk_samples=16000 * 30):
    """Transcribe audio in 30s chunks (Whisper limit) and join."""
    import numpy as np
    audio_16k = np.asarray(audio_16k, dtype=np.float32)
    if audio_16k.ndim > 1:
        audio_16k = audio_16k.mean(axis=1)
    if len(audio_16k) == 0:
        return ""
    texts = []
    for start in range(0, len(audio_16k), chunk_samples):
        chunk = audio_16k[start : start + chunk_samples]
        if len(chunk) < 100:
            continue
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = _whisper_generate(model, processor, inputs.input_features, device)
        t = processor.batch_decode(pred, skip_special_tokens=True)[0].strip()
        if t:
            texts.append(t)
    return " ".join(texts)


def _load_whisper(device="cuda"):
    """Load Whisper processor and model. Try cache first to avoid network timeout."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    # Try loading from local cache first (no network)
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-base", local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", local_files_only=True).to(device)
        return processor, model
    except Exception:
        pass
    
    # Fallback: download from Hub (may timeout on slow/unstable network)
    try:
        import os
        # Increase timeout for Hugging Face Hub (env var used by huggingface_hub)
        prev = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
        try:
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
            return processor, model
        finally:
            if prev is not None:
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = prev
            else:
                os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
    except Exception as e:
        raise RuntimeError(
            f"Could not load Whisper model: {e}. "
            "If offline, run once with internet to cache the model, or use --from_transcripts with existing CSV."
        ) from e


def transcribe_audio(audio_paths, labels, save_csv, device="cuda"):
    """Transcribe audio files using Whisper."""
    import librosa
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    processor, model = _load_whisper(device)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
    
    records = []
    for path, label in tqdm(list(zip(audio_paths, labels)), desc="Transcribing audio"):
        try:
            if not os.path.isfile(path):
                print(f"Skip (not found): {path}")
                continue
            audio, _ = librosa.load(path, sr=16000, mono=True)
            if audio is None or len(audio) == 0:
                print(f"Skip (empty audio): {path}")
                continue
            text = _transcribe_audio_chunked(audio, model, processor, device)
            records.append({"filename": os.path.basename(path), "text": text.strip(), "label": label})
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved {len(records)} transcripts to {save_csv}")
    return df


def _check_video_extraction_available():
    """Check if ffmpeg or moviepy/imageio_ffmpeg is available for video audio extraction."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    # imageio_ffmpeg (bundled with moviepy) provides an ffmpeg binary - no moviepy.editor needed
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        if get_ffmpeg_exe():
            return "moviepy"
    except (ImportError, RuntimeError, Exception):
        pass
    try:
        import moviepy.editor
        return "moviepy"
    except ImportError:
        return None


def transcribe_video(video_paths, labels, save_csv, device="cuda"):
    """Extract audio from videos and transcribe using Whisper."""
    from video_utils import extract_audio
    import librosa
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    ext = _check_video_extraction_available()
    if not ext:
        print("ERROR: Neither ffmpeg nor moviepy is available. Cannot extract audio from video.")
        print("  Install ffmpeg: https://ffmpeg.org/download.html (add to PATH)")
        print("  Or: pip install moviepy")
        return pd.DataFrame(columns=["filename", "text", "label"])
    print(f"Using {ext} for video audio extraction.")
    
    processor, model = _load_whisper(device)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
    
    records = []
    for path, label in tqdm(list(zip(video_paths, labels)), desc="Transcribing video"):
        wav_path = None
        try:
            if not os.path.isfile(path):
                print(f"Skip (file not found): {path}")
                continue
            wav_path = extract_audio(path, sample_rate=16000)
            if wav_path is None:
                print(f"Could not extract audio (install ffmpeg or moviepy): {path}")
                continue
            audio, _ = librosa.load(wav_path, sr=16000, mono=True)
            if audio is None or len(audio) == 0:
                print(f"Skip (empty extracted audio): {path}")
                continue
            text = _transcribe_audio_chunked(audio, model, processor, device)
            records.append({"filename": os.path.basename(path), "text": text.strip(), "label": label})
        except Exception as e:
            print(f"Skip {path}: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved {len(records)} transcripts to {save_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train all models (LSTM, LR, SVM, BERT) for all modalities")
    parser.add_argument("--audio_dataset", type=str, default=AUDIO_VIDEO_DATASET_PATH,
                        help="Path to audio dataset root")
    parser.add_argument("--video_dataset", type=str, default=AUDIO_VIDEO_DATASET_PATH,
                        help="Path to video dataset root")
    parser.add_argument("--text_dataset", type=str, default=DATASET_PATH,
                        help="Path to text dataset root")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Output directory for models")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"),
                        help="Device for training")
    parser.add_argument("--skip_audio", action="store_true", help="Skip audio training")
    parser.add_argument("--skip_video", action="store_true", help="Skip video training")
    parser.add_argument("--skip_text", action="store_true", help="Skip text training")
    parser.add_argument("--from_transcripts", action="store_true",
                        help="Load transcripts from CSV instead of transcribing")
    parser.add_argument("--use_text_models", action="store_true",
                        help="Use text models for audio/video (transcribe then use text models, don't train separate audio/video models)")
    args = parser.parse_args()
    
    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Train text models
    if not args.skip_text:
        print("\n" + "="*60)
        print("TRAINING TEXT MODELS")
        print("="*60)
        text_loader = TextDataLoader(args.text_dataset)
        texts_pos, labels_pos = text_loader.load_text_data(1)  # depressed
        texts_neg, labels_neg = text_loader.load_text_data(0)  # normal
        
        all_texts = texts_pos + texts_neg
        all_labels = labels_pos + labels_neg
        
        if all_texts:
            train_modality_models(all_texts, all_labels, output_dir, prefix="text_", device=device)
            # Copy text BERT to main app paths (bert_model.pt, bert_tokenizer.json)
            import shutil
            text_bert_pt = os.path.join(output_dir, "text_bert.pt")
            text_bert_tok = os.path.join(output_dir, "text_bert_tokenizer.json")
            main_bert_pt = os.path.join(output_dir, "bert_model.pt")
            main_bert_tok = os.path.join(output_dir, "bert_tokenizer.json")
            if os.path.exists(text_bert_pt):
                shutil.copy2(text_bert_pt, main_bert_pt)
                print(f"Copied text_bert.pt -> bert_model.pt (for app predictions)")
            if os.path.exists(text_bert_tok):
                shutil.copy2(text_bert_tok, main_bert_tok)
                print(f"Copied text_bert_tokenizer.json -> bert_tokenizer.json")
    
    # Transcribe audio; optionally train dedicated AUDIO models from transcripts
    if not args.skip_audio:
        print("\n" + "="*60)
        print("TRANSCRIBING AUDIO FILES")
        print("="*60)
        audio_paths, audio_labels = load_audio_paths_and_labels(args.audio_dataset)
        
        if audio_paths:
            transcripts_csv = os.path.join(output_dir, "audio_transcripts.csv")
            df = None
            if args.from_transcripts and os.path.exists(transcripts_csv):
                print("Loading transcripts from CSV...")
                try:
                    df = pd.read_csv(transcripts_csv)
                    if df.empty or 'text' not in df.columns or 'label' not in df.columns:
                        df = None
                except Exception as e:
                    print(f"Warning: Could not load audio CSV: {e}")
                    df = None
            if df is None:
                try:
                    print("Transcribing audio files...")
                    df = transcribe_audio(audio_paths, audio_labels, transcripts_csv, device=device)
                except (RuntimeError, Exception) as e:
                    if "timeout" in str(e).lower() or "ConnectTimeout" in str(type(e).__name__):
                        print("Audio transcription skipped: connection to Hugging Face timed out.")
                        print("  Tip: Run once with internet to cache Whisper, or use --from-transcripts with existing audio_transcripts.csv")
                    else:
                        print(f"Audio transcription failed: {e}")
                    df = pd.DataFrame(columns=["filename", "text", "label"])
            if df.empty or 'text' not in df.columns or 'label' not in df.columns:
                print("No audio transcripts available.")
            else:
                print(f"Loaded {len(df)} audio transcripts.")
                texts = df['text'].astype(str).tolist()
                labels_from_csv = df['label'].astype(int).tolist()
                n_classes = len(set(labels_from_csv))
                if args.use_text_models:
                    print("Using TEXT models for audio (no separate audio models will be trained).")
                else:
                    if n_classes < 2:
                        print("  Tip: To train audio models with both classes, re-run WITHOUT --from_transcripts")
                        print("  so audio is re-transcribed from the dataset (Normal + Depression/Stage1/Stage2).")
                    print("Training AUDIO models (LSTM, LR, SVM, BERT) from audio transcripts...")
                    train_modality_models(texts, labels_from_csv, output_dir, prefix="audio_", device=device)
        else:
            print("No audio files found!")
    
    # Transcribe video (extract audio then transcribe); optionally train dedicated VIDEO models
    if not args.skip_video:
        print("\n" + "="*60)
        print("TRANSCRIBING VIDEO FILES")
        print("="*60)
        video_paths, video_labels = load_video_paths_and_labels(args.video_dataset)
        
        if video_paths:
            transcripts_csv = os.path.join(output_dir, "video_transcripts.csv")
            df = None
            if args.from_transcripts and os.path.exists(transcripts_csv):
                print("Loading video transcripts from CSV...")
                try:
                    df = pd.read_csv(transcripts_csv)
                    if df.empty or 'text' not in df.columns or 'label' not in df.columns:
                        df = None
                except Exception as e:
                    print(f"Warning: Could not load video CSV: {e}")
                    df = None
            if df is None:
                try:
                    print("Extracting audio from videos and transcribing...")
                    df = transcribe_video(video_paths, video_labels, transcripts_csv, device=device)
                except (RuntimeError, Exception) as e:
                    if "timeout" in str(e).lower() or "ConnectTimeout" in str(type(e).__name__):
                        print("Video transcription skipped: connection to Hugging Face timed out.")
                        print("  Tip: Run once with internet to cache Whisper, or use --from-transcripts with existing video_transcripts.csv")
                    else:
                        print(f"Video transcription failed: {e}")
                    df = pd.DataFrame(columns=["filename", "text", "label"])
            if df.empty or 'text' not in df.columns or 'label' not in df.columns:
                print("No video transcripts available.")
            else:
                print(f"Loaded {len(df)} video transcripts.")
                texts = df['text'].astype(str).tolist()
                labels_from_csv = df['label'].astype(int).tolist()
                if args.use_text_models:
                    print("Using TEXT models for video (no separate video models will be trained).")
                else:
                    print("Training VIDEO models (LSTM, LR, SVM, BERT) from video transcripts...")
                    train_modality_models(texts, labels_from_csv, output_dir, prefix="video_", device=device)
        else:
            print("No video files found!")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Text models saved to: {output_dir}")
    print(f"Audio/Video transcripts saved to: {output_dir}")
    print("\nNote: Audio and video files are transcribed and analyzed using the SAME text models.")
    print("      No separate audio/video models are needed.")


if __name__ == "__main__":
    main()
