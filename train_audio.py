"""
Audio pipeline for depression detection (nlp(1).ipynb style).
1. Load audio paths by class (normal / depressed)
2. Transcribe with Whisper (16 kHz)
3. Clean text
4. Train LSTM and/or TF-IDF+SVM; save to models/
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from config import AUDIO_VIDEO_DATASET_PATH
except ImportError:
    AUDIO_VIDEO_DATASET_PATH = "Multimodel_Dataset"

from data_loader_audio import load_audio_paths_and_labels, CLASSES


def clean_text(text: str) -> str:
    """Same as nlp(1).ipynb: lower, remove non-alpha, normalize spaces."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_whisper_transcribe(audio_paths: list, labels: list, save_csv: str, device: str = "cuda"):
    """Load each audio at 16 kHz, run Whisper, save CSV (filename, text, label)."""
    import librosa
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
    model.config.forced_decoder_ids = None

    records = []
    for path, label in tqdm(list(zip(audio_paths, labels)), desc="Transcribing"):
        try:
            audio, _ = librosa.load(path, sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                predicted_ids = model.generate(inputs.input_features)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            records.append({"filename": os.path.basename(path), "text": text.strip(), "label": label})
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved {len(records)} transcripts to {save_csv}")
    return df


def train_audio_models(transcripts_csv: str, output_dir: str):
    """From transcripts CSV: clean, then train LSTM and SVM; save artifacts."""
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    df = pd.read_csv(transcripts_csv)
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    texts = df["clean_text"].tolist()
    labels = np.array(df["label"].values)

    # Drop empty
    mask = [bool(t.strip()) for t in texts]
    texts = [t for t, m in zip(texts, mask) if m]
    labels = labels[mask]
    if len(texts) < 10:
        print("Too few samples after cleaning. Need at least 10.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    os.makedirs(output_dir, exist_ok=True)

    # --- LSTM (Keras) ---
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
        preds = (model.predict(X_test_seq, verbose=0).squeeze() > 0.5).astype(int)
        print("Audio LSTM test accuracy:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds, zero_division=0))
        model.save(os.path.join(output_dir, "audio_lstm.keras"))
        with open(os.path.join(output_dir, "audio_tokenizer.json"), "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json())
        with open(os.path.join(output_dir, "audio_lstm_config.json"), "w", encoding="utf-8") as f:
            json.dump({"maxlen": MAXLEN, "num_words": MAX_WORDS}, f)
    except ImportError as e:
        print("Keras not available, skip LSTM:", e)

    # --- SVM (TF-IDF) ---
    # Guard against degenerate case where training split has only one class
    unique_train_labels = np.unique(y_train)
    if len(unique_train_labels) < 2:
        print(f"Skipping SVM training: need at least 2 classes in y_train, got {unique_train_labels}.")
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
        print("Audio SVM test accuracy:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds, zero_division=0))
        print(classification_report(y_test, preds, target_names=["normal", "depressed"]))

        import pickle
        with open(os.path.join(output_dir, "audio_tfidf.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)
        with open(os.path.join(output_dir, "audio_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(output_dir, "audio_svm.pkl"), "wb") as f:
            pickle.dump(svm, f)
        print(f"Audio models saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train audio pipeline (Whisper -> clean -> LSTM/SVM)")
    parser.add_argument("--dataset_path", type=str, default=AUDIO_VIDEO_DATASET_PATH,
                        help="Path to audio/video dataset root (class folders or audio/<class>/)")
    parser.add_argument("--audio_subdir", type=str, default="",
                        help="Optional: subdir under dataset_path for audio (e.g. 'audio')")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Where to save transcripts and models")
    parser.add_argument("--transcripts_only", action="store_true",
                        help="Only run Whisper and save CSV; skip LSTM/SVM training")
    parser.add_argument("--from_csv", type=str, default="",
                        help="Skip Whisper; load transcripts from this CSV and train LSTM/SVM")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.from_csv:
        train_audio_models(args.from_csv, output_dir)
        return

    dataset_path = args.dataset_path
    if args.audio_subdir:
        dataset_path = os.path.join(dataset_path, args.audio_subdir)
    paths, labels = load_audio_paths_and_labels(dataset_path)
    if not paths:
        print("No audio files found. Check path and folder layout (e.g. <path>/normal/*.wav, <path>/depressed/*.wav)")
        return

    transcripts_csv = os.path.join(output_dir, "audio_transcripts.csv")
    if not args.from_csv:
        run_whisper_transcribe(paths, labels, transcripts_csv, device=args.device)

    if not args.transcripts_only:
        train_audio_models(transcripts_csv, output_dir)


if __name__ == "__main__":
    main()
