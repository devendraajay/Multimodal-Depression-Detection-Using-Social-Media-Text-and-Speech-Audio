"""
Evaluate text models (LR, SVM, BERT, LSTM) on audio and video transcripts.

Audio/video are first transcribed to text (already saved in CSVs), and then
the SAME text models used for tweet text are applied to those transcripts.
This script reports per-model metrics and confusion matrices.
"""

import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import app as app_module


def _load_text_models():
    """Load core text models once from app.py helpers."""
    vectorizer = app_module.load_vectorizer()
    lr_model = app_module.load_lr_model()
    svm_model = app_module.load_svm_model()
    bert_model, bert_tokenizer, device, _ = app_module.load_bert_model()
    lstm_model, lstm_tokenizer, lstm_maxlen = app_module.load_text_lstm()
    return {
        "vectorizer": vectorizer,
        "lr_model": lr_model,
        "svm_model": svm_model,
        "bert_model": bert_model,
        "bert_tokenizer": bert_tokenizer,
        "device": device,
        "lstm_model": lstm_model,
        "lstm_tokenizer": lstm_tokenizer,
        "lstm_maxlen": lstm_maxlen or 128,
    }


def _eval_transcripts(csv_path: str, label: str, models: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all four text models on a transcripts CSV."""
    if not os.path.exists(csv_path):
        return {"error": f"{csv_path} not found"}

    df = pd.read_csv(csv_path)
    if df.empty or "text" not in df.columns or "label" not in df.columns:
        return {"error": f"{csv_path} missing text/label columns or is empty"}

    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).to_numpy()

    # Use app's multi-model helper so logic matches the UI/API exactly
    res = app_module.analyze_texts_all_models(
        texts,
        models["bert_model"],
        models["bert_tokenizer"],
        models["device"],
        models["vectorizer"],
        models["lr_model"],
        models["svm_model"],
        lstm_model=models["lstm_model"],
        lstm_tokenizer=models["lstm_tokenizer"],
        lstm_maxlen=models["lstm_maxlen"],
        raw_texts=texts,
    )
    if not res:
        return {"error": "analyze_texts_all_models returned no results"}

    out: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "models": {},
    }

    for key, pretty in [
        ("lr", "Logistic Regression"),
        ("svm", "SVM"),
        ("bert", "BERT"),
        ("lstm", "LSTM"),
    ]:
        probs = np.array(res[key]["probs"], dtype=float)
        if probs.size == 0:
            continue
        # Align labels length just in case texts were skipped
        n = min(len(y_true), len(probs))
        y = y_true[:n]
        p = probs[:n]
        y_pred = (p >= 0.5).astype(int)

        # Force a full 2x2 matrix even if one class is missing
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["models"][pretty] = {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    return out


def main() -> None:
    base = os.path.normpath("models")
    models = _load_text_models()

    audio_csv = os.path.join(base, "audio_transcripts.csv")
    video_csv = os.path.join(base, "video_transcripts.csv")

    results = {
        "audio": _eval_transcripts(audio_csv, "audio", models),
        "video": _eval_transcripts(video_csv, "video", models),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

