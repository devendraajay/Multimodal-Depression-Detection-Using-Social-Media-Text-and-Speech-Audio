"""
Pre-trained models for audio and video depression-related detection.

- Audio: HuggingFace speech emotion recognition (e.g. superb/hubert-large-superb-er).
  Emotion labels are mapped to a depression-risk score (sadness/anger -> higher risk).
- Video: Same as audio after extracting the audio track (no separate video model).
"""

import os
import numpy as np

# Emotion -> depression-risk probability (0-1). Based on literature linking negative emotions to depression.
# IEMOCAP/superb style: hap, sad, ang, neu
EMOTION_TO_DEPRESSION_RISK = {
    "sad": 0.85,
    "sadness": 0.85,
    "angry": 0.60,
    "anger": 0.60,
    "ang": 0.60,
    "fear": 0.55,
    "disgust": 0.50,
    "neutral": 0.40,
    "neu": 0.40,
    "happy": 0.20,
    "happiness": 0.20,
    "hap": 0.20,
    "surprise": 0.35,
    "surprised": 0.35,
}

# Default for unknown labels
DEFAULT_RISK = 0.45

_PIPELINE_CACHE = None


def _get_emotion_pipeline():
    """Lazy-load HuggingFace audio-classification pipeline for emotion (16 kHz)."""
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is not None:
        return _PIPELINE_CACHE
    try:
        from transformers import pipeline
        # superb/hubert-large-superb-er: 4 classes (happiness, sadness, anger, neutral) on IEMOCAP
        _PIPELINE_CACHE = pipeline(
            "audio-classification",
            model="superb/hubert-large-superb-er",
            top_k=4,
        )
        return _PIPELINE_CACHE
    except Exception as e:
        raise RuntimeError(f"Could not load pre-trained emotion model: {e}") from e


def _emotion_to_depression_prob(predictions):
    """
    Convert emotion pipeline output to a single depression-risk probability.
    predictions: list of {"label": "...", "score": float} from pipeline.
    Uses weighted average by score over emotion-to-risk mapping.
    """
    if not predictions:
        return DEFAULT_RISK
    total_score = 0.0
    weighted_risk = 0.0
    for p in predictions:
        label = (p.get("label") or "").strip().lower()
        score = float(p.get("score", 0))
        risk = EMOTION_TO_DEPRESSION_RISK.get(label)
        if risk is None:
            # try without last 's' or partial match
            for k, v in EMOTION_TO_DEPRESSION_RISK.items():
                if k in label or label in k:
                    risk = v
                    break
        if risk is None:
            risk = DEFAULT_RISK
        weighted_risk += risk * score
        total_score += score
    if total_score <= 0:
        return DEFAULT_RISK
    return float(np.clip(weighted_risk / total_score, 0.0, 1.0))


def predict_depression_from_audio(audio_path=None, audio_array=None, sr=16000):
    """
    Run pre-trained speech-emotion model on audio and return depression-risk result
    in the same shape as app.py expects for combined results (no per-model breakdown).

    Either audio_path (file path) or (audio_array, sr) must be provided.
    Audio must be mono, 16 kHz (or will be resampled if array provided).

    Returns:
        dict with keys: bert, lr, svm, lstm, combined.
        Only combined is filled; others are empty/None for UI compatibility.
    """
    if audio_path is None and audio_array is None:
        raise ValueError("Provide either audio_path or audio_array")

    if audio_path is not None:
        if not os.path.isfile(audio_path):
            raise ValueError("Audio file not found.")
        import librosa
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("Could not load audio (file may be empty or unsupported).")

    audio_array = np.asarray(audio_array, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if len(audio_array) == 0:
        raise ValueError("Audio array is empty.")

    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    pipeline_fn = _get_emotion_pipeline()
    try:
        predictions = pipeline_fn({"array": audio_array, "sampling_rate": 16000})
    except Exception as e:
        raise RuntimeError(f"Pre-trained emotion model failed: {e}") from e

    prob = _emotion_to_depression_prob(predictions)
    uncertainty = 1.0 - max(prob, 1 - prob)  # simple uncertainty
    risk_level = "Depressed" if prob >= 0.5 else "Not depressed"

    # Same structure as analyze_texts_all_models / _speech_results_to_text_format
    results = {
        "bert": {"probs": [], "uncertainties": [], "avg": None, "avg_uncertainty": None, "risk_level": None},
        "lr": {"probs": [], "uncertainties": [], "avg": None, "avg_uncertainty": None, "risk_level": None},
        "svm": {"probs": [], "uncertainties": [], "avg": None, "avg_uncertainty": None, "risk_level": None},
        "lstm": {"probs": [], "uncertainties": [], "avg": None, "avg_uncertainty": None, "risk_level": None},
        "combined": {
            "avg": prob,
            "avg_uncertainty": uncertainty,
            "risk_level": risk_level,
            "model_disagreement": None,
        },
    }
    return results
