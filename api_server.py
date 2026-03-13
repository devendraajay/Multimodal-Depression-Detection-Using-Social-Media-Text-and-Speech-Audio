"""
Flask API server for the React frontend.
Serves /api/predict-text, /api/predict-json (JSON file input, all 4 models), /api/health.

Run from project root: python api_server.py
Then start the frontend (cd website && npm run dev) — Vite proxies /api to http://localhost:5000.
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to call API when using direct URL (e.g. dev mode)


def extract_text_from_json(json_data):
    """
    Extract text content from JSON data (same logic as app.py).
    Supports: single tweet {"text": "..."}, array of tweets, user data with tweets, etc.
    """
    texts = []
    if isinstance(json_data, dict):
        if "text" in json_data:
            t = json_data.get("text", "")
            if t and isinstance(t, str) and t.strip():
                texts.append(t.strip())
        if "tweets" in json_data:
            for tweet in json_data["tweets"] or []:
                if isinstance(tweet, dict) and "text" in tweet:
                    t = tweet.get("text", "")
                    if t and isinstance(t, str) and t.strip():
                        texts.append(t.strip())
        for key in ("content", "post", "message", "body", "full_text"):
            if key in json_data:
                t = json_data.get(key, "")
                if t and isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        for key, value in json_data.items():
            if isinstance(value, dict) and "text" in value:
                t = value.get("text", "")
                if t and isinstance(t, str) and t.strip():
                    texts.append(t.strip())
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict):
                if "text" in item:
                    t = item.get("text", "")
                    if t and isinstance(t, str) and t.strip():
                        texts.append(t.strip())
                texts.extend(extract_text_from_json(item))
            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())
    seen = set()
    return [t for t in texts if t not in seen and not seen.add(t)]

# Optional: load real prediction logic from app (may fail if Streamlit runs at import)
_predict_text_real = None
_clean_text = None
_clean_text_for_bert = None
_load_vectorizer = _load_lr = _load_svm = _load_bert = _load_text_lstm = None
_predict_proba_ml = _predict_proba_bert = _predict_proba_lstm = None
# Audio / video specific loaders and predictors (optional)
_load_audio_models = _load_video_models = None
_predict_from_audio_path = _predict_from_video_path = None


def _try_load_app():
    global _predict_text_real, _clean_text, _clean_text_for_bert, _load_vectorizer, _load_lr, _load_svm
    global _load_bert, _load_text_lstm, _predict_proba_ml, _predict_proba_bert, _predict_proba_lstm
    global _load_audio_models, _load_video_models, _predict_from_audio_path, _predict_from_video_path
    if _predict_text_real is not None:
        return
    try:
        import app as app_module
        _clean_text = getattr(app_module, "clean_text_for_ml", None)
        _clean_text_for_bert = getattr(app_module, "clean_text_for_bert", None)
        _load_vectorizer = getattr(app_module, "load_vectorizer", None)
        _load_lr = getattr(app_module, "load_lr_model", None)
        _load_svm = getattr(app_module, "load_svm_model", None)
        _load_bert = getattr(app_module, "load_bert_model", None)
        _load_text_lstm = getattr(app_module, "load_text_lstm", None)
        _predict_proba_ml = getattr(app_module, "predict_proba_ml", None)
        _predict_proba_bert = getattr(app_module, "predict_proba_bert", None)
        _predict_proba_lstm = getattr(app_module, "predict_proba_lstm", None)
        # Optional audio/video helpers
        _load_audio_models = getattr(app_module, "load_audio_models", None)
        _load_video_models = getattr(app_module, "load_video_models", None)
        _predict_from_audio_path = getattr(app_module, "predict_from_audio_path", None)
        _predict_from_video_path = getattr(app_module, "predict_from_video_path", None)
        if all([
            _clean_text, _load_vectorizer, _load_lr, _load_svm,
            _load_bert, _load_text_lstm,
            _predict_proba_ml, _predict_proba_bert, _predict_proba_lstm
        ]):
            _predict_text_real = "ok"
    except Exception as e:
        print(f"API: using mock predictions (real models not loaded: {e})")


def _mock_response(text_clean, model_key):
    """Return a valid response shape so the frontend never 404s."""
    return {
        "cleaned_text": text_clean or (text_clean if text_clean is not None else ""),
        "probability": 0.35,
        "confidence": 0.75,
        "emotion_probabilities": {"neutral": 0.5, "sad": 0.3, "anxious": 0.2},
    }


def _results_to_all_models_response(results):
    """Convert app.py result (bert/lr/svm/lstm/combined) to API shape with all 4 models for frontend."""
    if not results or not isinstance(results, dict):
        return None
    combined = results.get("combined") or {}
    model_names = {"bert": "BERT", "lr": "Logistic Regression", "svm": "SVM", "lstm": "LSTM"}
    models_out = {}
    for key, name in model_names.items():
        r = results.get(key, {})
        avg = r.get("avg")
        if avg is not None:
            models_out[name] = {
                "probs": r.get("probs") or [avg],
                "avg": round(avg, 4),
                "risk_level": r.get("risk_level") or ("Depressed" if avg >= 0.5 else "Not depressed"),
            }
    return {
        "models": models_out,
        "combined": {
            "avg": combined.get("avg"),
            "risk_level": combined.get("risk_level"),
            "model_disagreement": combined.get("model_disagreement"),
        },
    }


def _predict_one_model(text, model_key, text_clean):
    """Run one model and return { cleaned_text, probability, confidence, emotion_probabilities }."""
    _try_load_app()
    if _predict_text_real is None:
        return _mock_response(text_clean, model_key)

    try:
        # Map frontend model names to backend
        model_map = {
            "logistic_regression": "lr",
            "svm": "svm",
            "lstm": "lstm",
            "bert": "bert",
        }
        backend_key = model_map.get(model_key, model_key)
        prob, confidence = None, None

        if backend_key == "lr" and _load_lr and _load_vectorizer and _predict_proba_ml:
            vec = _load_vectorizer()
            lr = _load_lr()
            prob, unc = _predict_proba_ml(vec, lr, text_clean)
            confidence = 1.0 - unc if unc is not None else 0.8
        elif backend_key == "svm" and _load_svm and _load_vectorizer and _predict_proba_ml:
            vec = _load_vectorizer()
            svm = _load_svm()
            prob, unc = _predict_proba_ml(vec, svm, text_clean)
            confidence = 1.0 - unc if unc is not None else 0.8
        elif backend_key == "lstm" and _load_text_lstm and _predict_proba_lstm:
            lstm_model, lstm_tok, lstm_maxlen = _load_text_lstm()
            if lstm_model is not None:
                prob, unc = _predict_proba_lstm(lstm_model, lstm_tok, lstm_maxlen, text_clean)
                confidence = 1.0 - unc if unc is not None else 0.8
        elif backend_key == "bert" and _load_bert and _predict_proba_bert:
            bert_model, bert_tok, device, _ = _load_bert()
            # BERT was trained with train_multimodal.clean_text (lower, [a-z\s] only, no stopwords)
            bert_text = _clean_text_for_bert(text) if _clean_text_for_bert else text.lower().strip() or text
            if not bert_text.strip():
                bert_text = text.strip().lower() or text
            prob, unc = _predict_proba_bert(bert_model, bert_tok, device, bert_text)
            confidence = 1.0 - unc if unc is not None else 0.8

        if prob is not None:
            return {
                "cleaned_text": text_clean,
                "probability": float(prob),
                "confidence": float(confidence) if confidence is not None else 0.8,
                "emotion_probabilities": {},
            }
    except Exception as e:
        print(f"Predict {model_key} error: {e}")
    return _mock_response(text_clean, model_key)


@app.route("/api/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    model = (data.get("model") or "logistic_regression").strip()

    if not text:
        return jsonify({"error": "Missing or empty 'text'"}), 400

    text_clean = _clean_text(text) if _clean_text else (text.lower().strip())
    if not text_clean:
        text_clean = text

    result = _predict_one_model(text, model, text_clean)
    return jsonify(result)


# Backwards-compatible routes without /api prefix (in case frontend calls these)
@app.route("/predict-text", methods=["POST"])
def predict_text_no_prefix():
    return predict_text()


@app.route("/api/predict-json", methods=["POST"])
def predict_json():
    """
    Accept JSON body (tweet-style or { "texts": ["...", "..."] }) or file upload.
    Extract texts, run all 4 models (BERT, LR, SVM, LSTM), return combined + per-model results.
    """
    texts = []
    content_type = request.content_type or ""
    if "application/json" in content_type:
        data = request.get_json(silent=True) or {}
        if "texts" in data and isinstance(data["texts"], list):
            texts = [str(t).strip() for t in data["texts"] if t and str(t).strip()]
        else:
            texts = extract_text_from_json(data)
    elif "multipart/form-data" in content_type and "file" in request.files:
        f = request.files["file"]
        if f.filename and f.filename.lower().endswith(".json"):
            try:
                import json as _json
                data = _json.load(f)
                if "texts" in data and isinstance(data["texts"], list):
                    texts = [str(t).strip() for t in data["texts"] if t and str(t).strip()]
                else:
                    texts = extract_text_from_json(data)
            except Exception as e:
                return jsonify({"error": f"Invalid JSON file: {e}"}), 400
        else:
            return jsonify({"error": "Upload a .json file"}), 400
    else:
        return jsonify({"error": "Send JSON body or multipart file (key: file)"}), 400

    if not texts:
        return jsonify({"error": "No text content found in JSON"}), 400

    _try_load_app()
    if _predict_text_real is None:
        return jsonify({
            "error": "Models not loaded. Train with train_models.py and ensure models/ exists.",
            "texts": texts[:5],
            "models": {},
            "combined": {"avg": 0.5, "risk_level": "Unknown", "model_disagreement": 0},
        })

    model_keys = ["bert", "lr", "svm", "lstm"]
    model_names = {"bert": "BERT", "lr": "Logistic Regression", "svm": "SVM", "lstm": "LSTM"}
    backend_map = {"bert": "bert", "lr": "logistic_regression", "svm": "svm", "lstm": "lstm"}
    results = {k: {"probs": [], "avg": None, "risk_level": None} for k in model_keys}
    results["combined"] = {"avg": None, "risk_level": None, "model_disagreement": None}

    for text in texts:
        if not text.strip():
            continue
        text_clean = _clean_text(text) if _clean_text else text.lower().strip() or text
        for key in model_keys:
            out = _predict_one_model(text, backend_map[key], text_clean)
            p = out.get("probability")
            if p is not None:
                results[key]["probs"].append(float(p))

    for key in model_keys:
        probs = results[key]["probs"]
        if probs:
            avg = sum(probs) / len(probs)
            results[key]["avg"] = round(avg, 4)
            results[key]["risk_level"] = "Depressed" if avg >= 0.5 else "Not depressed"

    avgs = [results[k]["avg"] for k in model_keys if results[k]["avg"] is not None]
    if avgs:
        mean_avg = sum(avgs) / len(avgs)
        results["combined"]["avg"] = round(mean_avg, 4)
        results["combined"]["risk_level"] = "Depressed" if mean_avg >= 0.5 else "Not depressed"
        if len(avgs) > 1:
            variance = sum((x - mean_avg) ** 2 for x in avgs) / len(avgs)
            results["combined"]["model_disagreement"] = round(variance ** 0.5, 4)
        else:
            results["combined"]["model_disagreement"] = 0.0

    return jsonify({
        "texts": texts[:100],
        "text_count": len(texts),
        "models": {model_names[k]: results[k] for k in model_keys},
        "combined": results["combined"],
    })


@app.route("/predict-json", methods=["POST"])
def predict_json_no_prefix():
    return predict_json()


@app.route("/api/predict-audio", methods=["POST", "OPTIONS"])
def predict_audio():
    if request.method == "OPTIONS":
        return "", 204
    """
    Accept an uploaded audio file; use pre-trained speech-emotion model for depression-related detection.
    """
    _try_load_app()
    if _predict_from_audio_path is None:
        return jsonify({"error": "Audio pipeline not available in backend."}), 500

    if "file" not in request.files:
        return jsonify({"error": "Missing file field 'file'"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    import tempfile
    import os as _os

    suffix = _os.path.splitext(f.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        results = _predict_from_audio_path(tmp_path)
        if not results or not isinstance(results, dict):
            return jsonify({"error": "Audio prediction failed."}), 500
        combined = results.get("combined") or {}
        prob = combined.get("avg")
        if prob is None:
            return jsonify({"error": "Audio prediction did not return a probability."}), 500
        unc = combined.get("avg_uncertainty")
        confidence = float(1.0 - unc) if isinstance(unc, (int, float)) else 0.8
        label = combined.get("risk_level") or ("Depressed" if prob >= 0.5 else "Not depressed")
        payload = {
            "probability": float(prob),
            "confidence": confidence,
            "label": label,
            "cleaned_text": "",
        }
        all_models = _results_to_all_models_response(results)
        if all_models:
            payload["models"] = all_models["models"]
            payload["combined"] = all_models["combined"]
        return jsonify(payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            _os.remove(tmp_path)
        except OSError:
            pass


@app.route("/predict-audio", methods=["POST", "OPTIONS"])
def predict_audio_no_prefix():
    return predict_audio()


@app.route("/api/predict-video", methods=["POST", "OPTIONS"])
def predict_video():
    if request.method == "OPTIONS":
        return "", 204
    """
    Accept an uploaded video file; extract audio and use pre-trained speech-emotion model.
    """
    _try_load_app()
    if _predict_from_video_path is None:
        return jsonify({"error": "Video pipeline not available in backend."}), 500

    if "file" not in request.files:
        return jsonify({"error": "Missing file field 'file'"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    import tempfile
    import os as _os

    suffix = _os.path.splitext(f.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        results = _predict_from_video_path(tmp_path)
        if not results or not isinstance(results, dict):
            return jsonify({"error": "Video prediction failed."}), 500
        combined = results.get("combined") or {}
        prob = combined.get("avg")
        if prob is None:
            return jsonify({"error": "Video prediction did not return a probability."}), 500
        unc = combined.get("avg_uncertainty")
        confidence = float(1.0 - unc) if isinstance(unc, (int, float)) else 0.8
        label = combined.get("risk_level") or ("Depressed" if prob >= 0.5 else "Not depressed")
        payload = {
            "probability": float(prob),
            "confidence": confidence,
            "label": label,
            "cleaned_text": "",
        }
        all_models = _results_to_all_models_response(results)
        if all_models:
            payload["models"] = all_models["models"]
            payload["combined"] = all_models["combined"]
        return jsonify(payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            _os.remove(tmp_path)
        except OSError:
            pass


@app.route("/predict-video", methods=["POST", "OPTIONS"])
def predict_video_no_prefix():
    return predict_video()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/routes", methods=["GET"])
def routes_list():
    """List registered API routes (so you can confirm predict-audio and predict-video exist)."""
    rules = [r.rule for r in app.url_map.iter_rules() if r.rule.startswith("/api/")]
    return jsonify({"routes": sorted(rules)})


@app.errorhandler(404)
def not_found(e):
    # Help debug wrong URLs by including the path and method that failed.
    from flask import request as _request
    path = _request.path
    method = _request.method
    print(f"404 for {method} {path}")
    return jsonify({
        "error": "Not found",
        "path": path,
        "method": method,
        "message": "API route does not exist. Use /api/predict-text, /api/predict-json, /api/predict-audio, /api/predict-video, /api/health",
    }), 404


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return which of the 4 models are available (for UI)."""
    _try_load_app()
    available = []
    if _load_lr and _load_vectorizer:
        try:
            _load_lr()
            available.append("logistic_regression")
        except Exception:
            pass
    if _load_svm and _load_vectorizer:
        try:
            _load_svm()
            available.append("svm")
        except Exception:
            pass
    if _load_text_lstm:
        try:
            m, t, _ = _load_text_lstm()
            if m is not None:
                available.append("lstm")
        except Exception:
            pass
    if _load_bert:
        try:
            _load_bert()
            available.append("bert")
        except Exception:
            pass
    return jsonify({"models": available})


if __name__ == "__main__":
    # Use 5001 by default to avoid conflict with Windows/other apps on 5000
    port = int(os.environ.get("PORT", 5001))
    api_routes = sorted(r.rule for r in app.url_map.iter_rules() if r.rule.startswith("/api/"))
    print(f"API server starting on http://127.0.0.1:{port}")
    print("Registered routes:", ", ".join(api_routes))
    print("Start frontend: cd website && npm run dev")
    print(f"Then open the app and use Audio/Video - backend must be at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
