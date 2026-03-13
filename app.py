"""
Streamlit Web Application for Depression Detection
Uses locally trained models only (no pretrained model weights)
"""

import streamlit as st
import os
import json
import re
import torch
import numpy as np
from feature_extraction import FeatureExtractor
from models_ml import LogisticRegressionScratch, SVMScratch, probability_uncertainty
from model_bert import BERTDepressionModel, SimpleTokenizer

# NLTK stopwords for text cleaning (must match training pipeline)
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


def clean_text_for_ml(text: str) -> str:
    """
    Clean text the SAME way as during training (data_loader.clean_text).
    CRITICAL: Logistic Regression and SVM were trained on cleaned text;
    we must apply the same preprocessing at prediction time.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in _STOP_WORDS]
    text = ' '.join(words)
    text = ' '.join(text.split())
    return text.strip()

# Page config
st.set_page_config(
    page_title="Depression Detection from Social Media",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid;
        transition: transform 0.2s;
    }
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .depressed {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-color: #d63031;
        color: #ffffff;
    }
    .depressed h2, .depressed h3, .depressed p {
        color: #ffffff;
    }
    .not-depressed {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        border-color: #0984e3;
        color: #ffffff;
    }
    .not-depressed h2, .not-depressed h3, .not-depressed p {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Resolve model dir relative to this file so Streamlit works from any cwd
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_APP_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_model.pt')
BERT_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'bert_tokenizer.json')
BERT_MAX_LENGTH = 128
TEXT_LSTM_PATH = os.path.join(MODEL_DIR, 'text_lstm.keras')
TEXT_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'text_tokenizer.json')
TEXT_LSTM_CONFIG_PATH = os.path.join(MODEL_DIR, 'text_lstm_config.json')
# Audio pipeline (Whisper -> LSTM/SVM)
AUDIO_LSTM_PATH = os.path.join(MODEL_DIR, 'audio_lstm.keras')
AUDIO_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'audio_tokenizer.json')
AUDIO_LSTM_CONFIG_PATH = os.path.join(MODEL_DIR, 'audio_lstm_config.json')
AUDIO_TFIDF_PATH = os.path.join(MODEL_DIR, 'audio_tfidf.pkl')
AUDIO_SCALER_PATH = os.path.join(MODEL_DIR, 'audio_scaler.pkl')
AUDIO_SVM_PATH = os.path.join(MODEL_DIR, 'audio_svm.pkl')
# Video pipeline
VIDEO_LSTM_PATH = os.path.join(MODEL_DIR, 'video_lstm.keras')
VIDEO_TOKENIZER_PATH = os.path.join(MODEL_DIR, 'video_tokenizer.json')
VIDEO_LSTM_CONFIG_PATH = os.path.join(MODEL_DIR, 'video_lstm_config.json')
VIDEO_TFIDF_PATH = os.path.join(MODEL_DIR, 'video_tfidf.pkl')
VIDEO_SCALER_PATH = os.path.join(MODEL_DIR, 'video_scaler.pkl')
VIDEO_SVM_PATH = os.path.join(MODEL_DIR, 'video_svm.pkl')


_vectorizer_cache = None


def load_vectorizer():
    """Load TF-IDF vectorizer. Works in both Streamlit and Flask/API context."""
    global _vectorizer_cache
    if _vectorizer_cache is not None:
        return _vectorizer_cache
    fe = FeatureExtractor()
    fe.load(VECTORIZER_PATH)
    _vectorizer_cache = fe
    return fe


def _get_device(prefer_cuda=True):
    """Use GPU (cuda) if available, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda'), f"GPU ({torch.cuda.get_device_name(0)})"
    return torch.device('cpu'), "CPU"


_bert_model_cache = None


def load_bert_model():
    """Load local transformer model and tokenizer on GPU if available. Works in both Streamlit and Flask/API."""
    global _bert_model_cache
    if _bert_model_cache is not None:
        return _bert_model_cache
    device, device_label = _get_device(prefer_cuda=True)
    
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
        # The checkpoint contains the exact vocab_size used during training
        saved_vocab_size = checkpoint.get('vocab_size', None)
        if saved_vocab_size is None:
            saved_vocab_size = model_config.get('vocab_size', None)
        
        if saved_vocab_size is None:
            # Last resort: check the state_dict for token_embedding shape
            if 'token_embedding.weight' in checkpoint['state_dict']:
                saved_vocab_size = checkpoint['state_dict']['token_embedding.weight'].shape[0]
            else:
                # Use tokenizer vocab_size as fallback
                saved_vocab_size = tokenizer.vocab_size
        
        # Use saved vocab_size to ensure exact match
        model_config['vocab_size'] = saved_vocab_size
        
        # Ensure max_length is set
        if 'max_length' not in model_config:
            saved_max_length = checkpoint.get('max_length', BERT_MAX_LENGTH)
            model_config['max_length'] = saved_max_length
        
        # Create model with exact saved configuration
        model = BERTDepressionModel(**model_config).to(device)
        
        # Load state dict with strict=True to catch any mismatches
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as e:
            # If strict loading fails, try non-strict and warn
            print(f"Warning: Strict loading failed: {e}")
            print(f"Attempting non-strict loading...")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # Fallback: try to infer vocab_size from checkpoint
        if isinstance(checkpoint, dict):
            # Try to find vocab_size from state_dict
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
    _bert_model_cache = (model, tokenizer, device, device_label)
    return _bert_model_cache


def load_lr_model():
    """Load Logistic Regression (custom)"""
    m = LogisticRegressionScratch()
    m.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
    return m


def load_svm_model():
    """Load SVM model"""
    m = SVMScratch()
    m.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    return m


def load_text_lstm():
    """Load text LSTM model, tokenizer, and config (Keras). Returns (model, tokenizer, maxlen) or (None, None, None)."""
    if not os.path.exists(TEXT_LSTM_PATH) or not os.path.exists(TEXT_TOKENIZER_PATH):
        return None, None, None
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(TEXT_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            tokenizer = tokenizer_from_json(f.read())
        try:
            model = load_model(TEXT_LSTM_PATH, safe_mode=False)
        except TypeError:
            model = load_model(TEXT_LSTM_PATH)
        maxlen = 128
        if os.path.exists(TEXT_LSTM_CONFIG_PATH):
            with open(TEXT_LSTM_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                maxlen = cfg.get("maxlen", 128)
        return model, tokenizer, maxlen
    except Exception as e:
        print(f"Text LSTM load error: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"LSTM load failed: {e}") from e


def clean_text_speech(text: str) -> str:
    """Same as nlp(1).ipynb / train_audio: lower, remove non-alpha, normalize spaces (for audio/video transcripts)."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_for_bert(text: str) -> str:
    """
    Clean text the SAME way as during BERT training (train_multimodal.clean_text).
    BERT was trained on: lower, remove non-alpha, normalize spaces. NO stopword removal.
    CRITICAL: Must match train_multimodal.clean_text exactly for correct predictions.
    """
    return clean_text_speech(text)


def predict_proba_lstm(model, tokenizer, maxlen, text):
    """
    Get depression probability for text using the text LSTM.
    Text must be cleaned the same way as during training (clean_text_for_ml).
    Returns (probability, uncertainty).
    """
    if model is None or tokenizer is None:
        return None, None
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        text_cleaned = clean_text_for_ml(text)
        if not text_cleaned:
            text_cleaned = text.strip().lower() or text
        seq = tokenizer.texts_to_sequences([text_cleaned])
        X = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prob = float(np.squeeze(model.predict(X, verbose=0)))
        prob = np.clip(prob, 0.0, 1.0)
        return prob, probability_uncertainty(prob)
    except Exception as e:
        raise RuntimeError(f"LSTM prediction error: {str(e)}")


_whisper_cache = None


def load_whisper():
    """Load Whisper for audio/video transcription. Works in both Streamlit and Flask/API context."""
    global _whisper_cache
    if _whisper_cache is not None:
        return _whisper_cache
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
    _whisper_cache = (processor, model, device)
    return _whisper_cache


# Whisper-base expects up to 30 seconds per chunk
WHISPER_CHUNK_SAMPLES = 16000 * 30


def _transcribe_audio_array(audio_16k, processor, model, device):
    """Transcribe mono 16 kHz float array. Returns text (English). Chunks long audio for Whisper 30s limit."""
    import warnings
    import numpy as np
    audio_16k = np.asarray(audio_16k, dtype=np.float32)
    if audio_16k.ndim > 1:
        audio_16k = audio_16k.mean(axis=1)
    if len(audio_16k) == 0:
        return ""
    texts = []
    for start in range(0, len(audio_16k), WHISPER_CHUNK_SAMPLES):
        chunk = audio_16k[start : start + WHISPER_CHUNK_SAMPLES]
        if len(chunk) < 100:
            continue
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                ids = model.generate(
                    inputs.input_features,
                    language="en",
                    task="transcribe",
                )
            except TypeError:
                ids = model.generate(inputs.input_features)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            texts.append(text)
    return " ".join(texts)


_audio_models_cache = None


def load_audio_models():
    """Load all audio models (LSTM, LR, SVM, BERT). Works in both Streamlit and Flask/API."""
    global _audio_models_cache
    if _audio_models_cache is not None:
        return _audio_models_cache
    import pickle
    models = {}
    
    # Load LSTM
    audio_lstm_path = os.path.join(MODEL_DIR, "audio_lstm.keras")
    audio_tokenizer_path = os.path.join(MODEL_DIR, "audio_tokenizer.json")
    audio_lstm_config_path = os.path.join(MODEL_DIR, "audio_lstm_config.json")
    if os.path.exists(audio_lstm_path) and os.path.exists(audio_tokenizer_path):
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            with open(audio_tokenizer_path, "r", encoding="utf-8") as f:
                tok = tokenizer_from_json(f.read())
            mdl = load_model(audio_lstm_path)
            maxlen = 64
            if os.path.exists(audio_lstm_config_path):
                with open(audio_lstm_config_path, "r", encoding="utf-8") as f:
                    maxlen = json.load(f).get("maxlen", 64)
            models['lstm'] = (mdl, tok, maxlen)
        except Exception as e:
            print(f"Could not load audio LSTM: {e}")
    
    # Load LR and SVM (share TF-IDF vectorizer)
    audio_tfidf_path = os.path.join(MODEL_DIR, "audio_tfidf.pkl")
    audio_scaler_path = os.path.join(MODEL_DIR, "audio_scaler.pkl")
    audio_lr_path = os.path.join(MODEL_DIR, "audio_lr.pkl")
    audio_svm_path = os.path.join(MODEL_DIR, "audio_svm.pkl")
    
    if os.path.exists(audio_tfidf_path) and os.path.exists(audio_scaler_path):
        try:
            with open(audio_tfidf_path, "rb") as f:
                vec = pickle.load(f)
            with open(audio_scaler_path, "rb") as f:
                scal = pickle.load(f)
            
            if os.path.exists(audio_lr_path):
                with open(audio_lr_path, "rb") as f:
                    lr = pickle.load(f)
                models['lr'] = (vec, scal, lr)
            
            if os.path.exists(audio_svm_path):
                with open(audio_svm_path, "rb") as f:
                    svm = pickle.load(f)
                models['svm'] = (vec, scal, svm)
        except Exception as e:
            print(f"Could not load audio LR/SVM: {e}")
    
    # Load BERT
    audio_bert_path = os.path.join(MODEL_DIR, "audio_bert.pt")
    audio_bert_tokenizer_path = os.path.join(MODEL_DIR, "audio_bert_tokenizer.json")
    if os.path.exists(audio_bert_path) and os.path.exists(audio_bert_tokenizer_path):
        try:
            device, _ = _get_device(prefer_cuda=True)
            tokenizer = SimpleTokenizer.load(audio_bert_tokenizer_path)
            checkpoint = torch.load(audio_bert_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model_config = checkpoint.get('model_config', {})
                saved_vocab_size = checkpoint.get('vocab_size', model_config.get('vocab_size', tokenizer.vocab_size))
                model_config['vocab_size'] = saved_vocab_size
                model_config['max_length'] = checkpoint.get('max_length', 128)
                model = BERTDepressionModel(**model_config).to(device)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                vocab_size = tokenizer.vocab_size
                model = BERTDepressionModel(vocab_size=vocab_size, max_length=128).to(device)
                if isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models['bert'] = (model, tokenizer, device)
        except Exception as e:
            print(f"Could not load audio BERT: {e}")
    
    _audio_models_cache = models if models else None
    return _audio_models_cache


_video_models_cache = None


def load_video_models():
    """Load all video models (LSTM, LR, SVM, BERT). Works in both Streamlit and Flask/API."""
    global _video_models_cache
    if _video_models_cache is not None:
        return _video_models_cache
    import pickle
    models = {}
    
    # Load LSTM
    video_lstm_path = os.path.join(MODEL_DIR, "video_lstm.keras")
    video_tokenizer_path = os.path.join(MODEL_DIR, "video_tokenizer.json")
    video_lstm_config_path = os.path.join(MODEL_DIR, "video_lstm_config.json")
    if os.path.exists(video_lstm_path) and os.path.exists(video_tokenizer_path):
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            with open(video_tokenizer_path, "r", encoding="utf-8") as f:
                tok = tokenizer_from_json(f.read())
            mdl = load_model(video_lstm_path)
            maxlen = 64
            if os.path.exists(video_lstm_config_path):
                with open(video_lstm_config_path, "r", encoding="utf-8") as f:
                    maxlen = json.load(f).get("maxlen", 64)
            models['lstm'] = (mdl, tok, maxlen)
        except Exception as e:
            print(f"Could not load video LSTM: {e}")
    
    # Load LR and SVM (share TF-IDF vectorizer)
    video_tfidf_path = os.path.join(MODEL_DIR, "video_tfidf.pkl")
    video_scaler_path = os.path.join(MODEL_DIR, "video_scaler.pkl")
    video_lr_path = os.path.join(MODEL_DIR, "video_lr.pkl")
    video_svm_path = os.path.join(MODEL_DIR, "video_svm.pkl")
    
    if os.path.exists(video_tfidf_path) and os.path.exists(video_scaler_path):
        try:
            with open(video_tfidf_path, "rb") as f:
                vec = pickle.load(f)
            with open(video_scaler_path, "rb") as f:
                scal = pickle.load(f)
            
            if os.path.exists(video_lr_path):
                with open(video_lr_path, "rb") as f:
                    lr = pickle.load(f)
                models['lr'] = (vec, scal, lr)
            
            if os.path.exists(video_svm_path):
                with open(video_svm_path, "rb") as f:
                    svm = pickle.load(f)
                models['svm'] = (vec, scal, svm)
        except Exception as e:
            print(f"Could not load video LR/SVM: {e}")
    
    # Load BERT
    video_bert_path = os.path.join(MODEL_DIR, "video_bert.pt")
    video_bert_tokenizer_path = os.path.join(MODEL_DIR, "video_bert_tokenizer.json")
    if os.path.exists(video_bert_path) and os.path.exists(video_bert_tokenizer_path):
        try:
            device, _ = _get_device(prefer_cuda=True)
            tokenizer = SimpleTokenizer.load(video_bert_tokenizer_path)
            checkpoint = torch.load(video_bert_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model_config = checkpoint.get('model_config', {})
                saved_vocab_size = checkpoint.get('vocab_size', model_config.get('vocab_size', tokenizer.vocab_size))
                model_config['vocab_size'] = saved_vocab_size
                model_config['max_length'] = checkpoint.get('max_length', 128)
                model = BERTDepressionModel(**model_config).to(device)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                vocab_size = tokenizer.vocab_size
                model = BERTDepressionModel(vocab_size=vocab_size, max_length=128).to(device)
                if isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models['bert'] = (model, tokenizer, device)
        except Exception as e:
            print(f"Could not load video BERT: {e}")
    
    _video_models_cache = models if models else None
    return _video_models_cache


def analyze_speech_all_models(text_clean, models_dict):
    """Analyze speech (audio/video transcript) using all available models. Returns dict with results."""
    if not models_dict or not text_clean:
        return None
    
    results = {
        'lstm': {'prob': None, 'uncertainty': None},
        'lr': {'prob': None, 'uncertainty': None},
        'svm': {'prob': None, 'uncertainty': None},
        'bert': {'prob': None, 'uncertainty': None},
        'combined': {'avg': None, 'avg_uncertainty': None}
    }
    
    # LSTM
    if 'lstm' in models_dict:
        try:
            model, tokenizer, maxlen = models_dict['lstm']
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([text_clean])
            X = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
            prob = float(np.clip(np.squeeze(model.predict(X, verbose=0)), 0.0, 1.0))
            results['lstm']['prob'] = prob
            results['lstm']['uncertainty'] = probability_uncertainty(prob)
        except Exception as e:
            print(f"LSTM error: {e}")
    
    # LR
    if 'lr' in models_dict:
        try:
            vectorizer, scaler, lr_model = models_dict['lr']
            X = vectorizer.transform([text_clean])
            X = scaler.transform(X.toarray())
            proba = lr_model.predict_proba(X)[0]
            prob = float(proba[1] if len(proba) >= 2 else proba[0])
            results['lr']['prob'] = prob
            results['lr']['uncertainty'] = probability_uncertainty(prob)
        except Exception as e:
            print(f"LR error: {e}")
    
    # SVM
    if 'svm' in models_dict:
        try:
            vectorizer, scaler, svm_model = models_dict['svm']
            X = vectorizer.transform([text_clean])
            X = scaler.transform(X.toarray())
            proba = svm_model.predict_proba(X)[0]
            prob = float(proba[1] if len(proba) >= 2 else proba[0])
            results['svm']['prob'] = prob
            results['svm']['uncertainty'] = probability_uncertainty(prob)
        except Exception as e:
            print(f"SVM error: {e}")
    
    # BERT
    if 'bert' in models_dict:
        try:
            bert_model, bert_tokenizer, bert_device = models_dict['bert']
            prob, uncertainty = predict_proba_bert(bert_model, bert_tokenizer, bert_device, text_clean)
            results['bert']['prob'] = prob
            results['bert']['uncertainty'] = uncertainty
        except Exception as e:
            print(f"BERT error: {e}")
    
    # Combine
    probs = [r['prob'] for r in results.values() if isinstance(r, dict) and r.get('prob') is not None]
    if probs:
        results['combined']['avg'] = sum(probs) / len(probs)
        results['combined']['avg_uncertainty'] = probability_uncertainty(results['combined']['avg'])
    
    return results if probs else None


def _speech_results_to_text_format(speech_results):
    """Convert analyze_speech_all_models output to same shape as analyze_texts_all_models (single text)."""
    if not speech_results or speech_results.get('combined', {}).get('avg') is None:
        return None
    template = {'probs': [], 'uncertainties': [], 'avg': None, 'avg_uncertainty': None, 'risk_level': None}
    results = {
        'bert': dict(template), 'lr': dict(template), 'svm': dict(template), 'lstm': dict(template),
        'combined': {'avg': None, 'avg_uncertainty': None, 'risk_level': None, 'model_disagreement': None}
    }
    available_avgs = []
    for key in ('lstm', 'lr', 'svm', 'bert'):
        r = speech_results.get(key, {})
        p = r.get('prob')
        if p is not None:
            u = r.get('uncertainty') or probability_uncertainty(p)
            results[key]['probs'] = [p]
            results[key]['uncertainties'] = [u]
            results[key]['avg'] = p
            results[key]['avg_uncertainty'] = u
            results[key]['risk_level'] = get_depression_label(p)
            available_avgs.append(p)
    if available_avgs:
        results['combined']['avg'] = speech_results['combined']['avg']
        results['combined']['avg_uncertainty'] = speech_results['combined'].get('avg_uncertainty') or probability_uncertainty(results['combined']['avg'])
        results['combined']['risk_level'] = get_depression_label(results['combined']['avg'])
        results['combined']['model_disagreement'] = float(torch.tensor(available_avgs).std(unbiased=False).item())
    return results


def _log_warning(msg):
    """Log warning in both Streamlit and Flask/API context."""
    try:
        st.warning(msg)
    except Exception:
        print(f"Warning: {msg}")


def predict_from_audio_path(audio_path: str, bert_model=None, tokenizer=None, device=None,
                            vectorizer=None, lr_model=None, svm_model=None,
                            lstm_model=None, lstm_tokenizer=None, lstm_maxlen=128,
                            audio_models=None):
    """
    Use pre-trained speech-emotion model for audio-only depression-related detection.
    Audio is analyzed directly (no transcription); emotion labels are mapped to depression risk.
    """
    try:
        from pretrained_audio_video import predict_depression_from_audio
        return predict_depression_from_audio(audio_path=audio_path)
    except ValueError:
        raise
    except Exception as e:
        _log_warning(f"Audio error: {e}")
        raise RuntimeError(f"Audio prediction failed: {e}") from e


def predict_from_video_path(video_path: str, bert_model=None, tokenizer=None, device=None,
                            vectorizer=None, lr_model=None, svm_model=None,
                            lstm_model=None, lstm_tokenizer=None, lstm_maxlen=128,
                            video_models=None):
    """
    Extract audio from video, then use the same pre-trained speech-emotion model as for audio.
    Video is analyzed via its audio track only (no transcription).
    """
    try:
        from video_utils import extract_audio_with_reason
        from pretrained_audio_video import predict_depression_from_audio
        if not os.path.isfile(video_path):
            raise ValueError("Video file not found.")
        wav_path, extract_err = extract_audio_with_reason(video_path, 16000)
        if wav_path is None:
            msg = extract_err or "Could not extract audio from video."
            if "ffmpeg not found" in msg or "not found" in msg.lower():
                msg += " Install ffmpeg from https://ffmpeg.org/download.html and add it to your PATH."
            raise ValueError(msg)
        try:
            return predict_depression_from_audio(audio_path=wav_path)
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
    except ValueError:
        raise
    except Exception as e:
        _log_warning(f"Video error: {e}")
        raise RuntimeError(f"Video prediction failed: {e}") from e


def predict_proba_bert(model, tokenizer, device, text, max_length=None):
    """
    Get depression probability and uncertainty using local transformer.
    Returns: (probability, uncertainty) where probability is P(depressed=1)
    The BERT model outputs sigmoid values directly representing P(depressed).
    """
    try:
        # Get model configuration - use model's vocab_size, not tokenizer's
        model_max_length = model.model_config.get("max_length", BERT_MAX_LENGTH)
        model_vocab_size = model.model_config.get('vocab_size')
        
        if model_vocab_size is None:
            raise ValueError("Model vocab_size not found in model_config")
        
        if max_length is None:
            max_length = model_max_length
        else:
            max_length = min(max_length, model_max_length)
        
        # Tokenize text - tokenizer may have different vocab_size
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # CRITICAL: Map tokenizer token IDs to model vocab_size range
        # If tokenizer has larger vocab, map unknown tokens to UNK
        # If tokenizer has smaller vocab, this should be fine
        tokenizer_vocab_size = tokenizer.vocab_size
        
        if tokenizer_vocab_size > model_vocab_size:
            # Tokenizer has more tokens - map out-of-range tokens to UNK
            unk_token_id = tokenizer.vocab.get(tokenizer.UNK_TOKEN, 1)
            # Clamp to model vocab_size, but preserve special tokens
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
        
        # Ensure attention_mask matches input_ids shape
        if attention_mask.shape != input_ids.shape:
            attention_mask = attention_mask[:, :input_ids.shape[1]]
        
        # Predict with no_grad to save memory
        model.eval()
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        # Extract probability safely
        if output.dim() > 1:
            probability = float(np.clip(output.squeeze().cpu().item(), 0.0, 1.0))
        else:
            probability = float(np.clip(output.cpu().item(), 0.0, 1.0))
        
        uncertainty = probability_uncertainty(probability)
        return probability, uncertainty
    except RuntimeError as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            # Fallback to CPU if CUDA error
            input_ids_cpu = input_ids.cpu() if 'input_ids' in locals() else None
            attention_mask_cpu = attention_mask.cpu() if 'attention_mask' in locals() else None
            if input_ids_cpu is not None:
                model_cpu = model.cpu()
                model_cpu.eval()
                with torch.no_grad():
                    output = model_cpu(input_ids_cpu, attention_mask_cpu)
                probability = float(np.clip(output.squeeze().item(), 0.0, 1.0))
                uncertainty = probability_uncertainty(probability)
                return probability, uncertainty
        raise RuntimeError(f"BERT prediction error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"BERT prediction error: {str(e)}")


def predict_proba_ml(vectorizer, model, text):
    """
    Get depression probability and uncertainty for text using TF-IDF + ML model.
    Returns: (probability, uncertainty) where probability is P(depressed=1)
    CRITICAL: Text is cleaned the same way as during training (lowercase, no stopwords, etc.)
    so that Logistic Regression and SVM see the same feature distribution.
    """
    try:
        # CRITICAL: Clean text the SAME way as during training
        # Training uses data_loader.clean_text(); we must match it here
        text_cleaned = clean_text_for_ml(text)
        if not text_cleaned:
            # If nothing left after cleaning, use original (e.g. "sad" -> "sad")
            text_cleaned = text.strip().lower() or text
        # Transform cleaned text to TF-IDF features
        X = vectorizer.transform([text_cleaned])
        
        # Ensure X is in the right format (numpy array)
        if hasattr(X, 'toarray'):
            # Convert sparse matrix to dense array
            X = X.toarray()
        X = np.asarray(X)
        
        # Check if model has predict_proba method (works for both wrapper classes and sklearn models)
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                
                # Convert to numpy array if needed
                proba = np.asarray(proba)
                
                # Handle different output shapes
                if proba.ndim == 2:
                    # Standard format: [P(class=0), P(class=1)] or shape (1, 2)
                    # We want P(class=1) = P(depressed)
                    if proba.shape[1] >= 2:
                        probability = float(proba[0, 1])
                    elif proba.shape[1] == 1:
                        # Single column - assume it's P(class=1)
                        probability = float(np.clip(proba[0, 0], 0.0, 1.0))
                    else:
                        raise ValueError(f"Unexpected proba shape: {proba.shape}")
                elif proba.ndim == 1:
                    # 1D array - check length
                    if len(proba) >= 2:
                        probability = float(np.clip(proba[1], 0.0, 1.0))
                    else:
                        probability = float(np.clip(proba[0], 0.0, 1.0))
                else:
                    raise ValueError(f"Unexpected proba dimensions: {proba.ndim}")
                
                # Ensure probability is valid
                probability = float(np.clip(probability, 0.0, 1.0))
                return probability, probability_uncertainty(probability)
                
            except Exception as e:
                # If predict_proba fails, try direct sklearn model access
                if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                    proba = model.model.predict_proba(X)
                    proba = np.asarray(proba)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        probability = float(np.clip(proba[0, 1], 0.0, 1.0))
                    else:
                        raise ValueError(f"Unexpected proba format: {proba.shape}")
                    return probability, probability_uncertainty(probability)
                else:
                    raise RuntimeError(f"predict_proba failed: {e}")
        else:
            # Fallback: use predict and convert to probability estimate
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                # Handle array or scalar
                if isinstance(pred, np.ndarray):
                    pred_value = int(pred[0]) if len(pred) > 0 else 0
                else:
                    pred_value = int(pred)
                
                # Convert binary prediction (0 or 1) to probability estimate
                # If prediction is 1 (depressed), return high probability (0.8)
                # If prediction is 0 (non-depressed), return low probability (0.2)
                probability = 0.8 if pred_value == 1 else 0.2
                return probability, probability_uncertainty(probability)
            else:
                raise ValueError("Model has neither predict_proba nor predict method")
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise RuntimeError(f"ML prediction error: {str(e)}\nDetails: {error_details}")


def extract_text_from_json(json_data):
    """
    Extract text content from JSON data.
    Supports multiple formats:
    - Single tweet JSON: {"text": "..."}
    - Array of tweets: [{"text": "..."}, ...]
    - User data with tweets: {"tweets": [{"text": "..."}], ...}
    - Multiple tweets in a single file: [{"text": "..."}, {"text": "..."}, ...]
    """
    texts = []
    
    if isinstance(json_data, dict):
        # Check for single tweet format
        if 'text' in json_data:
            text = json_data.get('text', '')
            if text and isinstance(text, str) and text.strip():
                texts.append(text.strip())
        
        # Check for user data with tweets array
        if 'tweets' in json_data:
            tweets = json_data['tweets']
            if isinstance(tweets, list):
                for tweet in tweets:
                    if isinstance(tweet, dict) and 'text' in tweet:
                        text = tweet.get('text', '')
                        if text and isinstance(text, str) and text.strip():
                            texts.append(text.strip())
        
        # Check for other common fields
        for key in ['content', 'post', 'message', 'body', 'full_text']:
            if key in json_data:
                text = json_data.get(key, '')
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        
        # Check if the entire dict is a list of tweet-like objects
        # Sometimes JSON files have multiple tweets as separate keys
        for key, value in json_data.items():
            if isinstance(value, dict) and 'text' in value:
                text = value.get('text', '')
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
    
    elif isinstance(json_data, list):
        # Array of tweets
        for item in json_data:
            if isinstance(item, dict):
                if 'text' in item:
                    text = item.get('text', '')
                    if text and isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                # Recursive call for nested structures
                nested_texts = extract_text_from_json(item)
                texts.extend(nested_texts)
            elif isinstance(item, str):
                # Direct string in list
                if item.strip():
                    texts.append(item.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    
    return unique_texts


def process_json_file(uploaded_file):
    """Process uploaded JSON file and extract all text content."""
    try:
        json_data = json.load(uploaded_file)
        texts = extract_text_from_json(json_data)
        return texts, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"


def find_user_json_file(username):
    """Try to find JSON file for username in common locations."""
    # Common locations to check
    search_paths = [
        os.path.join('Dataset_MDDL (1)', 'Dataset', 'labeled', 'positive', 'data', 'tweet'),
        os.path.join('Dataset_MDDL (1)', 'Dataset', 'labeled', 'negative', 'data', 'tweet'),
        'data',
        'users',
    ]
    
    for base_path in search_paths:
        if os.path.exists(base_path):
            # Look for files matching username
            for file in os.listdir(base_path):
                if file.endswith('.json') and username.lower() in file.lower():
                    return os.path.join(base_path, file)
    
    return None


def analyze_texts_all_models(texts, bert_model, tokenizer, device, vectorizer, lr_model, svm_model,
                              lstm_model=None, lstm_tokenizer=None, lstm_maxlen=128, raw_texts=None):
    """
    Analyze multiple texts using BERT, Logistic Regression, SVM, and LSTM.
    raw_texts: optional list of raw (uncleaned) texts for BERT; when provided, BERT uses clean_text_for_bert.
    Returns: Dictionary with results from each model and combined results
    """
    if not texts:
        return None
    
    results = {
        'bert': {'probs': [], 'uncertainties': [], 'avg': None, 'avg_uncertainty': None, 'risk_level': None},
        'lr': {'probs': [], 'uncertainties': [], 'avg': None, 'avg_uncertainty': None, 'risk_level': None},
        'svm': {'probs': [], 'uncertainties': [], 'avg': None, 'avg_uncertainty': None, 'risk_level': None},
        'lstm': {'probs': [], 'uncertainties': [], 'avg': None, 'avg_uncertainty': None, 'risk_level': None},
        'combined': {'avg': None, 'avg_uncertainty': None, 'risk_level': None, 'model_disagreement': None}
    }
    
    for idx, text in enumerate(texts):
        if not text or not text.strip():
            continue
        
        text_clean = text.strip()
        # BERT was trained with train_multimodal.clean_text (no stopwords); use raw text when available
        raw = raw_texts[idx] if raw_texts and idx < len(raw_texts) else None
        bert_text = clean_text_for_bert(raw) if raw and str(raw).strip() else text_clean
        
        if bert_model is not None:
            try:
                bert_prob, bert_uncertainty = predict_proba_bert(bert_model, tokenizer, device, bert_text)
                results['bert']['probs'].append(bert_prob)
                results['bert']['uncertainties'].append(bert_uncertainty)
            except Exception as e:
                _log_warning(f"BERT error for text: {str(e)}")
        
        if lr_model is not None and vectorizer is not None:
            try:
                lr_prob, lr_uncertainty = predict_proba_ml(vectorizer, lr_model, text_clean)
                results['lr']['probs'].append(lr_prob)
                results['lr']['uncertainties'].append(lr_uncertainty)
            except Exception as e:
                _log_warning(f"LR error for text: {str(e)}")
        
        if svm_model is not None and vectorizer is not None:
            try:
                svm_prob, svm_uncertainty = predict_proba_ml(vectorizer, svm_model, text_clean)
                results['svm']['probs'].append(svm_prob)
                results['svm']['uncertainties'].append(svm_uncertainty)
            except Exception as e:
                _log_warning(f"SVM error for text: {str(e)}")
        
        if lstm_model is not None and lstm_tokenizer is not None:
            try:
                lstm_prob, lstm_uncertainty = predict_proba_lstm(lstm_model, lstm_tokenizer, lstm_maxlen, text_clean)
                if lstm_prob is not None:
                    results['lstm']['probs'].append(lstm_prob)
                    results['lstm']['uncertainties'].append(lstm_uncertainty)
            except Exception as e:
                _log_warning(f"LSTM error for text: {str(e)}")
    
    if results['bert']['probs']:
        results['bert']['avg'] = sum(results['bert']['probs']) / len(results['bert']['probs'])
        results['bert']['avg_uncertainty'] = sum(results['bert']['uncertainties']) / len(results['bert']['uncertainties'])
        results['bert']['risk_level'] = get_depression_label(results['bert']['avg'])
    
    if results['lr']['probs']:
        results['lr']['avg'] = sum(results['lr']['probs']) / len(results['lr']['probs'])
        results['lr']['avg_uncertainty'] = sum(results['lr']['uncertainties']) / len(results['lr']['uncertainties'])
        results['lr']['risk_level'] = get_depression_label(results['lr']['avg'])
    
    if results['svm']['probs']:
        results['svm']['avg'] = sum(results['svm']['probs']) / len(results['svm']['probs'])
        results['svm']['avg_uncertainty'] = sum(results['svm']['uncertainties']) / len(results['svm']['uncertainties'])
        results['svm']['risk_level'] = get_depression_label(results['svm']['avg'])
    
    if results['lstm']['probs']:
        results['lstm']['avg'] = sum(results['lstm']['probs']) / len(results['lstm']['probs'])
        results['lstm']['avg_uncertainty'] = sum(results['lstm']['uncertainties']) / len(results['lstm']['uncertainties'])
        results['lstm']['risk_level'] = get_depression_label(results['lstm']['avg'])
    
    available_avgs = []
    if results['bert']['avg'] is not None:
        available_avgs.append(results['bert']['avg'])
    if results['lr']['avg'] is not None:
        available_avgs.append(results['lr']['avg'])
    if results['svm']['avg'] is not None:
        available_avgs.append(results['svm']['avg'])
    if results['lstm']['avg'] is not None:
        available_avgs.append(results['lstm']['avg'])
    
    if available_avgs:
        results['combined']['avg'] = sum(available_avgs) / len(available_avgs)
        results['combined']['avg_uncertainty'] = probability_uncertainty(results['combined']['avg'])
        results['combined']['model_disagreement'] = float(torch.tensor(available_avgs).std(unbiased=False).item())
        results['combined']['risk_level'] = get_depression_label(results['combined']['avg'])
    
    return results if available_avgs else None


def get_depression_label(probability):
    """Binary classification: Depressed or Not depressed (threshold 0.5)."""
    return "Depressed" if probability >= 0.5 else "Not depressed"


def main():
    st.markdown('<h1 class="main-header">🧠 Depression Detection from Social Media</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    This application detects **depression or not** from social media text using **BERT**, **Logistic Regression**, **SVM**, and **LSTM**.
    Output is binary: **Depressed** or **Not depressed** (no severity levels).
    """)

    has_bert = os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_TOKENIZER_PATH)
    has_vectorizer = os.path.exists(VECTORIZER_PATH)
    has_lr = os.path.exists(os.path.join(MODEL_DIR, 'lr_model.pkl'))
    has_svm = os.path.exists(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    has_lstm = os.path.exists(TEXT_LSTM_PATH) and os.path.exists(TEXT_TOKENIZER_PATH)
    
    if not has_bert and not has_vectorizer and not has_lstm:
        st.error("No models found! Please train first using:")
        st.code("python train_models.py --dataset_path \"Dataset_MDDL (1)/Dataset\"", language="bash")
        return

    # Load all available models
    bert_model, tokenizer, device, device_label = None, None, None, None
    vectorizer = None
    lr_model = None
    svm_model = None
    lstm_model, lstm_tokenizer, lstm_maxlen = None, None, 128
    
    # Load BERT
    if has_bert:
        try:
            with st.spinner("Loading BERT model..."):
                bert_model, tokenizer, device, device_label = load_bert_model()
        except Exception as e:
            st.warning(f"Could not load BERT: {e}")
    else:
        st.info("BERT model not found (optional)")
    
    # Load vectorizer and ML models
    if has_vectorizer:
        try:
            vectorizer = load_vectorizer()
            
            # Load Logistic Regression (custom)
            if has_lr:
                try:
                    lr_model = load_lr_model()
                except Exception as e:
                    st.warning(f"Could not load LR: {e}")
            
            # Load SVM
            if has_svm:
                try:
                    svm_model = load_svm_model()
                except Exception as e:
                    st.warning(f"Could not load SVM: {e}")
        except Exception as e:
            st.error(f"Error loading vectorizer: {e}")
            return
    
    # Load Text LSTM
    if has_lstm:
        try:
            with st.spinner("Loading LSTM model..."):
                lstm_model, lstm_tokenizer, lstm_maxlen = load_text_lstm()
            if lstm_model is None:
                st.warning("Could not load LSTM (files missing or invalid). Run: python train_multimodal.py --output_dir models")
        except Exception as e:
            st.warning(f"Could not load LSTM: {e}")
    
    # Load modality-specific models (optional; used for audio/video when available)
    audio_models = None
    video_models = None
    if os.path.exists(os.path.join(MODEL_DIR, "audio_tfidf.pkl")) or os.path.exists(os.path.join(MODEL_DIR, "audio_bert.pt")):
        try:
            audio_models = load_audio_models()
        except Exception as e:
            st.warning(f"Could not load audio models: {e}")
    if os.path.exists(os.path.join(MODEL_DIR, "video_tfidf.pkl")) or os.path.exists(os.path.join(MODEL_DIR, "video_bert.pt")):
        try:
            video_models = load_video_models()
        except Exception as e:
            st.warning(f"Could not load video models: {e}")
    
    # Check if at least one model is loaded
    if bert_model is None and lr_model is None and svm_model is None and lstm_model is None:
        st.error("No models could be loaded! Please train models first.")
        return

    st.sidebar.header("Model Information")
    loaded_models = []
    if bert_model is not None:
        loaded_models.append("BERT")
    if lr_model is not None:
        loaded_models.append("Logistic Regression")
    if svm_model is not None:
        loaded_models.append("SVM")
    if lstm_model is not None:
        loaded_models.append("LSTM")
    
    sidebar_info = f"""
    **Loaded Models:** {', '.join(loaded_models) if loaded_models else 'None'}  
    **Evaluation:** Cross-validation + test set (see results below).
    """
    sidebar_info += "\n**Audio / Video:** Speech analysis."
    if bert_model is not None and device_label:
        sidebar_info += f"\n**Compute:** BERT runs on **{device_label}**."
    st.sidebar.info(sidebar_info)
    st.sidebar.header("Disclaimer")
    st.sidebar.warning("""
    This tool is for research only and is not a substitute
    for professional mental health diagnosis.
    """)

    # Load comparison results if available (supports both cv_results/test_results and legacy format)
    results_path = os.path.join(MODEL_DIR, 'model_comparison_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
        st.sidebar.subheader("Test set metrics")
        test_results = data.get('test_results', data) if isinstance(data, dict) else {}
        for name, m in test_results.items():
            if isinstance(m, dict) and 'accuracy' in m:
                st.sidebar.text(f"{name}: Acc={m['accuracy']:.2%} F1={m['f1_score']:.2%}")
        if 'cv_results' in data:
            st.sidebar.subheader("CV (mean ± std)")
            for name, m in data['cv_results'].items():
                acc = m.get('accuracy', [0, 0])
                f1 = m.get('f1_score', [0, 0])
                if isinstance(acc, list) and len(acc) >= 2:
                    st.sidebar.text(f"{name}: Acc={acc[0]:.2%}±{acc[1]:.2%} F1={f1[0]:.2%}±{f1[1]:.2%}")

    st.header("Input Method")
    
    input_method = st.radio(
        "Choose input method:",
        ["Username", "JSON File Upload"],
        horizontal=True
    )
    
    texts_to_analyze = []
    user_identifier = None
    
    if input_method == "Username":
        username = st.text_input(
            "Enter username:",
            placeholder="e.g., user123"
        )
        
        if username:
            user_identifier = username
            # Try to find JSON file for this username
            json_file_path = find_user_json_file(username)
            
            if json_file_path and os.path.exists(json_file_path):
                st.info(f"Found JSON file: {os.path.basename(json_file_path)}")
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    texts_to_analyze = extract_text_from_json(json_data)
                    if texts_to_analyze:
                        st.success(f"Extracted {len(texts_to_analyze)} text(s) from file.")
                    else:
                        st.warning("No text content found in the JSON file.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            else:
                st.info("Upload a JSON file below or provide text directly.")
                # Allow manual JSON upload even when username is provided
                uploaded_file = st.file_uploader(
                    "Upload JSON file for this user:",
                    type=['json'],
                    key="username_json"
                )
                if uploaded_file:
                    texts, error = process_json_file(uploaded_file)
                    if error:
                        st.error(error)
                    elif texts:
                        texts_to_analyze = texts
                        st.success(f"Extracted {len(texts)} text(s) from uploaded file.")
    
    else:  # JSON File Upload
        uploaded_file = st.file_uploader(
            "Upload JSON file containing user data/tweets:",
            type=['json'],
            help="Upload a JSON file containing tweet data. Supported formats: single tweet, array of tweets, or user data with tweets."
        )
        
        if uploaded_file:
            texts, error = process_json_file(uploaded_file)
            if error:
                st.error(error)
            elif texts:
                texts_to_analyze = texts
                user_identifier = uploaded_file.name
                st.success(f"Extracted {len(texts)} text(s) from file.")
            else:
                st.warning("No text content found in the JSON file.")
    
    # Also allow direct text input as fallback
    st.subheader("Or Enter Text Directly")
    direct_text = st.text_area(
        "Enter tweet or post text directly:",
        height=100,
        placeholder="Enter text content here..."
    )
    
    if direct_text.strip():
        texts_to_analyze.append(direct_text.strip())
        if not user_identifier:
            user_identifier = "Direct Input"

    # Audio/video use the same text models after transcription
    # Check if text models are available (required for audio/video)
    has_text_models = (bert_model is not None) or (lr_model is not None) or (svm_model is not None) or (lstm_model is not None)

    st.subheader("Audio and Video Input")
    audio_uploaded = None
    video_uploaded = None
    audio_uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3"], key="audio_upload")
    video_uploaded = st.file_uploader("Upload video file (audio will be extracted and analyzed)", type=["mp4", "avi", "mov"], key="video_upload")

    if st.button("Analyze Depression Risk", type="primary"):
        has_input = bool(texts_to_analyze) or (audio_uploaded is not None) or (video_uploaded is not None)
        if not has_input:
            st.warning("Please provide text (username/JSON/direct), and/or upload audio, and/or upload video.")
        else:
            modality_probs = []
            results = None

            if texts_to_analyze:
                with st.spinner(f"Analyzing {len(texts_to_analyze)} text(s)..."):
                    results = analyze_texts_all_models(
                        texts_to_analyze, bert_model, tokenizer, device, vectorizer, lr_model, svm_model,
                        lstm_model=lstm_model, lstm_tokenizer=lstm_tokenizer, lstm_maxlen=lstm_maxlen
                    )
                if results is not None:
                    modality_probs.append(("Text", results["combined"]["avg"]))

            audio_results = None
            if audio_uploaded is not None:
                with st.spinner("Analyzing audio..."):
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_uploaded.name)[1] or ".wav") as tmp:
                        tmp.write(audio_uploaded.getvalue())
                        tmp_path = tmp.name
                    try:
                        audio_results = predict_from_audio_path(
                            tmp_path, bert_model, tokenizer, device, vectorizer, lr_model, svm_model,
                            lstm_model=lstm_model, lstm_tokenizer=lstm_tokenizer, lstm_maxlen=lstm_maxlen,
                            audio_models=audio_models
                        )
                        if audio_results and audio_results.get('combined', {}).get('avg') is not None:
                            modality_probs.append(("Audio", audio_results['combined']['avg']))
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            video_results = None
            if video_uploaded is not None:
                with st.spinner("Extracting audio from video and analyzing..."):
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_uploaded.name)[1] or ".mp4") as tmp:
                        tmp.write(video_uploaded.getvalue())
                        tmp_path = tmp.name
                    try:
                        video_results = predict_from_video_path(
                            tmp_path, bert_model, tokenizer, device, vectorizer, lr_model, svm_model,
                            lstm_model=lstm_model, lstm_tokenizer=lstm_tokenizer, lstm_maxlen=lstm_maxlen,
                            video_models=video_models
                        )
                        if video_results and video_results.get('combined', {}).get('avg') is not None:
                            modality_probs.append(("Video", video_results['combined']['avg']))
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            display_results = results or audio_results or video_results
            display_label = "Text" if results is not None else ("Audio" if audio_results is not None else "Video")

            if not modality_probs:
                st.error("No modality could be analyzed. Check your inputs and that models are trained.")
            else:
                # Combine all modality predictions
                combined_avg = sum(p for _, p in modality_probs) / len(modality_probs)
                combined_label = get_depression_label(combined_avg)
                st.header("Multi-Modal Prediction Results")

                box_class = "depressed" if combined_avg >= 0.5 else "not-depressed"
                mod_list = ", ".join(m for m, _ in modality_probs)
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>Result: {combined_label}</h2>
                    <h3>Multi-Modal Combined ({mod_list}): <strong>{combined_label}</strong></h3>
                    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 1rem 0;">
                    <p><strong>Combined probability:</strong> {combined_avg:.1%}</p>
                    <p><strong>User:</strong> {user_identifier or 'Upload'}</p>
                    <p><strong>Modalities:</strong> {mod_list}</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(combined_avg)

                # Show per-modality results
                if len(modality_probs) > 1:
                    st.subheader("Per-Modality Results")
                    for name, p in modality_probs:
                        st.metric(name, f"{p:.2%}", get_depression_label(p))
                
            if display_results is not None:
                # Show individual model breakdown only for text input; hide for audio/video
                if display_label == "Text":
                    st.subheader(f"Results from Individual Models ({display_label})")
                    cols = st.columns(3)
                    
                    model_results = []
                    if display_results['bert']['avg'] is not None:
                        model_results.append(('BERT', display_results['bert']['avg'], display_results['bert']['avg_uncertainty'], display_results['bert']['risk_level']))
                    if display_results['lr']['avg'] is not None:
                        model_results.append(('Logistic Regression', display_results['lr']['avg'], display_results['lr']['avg_uncertainty'], display_results['lr']['risk_level']))
                    if display_results['svm']['avg'] is not None:
                        model_results.append(('SVM', display_results['svm']['avg'], display_results['svm']['avg_uncertainty'], display_results['svm']['risk_level']))
                    if display_results.get('lstm') and display_results['lstm']['avg'] is not None:
                        model_results.append(('LSTM', display_results['lstm']['avg'], display_results['lstm']['avg_uncertainty'], display_results['lstm']['risk_level']))
                    
                    for idx, (model_name, prob, uncertainty, label) in enumerate(model_results):
                        with cols[idx % 3]:
                            box_class = "depressed" if prob >= 0.5 else "not-depressed"
                            confidence = (1 - uncertainty) * 100
                            st.markdown(f"""
                            <div class="prediction-box {box_class}" style="padding: 1rem; margin: 0.5rem 0;">
                                <h3>{model_name}</h3>
                                <h4>{label}</h4>
                                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display individual probabilities by model (for multiple texts only)
                    num_probs = max(
                        len(display_results['bert']['probs']) if display_results['bert']['probs'] else 0,
                        len(display_results['lr']['probs']) if display_results['lr']['probs'] else 0,
                        len(display_results['svm']['probs']) if display_results['svm']['probs'] else 0,
                        len(display_results.get('lstm', {}).get('probs') or [])
                    )
                    if num_probs > 1:
                        st.subheader("Individual Probabilities by Model")
                        prob_data = {'Item': range(1, num_probs + 1)}
                        if display_results['bert']['probs']:
                            prob_data['BERT'] = [f"{p:.2%}" for p in display_results['bert']['probs']]
                        if display_results['lr']['probs']:
                            prob_data['Logistic Regression'] = [f"{p:.2%}" for p in display_results['lr']['probs']]
                        if display_results['svm']['probs']:
                            prob_data['SVM'] = [f"{p:.2%}" for p in display_results['svm']['probs']]
                        if display_results.get('lstm', {}).get('probs'):
                            prob_data['LSTM'] = [f"{p:.2%}" for p in display_results['lstm']['probs']]
                        for model_key, col_name in [('bert', 'BERT result'), ('lr', 'LR result'), ('svm', 'SVM result'), ('lstm', 'LSTM result')]:
                            if display_results.get(model_key, {}).get('probs'):
                                prob_data[col_name] = [
                                    "Depressed" if p >= 0.5 else "Not depressed"
                                    for p in display_results[model_key]['probs']
                                ]
                        st.dataframe(prob_data, use_container_width=True)
                        st.subheader("Statistics")
                        stat_cols = st.columns(len(model_results))
                        for idx, (model_name, prob, _, _) in enumerate(model_results):
                            with stat_cols[idx]:
                                model_key = {'BERT': 'bert', 'Logistic Regression': 'lr', 'SVM': 'svm', 'LSTM': 'lstm'}.get(model_name, model_name.lower())
                                probs = display_results.get(model_key, {}).get('probs') or []
                                if probs:
                                    st.metric(model_name, f"{prob:.2%}", delta=f"Range: {min(probs):.2%} - {max(probs):.2%}")
                                else:
                                    st.metric(model_name, f"{prob:.2%}", None)
                
                # Display interpretation (for all modalities: text, audio, video)
                st.subheader("Interpretation")
                interp_prob = display_results.get('combined', {}).get('avg')
                if interp_prob is not None:
                    if interp_prob >= 0.5:
                        st.warning("**Depressed**: The combined models classify this content as indicating depression. This is not a medical diagnosis; consider professional support if needed.")
                    else:
                        st.success("**Not depressed**: The combined models classify this content as not indicating depression. This is not a medical diagnosis.")
                
                # Show sample texts if available (for text modality)
                if texts_to_analyze and len(texts_to_analyze) > 0 and display_results is not None:
                    with st.expander("View Analyzed Texts"):
                        for idx, text in enumerate(texts_to_analyze[:10], 1):
                            text_info = f"Text {idx}: {text[:200]}"
                            if idx <= len(display_results.get('bert', {}).get('probs', [])):
                                text_info += f"\n  → BERT: {display_results['bert']['probs'][idx-1]:.2%}"
                            if idx <= len(display_results.get('lr', {}).get('probs', [])):
                                text_info += f" | LR: {display_results['lr']['probs'][idx-1]:.2%}"
                            if idx <= len(display_results.get('svm', {}).get('probs', [])):
                                text_info += f" | SVM: {display_results['svm']['probs'][idx-1]:.2%}"
                            if display_results.get('lstm', {}).get('probs') and idx <= len(display_results['lstm']['probs']):
                                text_info += f" | LSTM: {display_results['lstm']['probs'][idx-1]:.2%}"
                            st.text(text_info)
                        if len(texts_to_analyze) > 10:
                            st.info(f"... and {len(texts_to_analyze) - 10} more texts")


if __name__ == '__main__':
    main()
