# Multimodal Depression Detection Using Social Media Text and Speech Audio

> ⚠️ **Ethical Notice:** This project is for **research and educational purposes only**. It is not a medical device and must not be used for self-diagnosis or clinical decisions. Any real-world deployment requires informed consent, proper anonymization, and oversight by qualified mental health professionals.

---

## Overview

This project implements a **multimodal depression detection system** that fuses signals from three sources:

- **Social media text** — tweets and posts processed with BERT
- **Timeline behavioral features** — posting frequency, temporal gaps, sentiment trend
- **Speech audio** — transcribed via Whisper and classified with LSTM or TF-IDF + SVM

A unified **Streamlit web app** ties everything together into an interactive demo interface.

---

## Features

### Text + Timeline (Social Media)
- `bert-base-uncased` for deep contextual text embeddings
- Lightweight, interpretable timeline features (posting frequency, temporal gaps, sentiment trend)
- Late fusion of text embeddings and behavioral features for final classification

### Speech Audio
- Automatic directory scanning for `normal` and `depressed` speaker audio
- Transcription via **OpenAI Whisper** (`openai/whisper-base`)
- Two downstream classifiers trained on transcripts:
  - LSTM classifier (sequence-aware)
  - TF-IDF + SVM classifier (fast, interpretable baseline)

### Streamlit App
- Paste or type social media text → get depression probability
- Upload `.wav` / `.mp3` audio → transcription + prediction pipeline
- Unified interface for both modalities

---

## Project Structure

```
.
├── app.py / run_app.py          # Streamlit app entry point
├── config.py                    # Dataset paths, model settings, hyperparameters
├── model.py                     # PyTorch BERT-based model (text + timeline)
├── model_bert.py                # Additional BERT utilities (if present)
├── data_loader.py               # Social media data loading and feature extraction
├── data_loader_audio.py         # Audio dataset loader and labeling
├── data_loader_video.py         # Video pipeline (if used)
├── video_utils.py               # Video utilities (if used)
├── train.py / train_multimodal.py  # Training: text + timeline model
├── train_audio.py               # Training: audio pipeline (Whisper + LSTM + SVM)
├── evaluate.py / evaluate_models.py  # Evaluation scripts
├── predict_depression.py        # Inference utilities
├── inference.py                 # Inference utilities
├── models/                      # Saved model weights (not committed to Git)
├── Multimodel_Dataset/          # Audio dataset folder (not in Git)
└── Dataset_MDDL (1)/            # Social media dataset folder (not in Git)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/devendraajay/Multimodal-Depression-Detection-Using-Social-Media-Text-and-Speech-Audio.git
cd Multimodal-Depression-Detection-Using-Social-Media-Text-and-Speech-Audio
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should include at minimum:

```
streamlit
torch
transformers
tensorflow
scikit-learn
pandas
numpy
librosa
tqdm
```

---

## Datasets

Datasets are **not included** in this repository. Download them separately and update paths in `config.py`.

### Social Media Dataset

Expected directory structure:

```
Dataset_MDDL (1)/
└── Dataset/
    └── labeled/
        ├── positive/
        │   └── data/
        │       ├── tweet/
        │       └── timeline/
        └── negative/
            └── data/
                ├── tweet/
                └── timeline/
```

### Audio Dataset

Expected directory structure:

```
Multimodel_Dataset/
└── Audio_Dataset/
    ├── normal/
    │   └── *.wav
    └── depressed/
        └── *.wav
```

> See `data_loader_audio.py` for supported nested directory patterns.

---

## Usage

### Train Models (Optional but Recommended)

#### Audio Models

```bash
python train_audio.py --dataset_path Multimodel_Dataset --output_dir models
```

This will:
1. Scan audio files under `Multimodel_Dataset/`
2. Transcribe them using Whisper
3. Train LSTM and TF-IDF + SVM classifiers on the transcripts
4. Save model weights and vectorizers to `models/`

#### Text + Timeline Model

```bash
python train.py
# or
python train_multimodal.py
```

This will:
1. Load social media text and timeline data
2. Compute BERT embeddings and timeline features
3. Train the multimodal classifier and save weights to `models/`

> Make sure dataset paths in `config.py` are correctly set before training.

### Run the Streamlit App

```bash
streamlit run app.py
# or, if your entry file differs:
streamlit run run_app.py
```

Open the URL shown in the terminal (default: [http://localhost:8501](http://localhost:8501)).

**Text tab:** Paste social media text → click predict → view depression probability.  
**Audio tab:** Upload a `.wav` / `.mp3` file → transcription + prediction → view result.

---

## Results

> Replace the placeholders below with your actual evaluation numbers.

| Modality | Model | Accuracy | F1 (Depressed) |
|---|---|---|---|
| Text + Timeline | BERT + MLP | XX.X% | YY.Y |
| Audio | LSTM | AA.A% | BB.B |
| Audio | TF-IDF + SVM | CC.C% | DD.D |

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — BERT and Whisper
- [Scikit-learn](https://scikit-learn.org/) — TF-IDF, SVM, evaluation utilities
- [TensorFlow / Keras](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) — model implementation
- The creators of the depression datasets used in this project

---

## License

This project is licensed under the [MIT License](LICENSE).
