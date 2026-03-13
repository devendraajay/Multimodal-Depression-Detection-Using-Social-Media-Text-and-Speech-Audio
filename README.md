<<<<<<< HEAD

  # Build Website from Instructions

  This is a code bundle for Build Website from Instructions. The original project is available at https://www.figma.com/design/gp1840kcSg3sOn3x5Cct0n/Build-Website-from-Instructions.

  ## Running the code

  Run `npm i` to install the dependencies.

  Run `npm run dev` to start the development server.
  
=======
# Multimodal Depression Detection Using Social Media Text and Speech Audio

This repository contains the code for a multimodal depression detection system that leverages **social media text**, **timeline-based behavioral features**, and **speech audio** to estimate the likelihood that a user is experiencing depression.

The project combines modern NLP (BERT, Whisper) with classical machine learning (TF–IDF + SVM) and deep learning (LSTM) in a modular, extensible pipeline.

---

## ✨ Key Features

- **Social media text modeling with BERT**
  - Uses a pretrained `bert-base-uncased` encoder.
  - Fine-tunes on labeled depression / control users.

- **Timeline behavioral features**
  - Posting frequency and gaps.
  - Number of posts.
  - Simple sentiment trend using negative word lists.
  - Fused with BERT embeddings via a fully-connected classifier.

- **Speech audio pipeline**
  - Automatically discovers `normal` and `depressed` audio folders.
  - Transcribes audio using **Whisper** (`openai/whisper-base`).
  - Trains:
    - An **LSTM** classifier on transcripts.
    - A **TF–IDF + SVM** baseline.

- **Modular design**
  - Text + timeline model and audio pipeline can be used independently.
  - Easy to extend with new datasets or fusion strategies.

---

## 📂 Project Structure

> This is approximate; adapt to your actual files.

- `config.py` – Paths and hyperparameters (datasets, model configs, training settings).
- `model.py` – PyTorch BERT-based multimodal model (text + timeline).
- `train.py` / `evaluate.py` – Training and evaluation for the text/timeline model.
- `data_loader_audio.py` – Audio dataset loader and directory scanning logic.
- `train_audio.py` – Whisper transcription + LSTM + TF–IDF + SVM pipeline for audio.
- `evaluate_models.py` / `inference.py` – Optional scripts for testing/inference.
- `pretrained_audio_video.py`, `train_multimodal.py`, etc. – Additional utilities / experiments.
- `models/` – (Ignored in git) Saved models/checkpoints.
- `Multimodel_Dataset/`, `Dataset_MDDL (1)/` – (Ignored in git) Dataset folders.

---

## 🧪 Datasets

### 1. Social Media Depression Dataset

The text + timeline model expects a dataset organized similarly to:

- `Dataset_MDDL (1)/Dataset/labeled/positive/data/tweet`
- `Dataset_MDDL (1)/Dataset/labeled/negative/data/tweet`
- `Dataset_MDDL (1)/Dataset/labeled/positive/data/timeline`
- `Dataset_MDDL (1)/Dataset/labeled/negative/data/timeline`

> **Note:** The actual depression dataset is **not** included in this repository.  
> Please obtain it from the original source (or a similar public dataset) and place it according to your `config.py`.

### 2. Audio Depression Dataset

The audio pipeline expects an `Audio_Dataset` root with `normal` and `depressed` folders. Examples of supported layouts:

```text
Multimodel_Dataset/
  Audio_Dataset/
    normal/
      *.wav
    depressed/
      *.wav
>>>>>>> b90162ed008f262c3ca64cbe723c32e3555fbec3
