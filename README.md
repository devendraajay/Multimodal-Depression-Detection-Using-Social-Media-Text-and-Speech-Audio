# Multimodal Depression Detection Using Social Media Text and Speech Audio
This project implements a **multimodal depression detection system** that uses:
- **Social media text** (tweets/posts)
- **Timeline behavioral features** (posting patterns)
- **Speech audio**
It combines modern NLP models (BERT, Whisper) with machine learning classifiers and exposes an interactive **Streamlit web app** for demonstration.
---
## ✨ Features
- **Text + Timeline (Social Media)**
  - BERT (`bert-base-uncased`) for contextual text embeddings.
  - Simple, interpretable timeline features (posting frequency, temporal gaps, sentiment trend).
  - Fusion of text and behavior for depression classification.
- **Speech Audio**
  - Automatic directory scanning for `normal` and `depressed` speakers.
  - Transcription using **Whisper** (`openai/whisper-base`).
  - Two downstream models on transcripts:
    - LSTM classifier
    - TF–IDF + SVM classifier
- **Streamlit App**
  - Simple web UI to:
    - Enter / paste text.
    - Upload audio files.
    - View predicted depression probability.
---
## 🏗️ Project Structure
> Adjust this section to match your actual files.
- `app.py` or `run_app.py` – Streamlit app entry point.
- `config.py` – Dataset paths, model settings, and hyperparameters.
- `model.py` – PyTorch BERT-based model (text + timeline).
- `model_bert.py` – (If present) Additional BERT-related utilities.
- `data_loader.py` – Social media data loading and feature extraction.
- `data_loader_audio.py` – Audio dataset loader and labeling.
- `data_loader_video.py`, `video_utils.py` – (If used) Video pipeline.
- `train.py` / `train_multimodal.py` – Training for text + timeline model.
- `train_audio.py` – Audio pipeline (Whisper + LSTM + TF–IDF + SVM).
- `evaluate.py`, `evaluate_models.py` – Evaluation scripts.
- `predict_depression.py`, `inference.py` – Inference utilities.
- `models/` – Saved model weights (usually **not** committed to Git).
- `Multimodel_Dataset/`, `Dataset_MDDL (1)/` – Dataset folders (not in Git).
---
## 📦 Installation
1. **Clone the repository**
```bash
git clone https://github.com/devendraajay/Multimodal-Depression-Detection-Using-Social-Media-Text-and-Speech-Audio.git
cd Multimodal-Depression-Detection-Using-Social-Media-Text-and-Speech-Audio
Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # PowerShell (Windows)
# or: source .venv/bin/activate  # macOS / Linux
Install dependencies
pip install -r requirements.txt
requirements.txt should include at least:

streamlit
torch
transformers
tensorflow
scikit-learn
pandas
numpy
librosa
tqdm
(Add or remove packages based on your actual code.)

📂 Datasets (Not Included)
1. Social Media Dataset
The text + timeline model expects a structure similar to:

Dataset_MDDL (1)/
  Dataset/
    labeled/
      positive/data/tweet/
      positive/data/timeline/
      negative/data/tweet/
      negative/data/timeline/
The actual depression dataset is not included in this repo.
Please download it from the original source (or use a similar public dataset) and adjust the paths in config.py.

2. Audio Dataset
The audio pipeline expects something like:

Multimodel_Dataset/
  Audio_Dataset/
    normal/*.wav
    depressed/*.wav
or a nested version (see data_loader_audio.py for supported patterns).

🚀 How to Run
1. Train Models (Optional but Recommended)
a) Train Audio Models
python train_audio.py --dataset_path Multimodel_Dataset --output_dir models
This will:

Find audio files under Multimodel_Dataset.
Transcribe them using Whisper.
Train LSTM and TF–IDF + SVM models.
Save models and vectorizers into models/.
b) Train Text + Timeline Model
If you have a training script (e.g. train.py or train_multimodal.py):

python train.py  # or: python train_multimodal.py
This will:

Load social media text and timeline data.
Compute BERT embeddings and timeline features.
Train the multimodal classifier and save weights (e.g. in models/).
Make sure dataset paths in config.py are correct before running.

2. Run the Streamlit App
From the project root:

streamlit run app.py
or, if your entry file is different:

streamlit run run_app.py
Then open the URL shown in the terminal (usually http://localhost:8501).

Typical Streamlit UI flow:

Text input tab:
Paste social media text or a short paragraph.
Click a button to get depression probability.
Audio upload tab:
Upload a wav / mp3 file.
The app runs transcription + prediction and shows the result.
📊 Example Results
Replace with your actual numbers.

Text + Timeline (BERT)

Accuracy: XX.X %
F1-score (depressed class): YY.Y
Audio LSTM

Accuracy: AA.A %
F1-score: BB.B
Audio TF–IDF + SVM

Accuracy: CC.C %
F1-score: DD.D
⚖️ Ethical & Privacy Notice
This project is intended for research and educational purposes only.

It is not a medical device.
It must not be used for self-diagnosis or clinical decisions.
Any deployment on real data requires:
Informed consent
Proper anonymization
Oversight by qualified mental health professionals
📜 License
(Add your chosen license here, e.g. MIT or Apache-2.0, and include a LICENSE file.)

🙏 Acknowledgements
Hugging Face Transformers for BERT and Whisper.
Scikit-learn, TensorFlow/Keras, and PyTorch for model implementation.
The creators of the depression datasets used in this project.
