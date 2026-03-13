"""
Configuration file for Depression Detection Project
"""

# Dataset paths
DATASET_PATH = "Dataset_MDDL (1)/Dataset"
POSITIVE_TWEET_PATH = "Dataset_MDDL (1)/Dataset/labeled/positive/data/tweet"
NEGATIVE_TWEET_PATH = "Dataset_MDDL (1)/Dataset/labeled/negative/data/tweet"
POSITIVE_TIMELINE_PATH = "Dataset_MDDL (1)/Dataset/labeled/positive/data/timeline"
NEGATIVE_TIMELINE_PATH = "Dataset_MDDL (1)/Dataset/labeled/negative/data/timeline"
UNLABELED_TIMELINE_PATH = "Dataset_MDDL (1)/Dataset/unlabeled/data/timeline"

# Model configuration
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQUENCE_LENGTH = 128
HIDDEN_DIM = 256
DROPOUT = 0.3
NUM_TIMELINE_FEATURES = 4

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
TRAIN_VAL_SPLIT = 0.8

# Audio/Video dataset: root folder containing Audio_Dataset/ and Video_Dataset/
AUDIO_VIDEO_DATASET_PATH = "Multimodel_Dataset"

# Model paths
MODEL_SAVE_PATH = "models"
BEST_MODEL_PATH = "models/best_model.pt"
SCALER_PATH = "models/timeline_scaler.pkl"

# Inference configuration
PREDICTION_THRESHOLD = 0.5
OUTPUT_PREDICTIONS_PATH = "predictions_unlabeled.csv"

# Negative words for sentiment analysis (simplified)
NEGATIVE_WORDS = [
    'sad', 'depressed', 'lonely', 'anxious', 'stress', 'worried',
    'hurt', 'pain', 'cry', 'tears', 'hopeless', 'empty', 'tired',
    'suicide', 'kill', 'die', 'death', 'worthless', 'useless'
]
