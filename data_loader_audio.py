"""
Data loader for audio files (depression detection).
Expects: <root>/Audio_Dataset/<class>/*.wav, or <root>/audio/<class>/*.wav, or <root>/<class>/*.wav.
Also supports nested structure:
  - <root>/Audio_Dataset/Audio_Dataset/Depression/{Stage1,Stage2}/*.wav (depressed=1)
  - <root>/Audio_Dataset/Audio_Dataset/Normal/*.wav (normal=0)
  - Or Normal inside Depression: .../Depression/Normal/*.wav (normal=0).
"""

import os
from typing import List, Tuple

CLASSES = {"normal": 0, "depressed": 1}
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


def find_audio_dirs(dataset_path: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Scan dataset_path for class-named folders containing audio files.
    Returns (base_dir, [(file_path, label), ...]).
    Tries: <path>/Audio_Dataset/<class>/*, <path>/audio/<class>/*, then <path>/<class>/*.
    Also handles nested structure: <path>/Audio_Dataset/Audio_Dataset/Depression/{Stage1,Stage2,Normal}/*.wav
    """
    dataset_path = os.path.normpath(dataset_path)
    if not os.path.isdir(dataset_path):
        return dataset_path, []

    out: List[Tuple[str, int]] = []

    # Try Audio_Dataset (Multimodel_Dataset layout), then audio, then root
    for prefix in ["Audio_Dataset", "audio", ""]:
        base = os.path.join(dataset_path, prefix) if prefix else dataset_path
        if prefix and not os.path.isdir(base):
            continue
        
        # First, try standard structure: <base>/normal/*, <base>/depressed/*
        for class_name, label in CLASSES.items():
            class_dir = os.path.join(base, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith(AUDIO_EXTENSIONS):
                        out.append((os.path.join(class_dir, f), label))
        
        # Nested structure: <base>/Audio_Dataset/Depression/{Stage1,Stage2} and Normal (sibling or inside Depression)
        nested_audio = os.path.join(base, "Audio_Dataset")
        if os.path.isdir(nested_audio):
            depression_dir = os.path.join(nested_audio, "Depression")
            if os.path.isdir(depression_dir):
                # Stage1 and Stage2 are depressed (label 1)
                for stage in ["Stage1", "Stage2"]:
                    stage_dir = os.path.join(depression_dir, stage)
                    if os.path.isdir(stage_dir):
                        for f in os.listdir(stage_dir):
                            if f.lower().endswith(AUDIO_EXTENSIONS):
                                out.append((os.path.join(stage_dir, f), 1))  # depressed
                
                # Normal (label 0): either inside Depression/Normal or sibling Normal at nested_audio/Normal
                for normal_dir in [
                    os.path.join(depression_dir, "Normal"),
                    os.path.join(nested_audio, "Normal"),
                ]:
                    if os.path.isdir(normal_dir):
                        for f in os.listdir(normal_dir):
                            if f.lower().endswith(AUDIO_EXTENSIONS):
                                out.append((os.path.join(normal_dir, f), 0))  # normal
                        break
        
        if out:
            return base, out

    return dataset_path, out


def load_audio_paths_and_labels(dataset_path: str) -> Tuple[List[str], List[int]]:
    """
    Return (list of audio file paths, list of labels 0/1).
    """
    _, pairs = find_audio_dirs(dataset_path)
    if not pairs:
        return [], []
    paths = [p for p, _ in pairs]
    labels = [l for _, l in pairs]
    return paths, labels
