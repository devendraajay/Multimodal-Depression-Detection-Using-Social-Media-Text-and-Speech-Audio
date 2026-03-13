"""
Data loader for video files (depression detection).
Expects: <root>/Video_Dataset/<class>/*.mp4, or <root>/video/<class>/*.mp4, or <root>/<class>/*.mp4.
Also supports nested structure: <root>/Video_Dataset/Video_Dataset/Video_Speech_Actor_XX/Actor_XX/*.mp4
where videos in Actor folders are labeled as depressed (1).
"""

import os
from typing import List, Tuple

CLASSES = {"normal": 0, "depressed": 1}
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def find_video_dirs(dataset_path: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Scan dataset_path for class-named folders containing video files.
    Returns (base_dir, [(file_path, label), ...]).
    Tries: <path>/Video_Dataset/<class>/*, <path>/video/<class>/*, then <path>/<class>/*.
    Also handles nested structure: <path>/Video_Dataset/Video_Dataset/Video_Speech_Actor_XX/Actor_XX/*.mp4
    """
    dataset_path = os.path.normpath(dataset_path)
    if not os.path.isdir(dataset_path):
        return dataset_path, []

    out: List[Tuple[str, int]] = []

    # Try Video_Dataset (Multimodel_Dataset layout), then video, then root
    for prefix in ["Video_Dataset", "video", ""]:
        base = os.path.join(dataset_path, prefix) if prefix else dataset_path
        if prefix and not os.path.isdir(base):
            continue
        
        # First, try standard structure: <base>/normal/*, <base>/depressed/*
        for class_name, label in CLASSES.items():
            class_dir = os.path.join(base, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith(VIDEO_EXTENSIONS):
                        out.append((os.path.join(class_dir, f), label))
        
        # Also check for nested Video_Dataset structure: <base>/Video_Dataset/Video_Speech_Actor_XX/Actor_XX/*.mp4
        nested_video = os.path.join(base, "Video_Dataset")
        if os.path.isdir(nested_video):
            # First, check for standard normal/depressed folders (like audio dataset)
            normal_dir = os.path.join(nested_video, "normal")
            depressed_dir = os.path.join(nested_video, "depressed")
            if os.path.isdir(normal_dir):
                for f in os.listdir(normal_dir):
                    if f.lower().endswith(VIDEO_EXTENSIONS):
                        out.append((os.path.join(normal_dir, f), 0))  # normal
            if os.path.isdir(depressed_dir):
                for f in os.listdir(depressed_dir):
                    if f.lower().endswith(VIDEO_EXTENSIONS):
                        out.append((os.path.join(depressed_dir, f), 1))  # depressed
            
            # Also check for Video_Speech_Actor_XX structure (RAVDESS-style)
            # Look for Video_Speech_Actor_XX folders
            for item in os.listdir(nested_video):
                item_path = os.path.join(nested_video, item)
                if os.path.isdir(item_path) and item.startswith("Video_Speech_Actor"):
                    # Check for Actor_XX subfolder
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path) and subitem.startswith("Actor"):
                            # Determine label based on actor number pattern
                            # Common RAVDESS pattern: Actors 1-12 = neutral/normal (0), Actors 13-24 = emotional/depressed (1)
                            actor_num = None
                            try:
                                # Extract actor number from Actor_XX (e.g., "Actor_01" -> 1, "Actor_13" -> 13)
                                num_str = subitem.replace("Actor_", "").replace("Actor", "").lstrip("0")
                                actor_num = int(num_str) if num_str else None
                            except:
                                pass
                            
                            # Label assignment: Actor 1-12 = normal (0), Actor 13-24 = depressed (1)
                            # This matches RAVDESS-style emotion datasets
                            if actor_num is not None:
                                label = 0 if actor_num <= 12 else 1
                            else:
                                # Fallback: if we can't parse actor number, default to depressed (1)
                                label = 1
                            
                            for f in os.listdir(subitem_path):
                                if f.lower().endswith(VIDEO_EXTENSIONS):
                                    out.append((os.path.join(subitem_path, f), label))
        
        if out:
            return base, out

    return dataset_path, out


def load_video_paths_and_labels(dataset_path: str) -> Tuple[List[str], List[int]]:
    """Return (list of video file paths, list of labels 0/1)."""
    _, pairs = find_video_dirs(dataset_path)
    if not pairs:
        return [], []
    
    # Debug: print label distribution
    labels = [l for _, l in pairs]
    unique_labels, counts = zip(*[(l, labels.count(l)) for l in set(labels)]) if labels else ([], [])
    print(f"Loaded {len(pairs)} video files:")
    for label, count in zip(unique_labels, counts):
        class_name = "normal" if label == 0 else "depressed"
        print(f"  {class_name} (label {label}): {count} files")
    
    return [p for p, _ in pairs], [l for _, l in pairs]
