"""
List structure of the audio/video dataset (Multimodel_Dataset).
Usage: python inspect_audio_video_dataset.py [path]
Default path: config.AUDIO_VIDEO_DATASET_PATH or 'Multimodel_Dataset'
"""

import os
import sys

try:
    from config import AUDIO_VIDEO_DATASET_PATH
except ImportError:
    AUDIO_VIDEO_DATASET_PATH = "Multimodel_Dataset"


def main():
    path = (sys.argv[1] if len(sys.argv) > 1 else None) or AUDIO_VIDEO_DATASET_PATH
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        print(f"Path does not exist or is not a directory: {path}")
        return
    print(f"Dataset root: {os.path.abspath(path)}")
    print("\nTop-level contents:")
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            count = len(os.listdir(full))
            print(f"  [dir]  {name}/  ({count} items)")
        else:
            print(f"  [file] {name}")
    print("\nExpected layout for audio/video loaders:")
    print("  - Audio: <root>/Audio_Dataset/normal/*.wav, <root>/Audio_Dataset/depressed/*.wav")
    print("  - Video: <root>/Video_Dataset/normal/*.mp4, <root>/Video_Dataset/depressed/*.mp4")


if __name__ == "__main__":
    main()
