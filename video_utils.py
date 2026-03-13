"""
Extract audio from video file at 16 kHz mono for Whisper.
Uses ffmpeg (subprocess) or moviepy if available.
"""

import os
import subprocess
import tempfile
from typing import Optional, Tuple

# Optional: set to True to print extraction errors (e.g. when debugging)
DEBUG_EXTRACT = os.environ.get("DEBUG_VIDEO_EXTRACT", "").lower() in ("1", "true", "yes")


def _get_ffmpeg_exe() -> Optional[str]:
    """Return path to ffmpeg: system first, then imageio_ffmpeg (bundled with moviepy)."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _run_ffmpeg(video_path: str, wav_path: str, sample_rate: int, ffmpeg_exe: Optional[str] = None) -> Tuple[bool, str]:
    """Run ffmpeg. Returns (success, error_message)."""
    video_path = os.path.abspath(os.path.normpath(video_path))
    if not os.path.isfile(video_path):
        return False, "Video file not found"
    exe = ffmpeg_exe if ffmpeg_exe else _get_ffmpeg_exe()
    if not exe:
        return False, "ffmpeg not found. Install from https://ffmpeg.org/download.html and add to PATH, or: pip install moviepy"
    cmd = [
        exe, "-y", "-i", video_path,
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate),
        "-vn", wav_path
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if out.returncode != 0:
            err = (out.stderr or out.stdout or "").strip()
            if not err:
                err = "ffmpeg failed with code %s" % out.returncode
            return False, err
        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            return False, "ffmpeg produced no output file"
        return True, ""
    except FileNotFoundError:
        return False, "ffmpeg not found. Install from https://ffmpeg.org/download.html and add to PATH."
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"
    except Exception as e:
        return False, str(e)


def extract_audio_ffmpeg(video_path: str, sample_rate: int = 16000) -> Optional[str]:
    """
    Extract audio to a temp WAV file at 16 kHz mono.
    Uses system ffmpeg or imageio_ffmpeg (from pip install moviepy) if available.
    Returns path to temp wav file, or None on failure. Caller should delete the temp file.
    """
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        ok, err = _run_ffmpeg(video_path, wav_path, sample_rate, _get_ffmpeg_exe())
        if not ok:
            if DEBUG_EXTRACT:
                print(f"[video_utils] ffmpeg failed: {err}")
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            return None
        return wav_path
    except Exception:
        return None


def extract_audio_moviepy(video_path: str, sample_rate: int = 16000) -> Optional[str]:
    """Extract audio using moviepy to a temp WAV. Returns temp path or None."""
    try:
        from moviepy.editor import VideoFileClip
        video_path = os.path.abspath(os.path.normpath(video_path))
        if not os.path.isfile(video_path):
            return None
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        clip.audio.write_audiofile(wav_path, fps=sample_rate, nbytes=2, codec="pcm_s16le", logger=None)
        clip.close()
        return wav_path
    except Exception as e:
        if DEBUG_EXTRACT:
            print(f"[video_utils] moviepy failed: {e}")
        return None


def extract_audio(video_path: str, sample_rate: int = 16000) -> Optional[str]:
    """Try ffmpeg first, then moviepy. Returns path to temp wav or None."""
    wav = extract_audio_ffmpeg(video_path, sample_rate)
    if wav is not None:
        return wav
    return extract_audio_moviepy(video_path, sample_rate)


def extract_audio_with_reason(video_path: str, sample_rate: int = 16000) -> Tuple[Optional[str], str]:
    """
    Extract audio and return (wav_path, error_message).
    If successful, error_message is empty. Use this when you need to show the user why it failed.
    """
    video_path = os.path.abspath(os.path.normpath(video_path))
    if not os.path.isfile(video_path):
        return None, "Video file not found."
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        ok, err = _run_ffmpeg(video_path, wav_path, sample_rate, _get_ffmpeg_exe())
        if ok:
            return wav_path, ""
        try:
            os.remove(wav_path)
        except OSError:
            pass
        wav = extract_audio_moviepy(video_path, sample_rate)
        if wav is not None:
            return wav, ""
        return None, err or "Audio extraction failed (try installing ffmpeg or moviepy)."
    except Exception as e:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        return None, str(e)
