from typing import Union
import io
from faster_whisper import WhisperModel
from config import WHISPER_MODEL_ID, WHISPER_LOCAL_PATH
import torch
import os

_whisper = None

def load_whisper():
    """Load faster-whisper model (prefers local path if available)."""
    global _whisper
    if _whisper is not None:
        return _whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model_id_or_path = str(WHISPER_LOCAL_PATH) if WHISPER_LOCAL_PATH.exists() else WHISPER_MODEL_ID
    _whisper = WhisperModel(model_id_or_path, device=device, compute_type=compute_type)
    return _whisper

def transcribe_bytes(whisper_model: WhisperModel, wav_bytes: Union[bytes, bytearray]) -> str:
    """Transcribe WAV/PCM bytes recorded in the browser (audio_recorder_streamlit)."""
    # Save to temp buffer file-like object
    tmp = io.BytesIO(wav_bytes)
    segments, _ = whisper_model.transcribe(tmp)
    text = " ".join([seg.text for seg in segments]).strip()
    return text
