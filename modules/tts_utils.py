from pathlib import Path
import tempfile
import threading
from typing import Optional
import pyttsx3
from config import AUDIO_OUT_DIR, TTS_RATE_WPM, TTS_VOICE_ID, TTS_VOLUME

_engine: Optional[pyttsx3.Engine] = None
_engine_lock = threading.Lock()

def _get_engine() -> pyttsx3.Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            engine = pyttsx3.init()
            # Configure voice, rate, and volume if available
            try:
                if TTS_RATE_WPM is not None:
                    engine.setProperty("rate", int(TTS_RATE_WPM))
            except Exception:
                pass
            try:
                if TTS_VOLUME is not None:
                    engine.setProperty("volume", float(TTS_VOLUME))
            except Exception:
                pass
            if TTS_VOICE_ID:
                try:
                    engine.setProperty("voice", TTS_VOICE_ID)
                except Exception:
                    pass
            _engine = engine
        return _engine

def tts_speak(text: str) -> str:
    """Synthesize speech to a WAV file and return its path."""
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=AUDIO_OUT_DIR).name)
    engine = _get_engine()
    # pyttsx3 supports save_to_file on Windows (SAPI5). This writes a WAV file.
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    return str(out_path)
