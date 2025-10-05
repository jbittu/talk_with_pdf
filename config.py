from pathlib import Path

# === Whisper (STT) ===
# Either "small", "medium", "large-v3", etc. or a local path inside models/whisper/*
WHISPER_MODEL_ID = "small"              # fast + decent
WHISPER_LOCAL_PATH = Path("models/whisper/small")  # if present, will be used

# === Open-source LLM ===
# Choose a light, CPU-friendly default. You can switch to Mistral-7B if you have GPU.
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_LOCAL_PATH = Path("models/llm/tinyllama-1.1b-chat")  # if present, will be used

# Generation params
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2

# === Embeddings ===
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# === FAISS ===
FAISS_INDEX_DIR = Path("outputs/faiss_index")



# === Outputs ===
AUDIO_OUT_DIR = Path("outputs/audio")
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

# === TTS (offline pyttsx3) ===
# You can list voices with a small helper if needed; by default uses system default voice
TTS_RATE_WPM = 175  # words per minute
TTS_VOLUME = 1.0    # 0.0 - 1.0
TTS_VOICE_ID = None # e.g., a specific voice id string from pyttsx3