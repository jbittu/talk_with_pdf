# Talk with PDF (Text-Only)

A Streamlit application that allows you to upload a PDF and ask questions about it using text or voice input. The application uses Whisper for speech-to-text, TinyLlama for question answering, FAISS for vector search, and offline pyttsx3 for speaking the answers.

## Features

- Upload and process PDF documents
- Extract and display PDF metadata
- Ask questions via text input
- Ask questions via voice input (using Whisper)
- Get answers based on the PDF content
- Optional spoken answers (offline TTS via pyttsx3)
- Chat history with clear function
- Voice recording quality settings

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

```
streamlit run app.py
```

### Notes
- Spoken answers can be toggled in the sidebar (Speak answers aloud). Audio is saved under `outputs/audio`.
- You can adjust TTS settings in `config.py` (rate, volume, voice id).

## Model Information

### Whisper (Speech-to-Text)
- Default model: `small`

### TinyLlama (LLM)
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Configuration

You can modify the configuration in `config.py` to change model paths, generation parameters, and output directories.