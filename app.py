import io
import tempfile
import time
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from modules.pdf_utils import extract_text_from_pdf
from modules.vectorstore_utils import build_index, retrieve_context
from modules.qa_utils import load_llm, answer_with_llm
from modules.whisper_utils import load_whisper, transcribe_bytes
from modules.tts_utils import tts_speak


st.set_page_config(page_title="Talk with PDF (Text-Only)", page_icon="📄", layout="wide")
st.title(" Talk with PDF ")

with st.sidebar:
    st.markdown("### How to use")
    st.markdown("1. Upload a PDF\n2. Ask by text or voice\n3. Read the answer")
    st.markdown("---")
    st.markdown("**Tips**: Ask concise, PDF-specific questions.")

# App state
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "llm" not in st.session_state:
    st.session_state.llm = load_llm()
if "whisper" not in st.session_state:
    st.session_state.whisper = load_whisper()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = None
if "voice_reply" not in st.session_state:
    st.session_state.voice_reply = True

# Upload PDF
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_pdf:
    with st.spinner("Extracting text and metadata..."):
        pdf_text, metadata = extract_text_from_pdf(uploaded_pdf)
        # Clean the extracted text
        from modules.pdf_utils import clean_text
        pdf_text = clean_text(pdf_text)
        st.session_state.pdf_text = pdf_text
        
        # Store metadata if available
        if metadata:
            st.session_state.pdf_metadata = metadata

    if not pdf_text.strip():
        st.warning("Could not extract text from this PDF. Try another file.")
    else:
        # Display PDF metadata if available
        if 'pdf_metadata' in st.session_state and st.session_state.pdf_metadata:
            with st.expander("📑 PDF Metadata"):
                for key, value in st.session_state.pdf_metadata.items():
                    st.write(f"**{key.title()}:** {value}")
        
        with st.spinner("Building vector index (FAISS + MiniLM embeddings)..."):
            st.session_state.index = build_index(pdf_text)
        st.success("✅ PDF ready for questions!")

st.divider()

# Chat UI
if st.session_state.index is None:
    st.info("Upload a PDF to begin.")
    st.stop()

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
    
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Text question
user_q = st.text_input("💬 Type your question about the PDF:")
col1, col2 = st.columns([1, 1])

def handle_question(question: str):
    # Add question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Retrieve context from the PDF
    ctx = retrieve_context(st.session_state.index, question, k=3)
    
    # Create a placeholder for streaming effect
    answer_placeholder = st.empty()
    answer_placeholder.markdown("*Thinking...*")
    
    # Get answer from LLM
    ans = answer_with_llm(st.session_state.llm, question, ctx)
    
    # Add answer to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": ans})
    
    # Display answer with a simple streaming effect
    answer_placeholder.empty()
    answer_container = answer_placeholder.container()
    answer_container.markdown("**Answer:**")
    
    # Simple streaming effect
    full_answer = ""
    for i in range(min(10, len(ans))):
        chunk_size = max(1, len(ans) // 10)
        end_idx = min((i+1) * chunk_size, len(ans))
        full_answer = ans[:end_idx]
        answer_container.markdown(full_answer)
        time.sleep(0.1)
    
    # Display final answer
    answer_container.markdown(ans)

    # Voice out the answer if enabled
    if st.session_state.voice_reply:
        try:
            audio_path = tts_speak(ans)
            st.audio(audio_path, format="audio/wav")
        except Exception as e:
            st.warning(f"TTS error: {e}")

with col1:
    if st.button("Ask (text)", use_container_width=True) and user_q.strip():
        handle_question(user_q.strip())

# Voice settings in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Voice Settings")
    st.session_state.voice_reply = st.checkbox("Speak answers aloud", value=st.session_state.voice_reply)
    voice_quality = st.select_slider(
        "Recording Quality",
        options=["Low", "Medium", "High"],
        value="Medium",
        help="Higher quality uses more bandwidth but may improve transcription accuracy"
    )
    
    # Map quality settings to audio recorder parameters
    quality_settings = {
        "Low": {"sample_rate": 16000, "pause_threshold": 2.0},
        "Medium": {"sample_rate": 32000, "pause_threshold": 1.5},
        "High": {"sample_rate": 44100, "pause_threshold": 1.0}
    }
    
    # Show current settings
    st.markdown(f"**Sample Rate:** {quality_settings[voice_quality]['sample_rate']} Hz")
    st.markdown(f"**Pause Threshold:** {quality_settings[voice_quality]['pause_threshold']} seconds")

# Voice question
with col2:
    st.markdown("🎤 **Ask with your voice**")
    
    # Get settings based on selected quality
    current_settings = quality_settings[voice_quality]
    
    audio_bytes = audio_recorder(
        text="Hold to record • release to send",
        recording_color="#d13b3b",
        neutral_color="#4c8bf5",
        icon_name="microphone",
        icon_size="2x",
        sample_rate=current_settings["sample_rate"],
        pause_threshold=current_settings["pause_threshold"]
    )
    
    if audio_bytes:
        # Show audio length feedback
        audio_length = len(audio_bytes) / (current_settings["sample_rate"] * 2)  # Approximate length in seconds
        st.caption(f"Recording length: {audio_length:.1f} seconds")
        
        with st.spinner("Transcribing (Whisper)..."):
            try:
                q = transcribe_bytes(st.session_state.whisper, audio_bytes)
                # Show confidence feedback
                if q:
                    st.success("Transcription successful!")
                else:
                    st.warning("Transcription produced no text. Try speaking louder or closer to the microphone.")
            except Exception as e:
                st.error(f"STT error: {e}")
                q = ""
        if q:
            st.markdown(f"**You asked (voice):** {q}")
            handle_question(q)
