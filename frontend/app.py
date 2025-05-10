import os
import requests
import streamlit as st
import whisper
import tempfile
import re

# Point to your FastAPI backend via env var, defaulting to container name
API_URL = os.getenv("API_URL", "http://transcept-backend:8000/summarize")

# NYU color scheme
NYU_PURPLE = "#57068c"
NYU_ACCENT = "#8f43b2"
NYU_LIGHT_PURPLE = "#f3e6fa"

os.environ["PATH"] += os.pathsep + os.path.abspath(".")

st.set_page_config(page_title="Transcept | NYU", page_icon="🎤", layout="wide")

# Custom CSS for extra fanciness
st.markdown(f"""
    <style>
        body {{
            background: linear-gradient(120deg, {NYU_LIGHT_PURPLE} 0%, #fff 100%);
        }}
        .block-container {{
            padding-top: 0rem;
        }}
        .nyu-header {{
            background: linear-gradient(90deg, {NYU_PURPLE} 0%, {NYU_ACCENT} 100%);
            border-radius: 0 0 30px 30px;
            box-shadow: 0 4px 16px #b799e288;
            padding: 2rem 0 1rem 0;
            margin-bottom: 2rem;
        }}
        .nyu-card {{
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 2px 16px #b799e244;
            padding: 2em 1.5em 1.5em 1.5em;
            margin-bottom: 2em;
            transition: box-shadow 0.3s;
        }}
        .nyu-card:hover {{
            box-shadow: 0 6px 32px #57068c33;
        }}
        .nyu-footer {{
            text-align: center;
            color: {NYU_PURPLE};
            margin-top: 2em;
            font-size: 1.1em;
        }}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
    <div style='background: linear-gradient(90deg, {NYU_PURPLE} 0%, {NYU_ACCENT} 100%);
                padding: 2.5rem 0 2rem 0; border-radius: 0 0 30px 30px;
                box-shadow: 0 4px 16px #b799e288; text-align: center; margin-bottom: 2.5rem;'>
        <h1 style='color: white; font-size: 3em; margin: 0;'>Transcept</h1>
        <p style='color: #fff; font-size: 1.3em; margin: 0;'>NYU Audio/Video Transcription & QA</p>
    </div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div style='margin-bottom: 0.5em;'>
          <h2 style='color: #222; font-size: 2.2em; font-weight: 700; margin: 0;'>Input</h2>
          <hr style='height:3px; border:none; background:
             linear-gradient(90deg,#ff5858,#f09819,#43e97b,#38f9d7,#667eea);
             margin:0 0 1em 0;'>
        </div>
    """, unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload your audio or video file", type=["wav","mp3","m4a","mp4"])
    if audio_file:
        if audio_file.type.startswith("video") or audio_file.name.lower().endswith(".mp4"):
            st.video(audio_file)
        else:
            st.audio(audio_file, format="audio/wav")
    st.markdown("<div style='height:1.5em;'></div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style='margin-bottom: 0.5em;'>
          <h2 style='color: #222; font-size: 2.2em; font-weight: 700; margin: 0;'>Transcription Output</h2>
          <hr style='height:3px; border:none; background:
             linear-gradient(90deg,#ff5858,#f09819,#43e97b,#38f9d7,#667eea);
             margin:0 0 1em 0;'>
        </div>
    """, unsafe_allow_html=True)

    transcription = None
    summary = None

    if audio_file:
        # Whisper transcription
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        st.info("Transcribing with Whisper ASR... Please wait.")
        asr_model = whisper.load_model("base")
        result = asr_model.transcribe(tmp_path)
        transcription = result["text"]
        st.success(transcription)

        # Call FastAPI summarization
        try:
            resp = requests.post(API_URL, json={"text": transcription}, timeout=60)
            resp.raise_for_status()
            summary = resp.json().get("summary", "")
        except Exception as e:
            st.error(f"Error fetching summary: {e}")
            summary = None

        # Display summary
        if summary:
            st.markdown(f"""
                <div style='background: {NYU_LIGHT_PURPLE};
                            border-left: 6px solid {NYU_PURPLE};
                            border-radius: 12px;
                            padding: 1em;
                            margin-top: 1.5em;
                            font-size: 1.1em;'>
                  <b>Model Summary:</b> {summary}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No summary returned.")
    else:
        st.markdown("""
            <div style='max-width:420px;margin:0 auto;'>
              <div style='background:#eaf4fd;color:#113355;
                          border-radius:12px;padding:1.2em;
                          font-size:1.1em;text-align:center;'>
                Transcription will appear here after you upload an audio or video file.
              </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5em;'></div>", unsafe_allow_html=True)

# QA Section
st.markdown(f"""
    <div class='nyu-card' style='max-width:700px;margin:0 auto;'>
      <h2 style='color:{NYU_PURPLE};text-align:center;margin-bottom:1.5em;'>
        Question & Answer
      </h2>
""", unsafe_allow_html=True)

if transcription:
    question = st.text_area("Ask a question about the transcription:", key="qa_input", height=80)
    answer = ""
    if question:
        answer = f"(Sample answer to: '{question}')"
    st.text_area("Answer:", value=answer, height=80, key="qa_output", disabled=True)
    if not question:
        st.info("Type a question to get an answer based on the transcription.")
else:
    st.info("Upload and transcribe an audio file to enable the QA system.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"""
    <div class='nyu-footer'>
      &copy; 2024 New York University | Powered by Whisper ASR & Streamlit
    </div>
""", unsafe_allow_html=True)
