# app.py
import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
import base64
from pathlib import Path

from utils.preprocess import extract_frames
from utils.translator import translate_to_english
from utils.tts import generate_audio  # singing + echo + long vowels

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Lip2Song System", layout="centered")

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}
.main {
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    border-radius: 15px;
}
h1 {
    text-align: center;
    color: #4A00E0;
}
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;        
    height: 50px;
    width: 150px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Let's create a song! 🎶")
st.sidebar.image(
    "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWthMW11bzV1ZGhyNjk1NXJ4ZHRhcWk1enluMTFlbDBybXJ1NHNycyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/l0itSaTnyz59giRpfg/giphy.gif", 
    width=250
)
st.sidebar.image(
    "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGdvMXNvbnR0bWxmbDZiNDE1eHp2eW92eWtheG4wbHUzMWJoaDBhZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/6m7jegpkxj4c2zNx8f/giphy.gif", 
    width=250
)

# -------------------------
# Title
# -------------------------
st.title("🎬 Lip2Song")
st.write("Upload a silent video → Get AI lip-synced singing output!")

# -------------------------
# Upload
# -------------------------
uploaded_video = st.file_uploader("📤 Upload Silent Video", type=["mp4"])
language = st.selectbox("Select Output Language", ["--select--", "English", "Hindi", "Tamil"])

if uploaded_video is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Display small video
    video_bytes = Path(video_path).read_bytes()
    video_base64 = base64.b64encode(video_bytes).decode()
    st.markdown(f"""
        <div style="display:flex; justify-content:center;">
            <video width="280" height="170" controls style="border-radius:10px;">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # Button
    # -------------------------
    if st.button("🚀 Predict & Lip Sync"):
        st.info("⏳ Processing...")

        try:
            # -------------------------
            # Load Lip Model & Word Map
            # -------------------------
            model = tf.keras.models.load_model("models/lip_model.h5")
            with open("models/word_map.json") as f:
                word2idx = json.load(f)
            idx2word = {v: k for k, v in word2idx.items()}

            # -------------------------
            # Extract Frames
            # -------------------------
            frames = extract_frames(video_path)
            frames = np.expand_dims(frames, axis=0)

            # -------------------------
            # Predict Word
            # -------------------------
            prediction = model.predict(frames)
            word_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction) * 100)
            mal_word = idx2word[word_id]

            # -------------------------
            # Translate
            # -------------------------
            if language == "English":
                translated_word = translate_to_english(mal_word)
            else:
                translated_word = mal_word

            # -------------------------
            # Generate Singing Audio
            # -------------------------
            audio_path = generate_audio(translated_word)  # <-- no background_path

            # -------------------------
            # Show Prediction Results
            # -------------------------
            st.markdown("## 🎯 Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Word")
                st.success(mal_word)
            with col2:
                st.subheader("📊 Confidence")
                st.success(f"{confidence:.2f}%")

            st.markdown("### 🌐 Translated Word")
            st.info(translated_word)

            st.markdown("### 🔊 Singing Audio")
            st.audio(audio_path)

            # -------------------------
            # Wav2Lip Lip Sync
            # -------------------------
            st.markdown("### 🎬 Generating Lip Sync Video...")

            command = (
        f"cd Wav2Lip && "
        f"python inference.py "
        f"--checkpoint_path checkpoints/wav2lip_gan.pth "
        f"--face ../{video_path} "
        f"--audio ../{audio_path} "
        f"--outfile result.mp4 "
        f"--pads 30 0 0 0 "    # <-- push face up by 30 pixels
        f"--resize_factor 2"   # zoom lips 2x
    )

            with st.spinner("Running Wav2Lip... please wait ⏳"):
                os.system(command)

            # -------------------------
            # Show Output Video
            # -------------------------
            result_path = "Wav2Lip/result.mp4"
            if Path(result_path).exists():
                result_bytes = Path(result_path).read_bytes()
                result_base64 = base64.b64encode(result_bytes).decode()
                st.success("✅ Lip Sync Generated!")

                st.markdown(f"""
                    <div style="display:flex; justify-content:center;">
                        <video width="300" height="180" controls style="border-radius:10px;">
                            <source src="data:video/mp4;base64,{result_base64}" type="video/mp4">
                        </video>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Lip sync video not found")

        except Exception as e:
            st.error(f"❌ Error: {e}")