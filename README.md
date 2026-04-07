# 🎵 Lip2Song

Lip2Song is a deep learning-based system that converts **silent Malayalam singing videos into English audio with lip synchronization.The project combines Computer Vision, Natural Language Processing, and Speech Synthesis to bridge the gap between visual speech recognition and multilingual communication.

---

## 🚀 Features

- 🎥 Extracts lip movements from silent video
- 🧠 CNN + LSTM model for lip reading (Malayalam lyrics prediction)
- 🌐 Translates Malayalam text → English
- 🔊 Converts English text → Speech (TTS)
- 🎬 Generates lip-synced video using Wav2Lip

---

## 🏗️ Project Architecture

1. Input silent video  
2. Frame extraction  
3. CNN extracts spatial features  
4. LSTM predicts sequence (lyrics)  
5. Translation API converts to English  
6. Text-to-Speech generates audio  
7. Wav2Lip syncs audio with video  

---

## 📂 Project Structure
Lip2Song/
│── models/ # Pretrained model and customised lip model
│── src/ # Source code
│── data/ # Sample inputs
│── output/ # Generated results
│── requirements.txt
│── README.md


🧠 Model Details
CNN + LSTM
CNN extracts spatial features from frames
LSTM captures temporal sequence of lip movements
Outputs predicted Malayalam lyrics
Wav2Lip
Syncs generated English audio with lip movements
Produces realistic talking/singing video




📊 Technologies Used
Python
streamlit app(frontend)
TensorFlow / Keras
OpenCV
Wav2Lip
Google Translate API
Text-to-Speech (TTS)



🎯Applications
Accessibility for hearing-impaired users
Silent video understanding
Multilingual media translation
AI-based dubbing


📌Future Improvements
Real-time processing
Support for more languages
Improved lip-reading accuracy


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Lip2Song.git
cd Lip2Song
pip install -r requirements.txt




