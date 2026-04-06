import os
import json
import numpy as np
import tensorflow as tf
from utils.preprocess import extract_frames
from utils.translator import translate_to_english
from utils.tts import generate_audio

# -----------------------------
# 1. Check required files
# -----------------------------
if not os.path.exists("models/lip_model.h5"):
    print(" Model not found! Run train.py first.")
    exit()

if not os.path.exists("test.mp4"):
    print("test.mp4 not found! Place it in main project folder.")
    exit()

# -----------------------------
# 2. Load trained model
# -----------------------------
print(" Loading model...")
model = tf.keras.models.load_model("models/lip_model.h5")

# -----------------------------
# 3. Load word mapping
# -----------------------------
with open("models/word_map.json") as f:
    word2idx = json.load(f)

idx2word = {v: k for k, v in word2idx.items()}

# -----------------------------
# 4. Extract frames
# -----------------------------
print(" Processing video...")
frames = extract_frames("test.mp4")
frames = np.expand_dims(frames, axis=0)

# -----------------------------
# 5. Predict
# -----------------------------
prediction = model.predict(frames)
word_id = np.argmax(prediction)
confidence = np.max(prediction) * 100

mal_word = idx2word[word_id]

print("\n Predicted Malayalam Word:", mal_word)
print(" Confidence: {:.2f}%".format(confidence))

# -----------------------------
# 6. Translate
# -----------------------------
eng_text = translate_to_english(mal_word)
print(" English Translation:", eng_text)

# -----------------------------
# 7. Generate English Audio
# -----------------------------
generate_audio(eng_text)
print(" English audio generated: english_output.wav")