import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from utils.dataset_loader import load_dataset

print("Loading dataset...")

(X_train, X_test, y_train, y_test), word2idx = load_dataset("dataset")

print("Building model...")

model = Sequential([
    TimeDistributed(Conv2D(16,(3,3),activation='relu'), input_shape=(30,64,64,3)),
    TimeDistributed(MaxPooling2D()),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(word2idx), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training started...")

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=12,
          batch_size=4)

os.makedirs("models", exist_ok=True)

model.save("models/lip_model.h5")

with open("models/word_map.json", "w") as f:
    json.dump(word2idx, f)

print("Training Completed Successfully")