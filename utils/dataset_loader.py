import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from .preprocess import extract_frames

def load_dataset(dataset_path):
    X = []
    y = []

    labels = sorted(os.listdir(dataset_path))
    word2idx = {word: idx for idx, word in enumerate(labels)}

    for word in labels:
        folder = os.path.join(dataset_path, word)

        for video in os.listdir(folder):
            video_path = os.path.join(folder, video)

            frames = extract_frames(video_path)
            X.append(frames)
            y.append(word2idx[word])

    X = np.array(X)
    y = to_categorical(y, num_classes=len(labels))

    return train_test_split(X, y, test_size=0.2), word2idx