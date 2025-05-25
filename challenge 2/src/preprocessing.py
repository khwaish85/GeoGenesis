"""
Author: Annam.ai IIT Ropar
Team Name: GeoGenesis
Team Members: Khwaish Yadav, Hemant, Sparsh Patidar, Smarth Tripathi, Sai Pradeep
Leaderboard Rank: 86
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = (128, 128)
SUPPORTED_EXTS = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.ipeg']

def load_image(path, size=IMAGE_SIZE):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".gif":
            img = Image.open(path).convert('RGB').resize(size)
            return np.array(img)
        img = cv2.imread(path)
        if img is None: 
            return None
        return cv2.resize(img, size)
    except:
        return None

def preprocess_training_data(train_csv_path, train_dir):
    df = pd.read_csv(train_csv_path)
    df['label'] = 1  # soil class
    X, y = [], []

    for row in df.itertuples():
        img_path = os.path.join(train_dir, row.image_id)
        if os.path.exists(img_path) and os.path.splitext(img_path)[1].lower() in SUPPORTED_EXTS:
            img = load_image(img_path)
            if img is not None:
                X.append(img)
                y.append(row.label)

    for i in range(len(X) // 2):  # Flip half of images
        flipped = cv2.flip(X[i], 1)
        X.append(flipped)
        y.append(0)

    return np.array(X, dtype='float32') / 255.0, np.array(y)
