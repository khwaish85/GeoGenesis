"""
Author: Annam.ai IIT Ropar
Team Name: GeoGenesis
Team Members: Khwaish Yadav, Hemant, Sparsh Patidar, Smarth Tripathi, Sai Pradeep
Leaderboard Rank: 86
"""

import pandas as pd
import os
import numpy as np
from src.preprocessing import load_image, IMAGE_SIZE, SUPPORTED_EXTS
from tensorflow.keras.models import load_model

def run_inference(model, test_df, test_dir):
    results = []
    for row in test_df.itertuples():
        img_path = os.path.join(test_dir, row.image_id)
        if os.path.exists(img_path) and os.path.splitext(img_path)[1].lower() in SUPPORTED_EXTS:
            img = load_image(img_path)
            if img is not None:
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                pred = (model.predict(img)[0][0] > 0.5)
                results.append([row.image_id, int(pred)])
                continue
        results.append([row.image_id, 0])  # default to non-soil
    return pd.DataFrame(results, columns=["image_id", "predicted_label"])

def save_submission(df, output_csv):
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
