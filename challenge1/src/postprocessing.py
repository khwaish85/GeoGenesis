"""
Author: Annam.ai IIT Ropar
Team Name: GeoGenesis
Team Members: Khwaish Yadav, Hemant, Sparsh Patidar, Smarth Tripathi, Sai Pradeep
Leaderboard Rank: 86

This file handles all the postprocessing steps for the Soil Image Classification Challenge.
"""

import pandas as pd

# Decode predicted labels
def decode_predictions(preds, idx2label):
    return [idx2label[p] for p in preds]

# Create submission file
def create_submission(image_ids, preds, idx2label, output_file="submission.csv"):
    decoded = decode_predictions(preds, idx2label)
    df = pd.DataFrame({
        'image_id': image_ids,
        'soil_type': decoded
    })
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


def postprocessing():
    print("This is the file for postprocessing")
    return 0
