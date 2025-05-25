#!/bin/bash

# Download script for Soil Classification dataset from Kaggle

# Dataset slug on Kaggle (change as needed)
KAGGLE_DATASET="annam-ai/soilclassification-part-2"

# Target directory to save and unzip dataset
TARGET_DIR="./data"

echo "Downloading dataset: $KAGGLE_DATASET"
mkdir -p "$TARGET_DIR"

# Download and unzip the dataset
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
