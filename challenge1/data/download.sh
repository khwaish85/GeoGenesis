#!/bin/bash

# Download script for Soil Image Classification Part 1 dataset from Kaggle

# Dataset slug on Kaggle (update as needed)
KAGGLE_DATASET="annam-ai/soilclassification"

# Target directory to save and unzip dataset
TARGET_DIR="./data_part1"

echo "Downloading dataset: $KAGGLE_DATASET"
mkdir -p "$TARGET_DIR"

# Download and unzip the dataset
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
