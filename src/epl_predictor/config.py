
from pathlib import Path


# Main project folder
ROOT = Path(__file__).resolve().parents[2]

# Data folders
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"

# Trained models
MODELS_DIR = ROOT / "models"

# Historical seasons
FIRST_SEASON = 2014
LAST_SEASON = 2025

# Evaluation periods
VALIDATION_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

TEST_SEASON = "2025-26"

# Random seed for reproducibility
RANDOM_SEED = 42
