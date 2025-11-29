import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = r"C:\vishanthkumar\phishshield\data\raw\phishing_site_urls.csv"

# Columns in the CSV (adjust these if needed)
URL_COLUMN = "URL"
LABEL_COLUMN = "Label"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Where to save the trained model
MODEL_PATH = Path(r"C:\vishanthkumar\phishshield\models\phishing_model.joblib")
