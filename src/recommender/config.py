from pathlib import Path
import os

# Project root (…/Recommender System)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base data directory
DATA_DIR = Path(os.getenv("RECSYS_DATA_DIR", PROJECT_ROOT / "data"))

# Raw MovieLens files
RAW_DATA_DIR = DATA_DIR / "ml-latest-small"/ "ml-latest-small"

MOVIES_PATH = RAW_DATA_DIR / "movies.csv"
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
TAGS_PATH = RAW_DATA_DIR / "tags.csv"
LINKS_PATH = RAW_DATA_DIR / "links.csv"

# Where trained models are stored
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SVD_MODEL_PATH = MODELS_DIR / "svd_model.dump"
KNN_MODEL_PATH = MODELS_DIR / "knn_item_model.dump"