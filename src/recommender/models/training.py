from surprise import SVD, dump

from recommender.config import LINKS_PATH, MOVIES_PATH, RATINGS_PATH, SVD_MODEL_PATH, TAGS_PATH
from recommender.data.loaders import Load_Data


def train_and_save_svd(use_full_trainset=True) -> SVD:
    loader = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)

    if use_full_trainset:
        trainset = loader.load_full_trainset()
    else:
        trainset, _ = loader.load_rating_dataset()

    algo = SVD(random_state=42)
    algo.fit(trainset)

    SVD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump.dump(str(SVD_MODEL_PATH), algo=algo)
    return algo


def load_svd_from_disk() -> SVD:
    if not SVD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {SVD_MODEL_PATH}")

    _, algo = dump.load(str(SVD_MODEL_PATH))
    return algo
