from pathlib import Path
from surprise import SVD, dump

from recommender.config import SVD_MODEL_PATH, MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH
from recommender.data.loaders import Load_Data

def train_and_save_model() -> SVD:
    obj_load_data = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    trainset, testset = obj_load_data.load_rating_dataset()

    algo = SVD(random_state = 42)
    algo.fit(trainset)

    SVD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    dump.dump(str(SVD_MODEL_PATH), algo=algo)

    return algo


def load_model() -> SVD:
 
    if not SVD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {SVD_MODEL_PATH}")

    _, algo = dump.load(str(SVD_MODEL_PATH))
    return algo


def main():
    print("Training SVD model on full ratings dataset...")
    algo = train_and_save_model()
    print("✅ Training finished.")
    print("✅ Model saved.")

if __name__ == "__main__":
    main()