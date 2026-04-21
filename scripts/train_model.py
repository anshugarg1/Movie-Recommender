from surprise import KNNBasic, SVD, dump

from recommender.config import KNN_MODEL_PATH, LINKS_PATH, MOVIES_PATH, RATINGS_PATH, TAGS_PATH
from recommender.data.loaders import Load_Data
from recommender.models.training import train_and_save_svd


def train_and_save_model() -> SVD:
    return train_and_save_svd(use_full_trainset=True)


def train_and_save_knn():
    loader = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    trainset = loader.load_full_trainset()
    algo = KNNBasic(sim_options={"name": "cosine", "user_based": False})
    algo.fit(trainset)
    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump.dump(str(KNN_MODEL_PATH), algo=algo)
    return algo


def main():
    print("Training SVD model on full ratings dataset...")
    train_and_save_model()
    print("SVD model saved.")

    print("Training KNN item-based model...")
    train_and_save_knn()
    print("KNN model saved.")


if __name__ == "__main__":
    main()
