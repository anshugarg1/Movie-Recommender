from surprise import SVD, dump

from recommender.config import SVD_MODEL_PATH, KNN_MODEL_PATH, MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH
from recommender.data.loaders import Load_Data
from recommender.models.algorithms import cf_svd, cf_item_based


def train_and_save_svd() -> SVD:
    obj_load_data = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    trainset, testset = obj_load_data.load_rating_dataset()
    _, algo = cf_svd(trainset, testset)
    SVD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump.dump(str(SVD_MODEL_PATH), algo=algo)
    return algo


def train_and_save_knn():
    obj_load_data = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    trainset, testset = obj_load_data.load_rating_dataset()
    pred_item = cf_item_based(trainset, testset)
    # cf_item_based fits and tests; we need the fitted algo — refactor to return it
    from surprise import KNNBasic
    sim_options = {"name": "cosine", "user_based": False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    KNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump.dump(str(KNN_MODEL_PATH), algo=algo)
    return algo


def main():
    print("Training SVD model...")
    train_and_save_svd()
    print("SVD model saved.")

    print("Training KNN item-based model...")
    train_and_save_knn()
    print("KNN model saved.")


if __name__ == "__main__":
    main()
