import pandas as pd
from surprise import Dataset, Reader

from recommender.config import RATINGS_PATH
from recommender.models.algorithms import tune_svd


def main():
    ratings = pd.read_csv(RATINGS_PATH)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    best_rmse, best_params = tune_svd(data)

    print(f"Best CV RMSE: {best_rmse:.4f}")
    print("Best params:")
    for key, value in best_params.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
