from recommender.config import LINKS_PATH, MOVIES_PATH, RATINGS_PATH, SVD_MODEL_PATH, TAGS_PATH
from recommender.data.loaders import Load_Data
from recommender.models.training import load_svd_from_disk, train_and_save_svd
from recommender.recommender.service import Recommender_Service


def main():
    loader = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    movies_df, ratings_df, _, _ = loader.load_all_data()
    trainset, _ = loader.load_rating_dataset()

    if not SVD_MODEL_PATH.exists():
        print("Model not found. Training a new SVD model...")
        train_and_save_svd(use_full_trainset=True)

    algo = load_svd_from_disk()
    service = Recommender_Service(
        algo=algo, trainset=trainset, movies_df=movies_df, ratings_df=ratings_df
    )

    user_id = int(input("Enter a userId from ratings.csv: "))
    recs = service.recommend_top_n_movie_for_user(user_id=user_id, n=10)

    if not recs:
        recs = service.cold_start_recommendations(n=10)
        print("No personalized recommendations. Showing cold-start results.")

    print(f"\nTop recommendations for user {user_id}:\n")
    for i, rec in enumerate(recs, start=1):
        print(
            f"{i}. {rec['title']} (movieId={rec['movieId']}, predicted rating={rec['predicted_rating']})"
        )


if __name__ == "__main__":
    main()
