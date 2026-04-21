from src.recommender.data.loaders import Load_Data
from src.recommender.models.algorithms import cf_svd
from src.recommender.recommender.top_n_recomm import get_top_n_for_user

movie_path = './data/ml-latest-small/ml-latest-small/movies.csv'
rating_path = './data/ml-latest-small/ml-latest-small/ratings.csv'
tag_path = './data/ml-latest-small/ml-latest-small/tags.csv'
link_path = './data/ml-latest-small/ml-latest-small/links.csv'

def main():
    print("Hello from recommender-system!")
    loader = Load_Data(movie_path, rating_path, tag_path, link_path)
    movie_data = loader.load_movie()
    trainset, testset = loader.load_rating_dataset()
    pred_svd, algo_svd = cf_svd(trainset, testset)

    user_id = int(input("Enter a userId (e.g. 1–610 from ratings.csv): "))
    top_n = get_top_n_for_user(algo_svd, trainset, movie_data, user_id, n=10)
    
    print(f"\nTop 10 recommendations for user {user_id}:\n")
    for i, rec in enumerate(top_n, start=1):
        print(f"{i}. {rec['title']}  (movieId={rec['movieId']}, predicted rating={rec['pred_rating']})")


if __name__ == "__main__":
    main()
