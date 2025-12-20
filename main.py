from src.recommender.data.loaders import load_dataset, load_data, create_anti_test_set
from src.recommender.models.algorithms import cf_item_based, cf_user_based, cf_svd
from src.recommender.evaluation.metrics import eval_res, precision_at_k
from src.recommender.recommender.top_n_recomm import get_top_n_for_user

def main():
    print("Hello from recommender-system!")
    movie_data, rating_data, tag_data, link_data = load_data()
    trainset, testset = load_dataset(rating_data)
    # pred_user = cf_user_based(trainset, testset)
    # pred_item = cf_item_based(trainset, testset)
    pred_svd, algo_svd = cf_svd(trainset, testset)

    # eval_res(pred_user, pred_item, pred_svd)

    # print("CR used based Precision@K: ", precision_at_k(pred_user, k=10))
    # print("CR item based Precision@K: ", precision_at_k(pred_item, k=10))
    # print("SVD Precision@K: ", precision_at_k(pred_svd, k=10))

    # Simple CLI: ask for a user id
    user_id = int(input("Enter a userId (e.g. 1–610 from ratings.csv): "))
    
    # create_anti_test_set(trainset)
    top_n = get_top_n_for_user(algo_svd, trainset, movie_data, user_id, n=10)
    
    print(f"\nTop 10 recommendations for user {user_id}:\n")
    for i, rec in enumerate(top_n, start=1):
        print(f"{i}. {rec['title']}  (movieId={rec['movieId']}, predicted rating={rec['pred_rating']})")


if __name__ == "__main__":
    main()
