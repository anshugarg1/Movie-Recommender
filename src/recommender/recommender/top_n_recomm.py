import pandas as pd
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader

def get_top_n_for_user(algo, trainset, movies_df, raw_user_id, n=10):
    """
    algo        = trained Surprise algorithm (SVD, KNN, etc.)
    trainset    = Surprise trainset
    movies_df   = movies.csv as a DataFrame
    raw_user_id = userId as it appears in ratings.csv (e.g. 1, 2, 3...)
    n           = how many recommendations to return
    """

    # Convert to Surprise's internal user id
    inner_uid = trainset.to_inner_uid(raw_user_id)

    # All items user has already rated (inner ids) - movie inner ids
    user_rated_inner_iids = set(j for (j, _) in trainset.ur[inner_uid])

    # All item (movie) inner ids
    all_inner_iids = list(trainset.all_items())

    predictions = []

    # Predict for all items the user has NOT rated
    for inner_iid in all_inner_iids:
        if inner_iid in user_rated_inner_iids:
            continue  # skip already rated movies

        # from movie inner id to raw id
        raw_iid = trainset.to_raw_iid(inner_iid)  # back to original movieId
        pred = algo.predict(raw_user_id, raw_iid)
        predictions.append(pred)

    # Sort by estimated rating, descending
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:n]

    # Build a nice list with titles
    results = []
    for p in top_predictions:
        movie_id = int(p.iid)
        # look up title in movies_df
        title_row = movies_df.loc[movies_df['movieId'] == movie_id, 'title']
        title = title_row.iloc[0] if not title_row.empty else f"Movie {movie_id}"

        genre_row = movies_df.loc[movies_df['movieId'] == movie_id, 'genres']
        genre = genre_row.iloc[0] if not genre_row.empty else f"Movie {movie_id}"
        results.append({
            "movieId": movie_id,
            "title": title,
            "pred_rating": round(p.est, 2),
            "genre": genre
        })
    
    return results


    def get_user_rated_movies(user_id: int, trainset: Dataset):
        user_inner_id = trainset.to_inner_uid(user_id)
        user_movie_id = trainset

