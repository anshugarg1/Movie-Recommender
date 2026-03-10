def get_top_n_for_user(algo, trainset, movies_df, raw_user_id, n=10):
    try:
        inner_uid = trainset.to_inner_uid(raw_user_id)
    except ValueError:
        return []

    user_rated_inner_iids = {iid for (iid, _) in trainset.ur[inner_uid]}
    all_inner_iids = list(trainset.all_items())

    predictions = []
    for inner_iid in all_inner_iids:
        if inner_iid in user_rated_inner_iids:
            continue

        raw_iid = trainset.to_raw_iid(inner_iid)
        predictions.append(algo.predict(raw_user_id, raw_iid))

    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:n]

    results = []
    for pred in top_predictions:
        movie_id = int(pred.iid)
        row = movies_df.loc[movies_df["movieId"] == movie_id]

        if row.empty:
            title = f"Movie {movie_id}"
            genres = ""
        else:
            title = row["title"].iloc[0]
            genres = row["genres"].iloc[0]

        results.append(
            {
                "movieId": movie_id,
                "title": title,
                "predicted_rating": round(float(pred.est), 2),
                "genres": genres,
            }
        )

    return results
