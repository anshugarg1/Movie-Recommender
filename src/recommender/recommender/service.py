from .top_n_recomm import get_top_n_for_user
from collections import defaultdict
from typing import List, Dict


class Recommender_Service:
    def __init__(self, algo, trainset, movies_df):
        self.algo = algo
        self.trainset = trainset
        self.movies_df = movies_df

    def recommend_top_n_movie_for_user(self, user_id: int, n: int = 10) -> list[dict]:
        
        return get_top_n_for_user(
            algo= self.algo, 
            trainset= self.trainset,
            
            movies_df= self.movies_df, 
            raw_user_id= user_id, 
            n= n)


    def get_user_rated_movies(self, user_id: int) -> list[dict]:
        """
        Return movies the user has already rated, with their ratings.
        Result is a list of dicts sorted by rating (desc).
        """
        # user_id is the raw userId from ratings.csv
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            # user not present in the training data
            return []

        # self.trainset.ur[inner_uid] -> list of (inner_iid, rating)
        rated_items = self.trainset.ur[inner_uid]

        results: list[dict] = []
        for inner_iid, rating in rated_items:
            raw_movie_id = int(self.trainset.to_raw_iid(inner_iid))

            row = self.movies_df.loc[self.movies_df["movieId"] == raw_movie_id]
            if row.empty:
                title = f"Movie {raw_movie_id}"
                genres = ""
            else:
                title = row["title"].iloc[0]
                genres = row["genres"].iloc[0]

            results.append(
                {
                    "movieId": raw_movie_id,
                    "title": title,
                    "genres": genres,
                    "rating": float(rating),
                }
            )

        # Sort by rating descending (highest rated first)
        results.sort(key=lambda x: x["rating"], reverse=True)
        return results


    def compute_genre_profile(self, user_rated: List[Dict], top_k: int) -> Dict[str, float]:
        """
        Given the list returned by get_user_rated_movies(user_id),
        compute a genre preference profile from the top_k movies.

        Returns a dict: {genre: score}, where score is roughly the
        sum of ratings for that genre (higher = more preferred).
        """
        genre_scores: dict[str, float] = defaultdict(float)

        # Use only the top_k highest rated movies
        top_movies = user_rated[:top_k]

        for movie in top_movies:
            rating = movie.get("rating", 0.0)
            genres_str = movie.get("genres", "")

            if not genres_str:
                continue

            # MovieLens genres are separated by "|"
            genres = [g.strip() for g in genres_str.split("|") if g.strip()]

            # Weight each genre by the rating
            for g in genres:
                genre_scores[g] += float(rating)

        # Sort by score (descending) and return a normal dict
        sorted_genres = dict(
            sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        )
        return sorted_genres


    def compute_genre_profile_from_recs(self, recs: List[Dict]) -> Dict[str, float]:
        """
        Given the list returned by recommend_top_n_movies_for_user (with 'genres'
        and 'predicted_rating'), compute a genre profile for the recommendations.

        Returns a dict: {genre: score}, where score is the sum of predicted ratings
        per genre (higher score = genre appears often and with high predicted rating).
        """
        genre_scores: dict[str, float] = defaultdict(float)

        for movie in recs:
            pred = movie.get("predicted_rating", 0.0)
            genres_str = movie.get("genres", "")

            if not genres_str:
                continue

            genres = [g.strip() for g in genres_str.split("|") if g.strip()]

            for g in genres:
                genre_scores[g] += float(pred)

        sorted_genres = dict(
            sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        )
        return sorted_genres

    def similar_movies(self, movie_id: int, n: int = 10) -> list[dict]:
        """
        Return up to n movies that are most similar to the given movie_id,
        using an item-based KNN model (algo must have a similarity matrix).
        Result is a list of dicts with similarity scores.
        """
        # movie_id is the raw movieId from movies.csv
        try:
            inner_iid = self.trainset.to_inner_iid(movie_id)
        except ValueError:
            # movie not present in training data
            return []

        # We need a KNN-based algo with a similarity matrix
        if not hasattr(self.algo, "get_neighbors") or not hasattr(self.algo, "sim"):
            raise ValueError(
                "similar_movies requires an item-based KNN model "
                "with a similarity matrix (e.g. KNNBasic with user_based=False)."
            )

        # Get nearest neighbors in inner-id space
        neighbor_inner_iids = self.algo.get_neighbors(inner_iid, k=n)

        results: list[dict] = []
        for neigh_inner_iid in neighbor_inner_iids:
            raw_movie_id = int(self.trainset.to_raw_iid(neigh_inner_iid))
            sim_score = float(self.algo.sim[inner_iid, neigh_inner_iid])

            row = self.movies_df.loc[self.movies_df["movieId"] == raw_movie_id]
            if row.empty:
                title = f"Movie {raw_movie_id}"
                genres = ""
            else:
                title = row["title"].iloc[0]
                genres = row["genres"].iloc[0]

            results.append(
                {
                    "movieId": raw_movie_id,
                    "title": title,
                    "genres": genres,
                    "similarity": sim_score,
                }
            )

        # Neighbors should already be sorted by similarity, but just in case:
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

