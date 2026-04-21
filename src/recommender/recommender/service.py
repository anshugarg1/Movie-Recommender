from collections import defaultdict
from typing import Dict, List

from .top_n_recomm import get_top_n_for_user


class Recommender_Service:
    def __init__(self, algo, trainset, movies_df, ratings_df):
        self.algo = algo
        self.trainset = trainset
        self.movies_df = movies_df
        self.ratings_df = ratings_df

    def recommend_top_n_movie_for_user(self, user_id: int, n: int = 10) -> list[dict]:
        return get_top_n_for_user(
            algo=self.algo,
            trainset=self.trainset,
            movies_df=self.movies_df,
            raw_user_id=user_id,
            n=n,
        )

    def get_user_rated_movies(self, user_id: int) -> list[dict]:
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []

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

        results.sort(key=lambda x: x["rating"], reverse=True)
        return results

    def compute_genre_profile(self, user_rated: List[Dict], top_k: int) -> Dict[str, float]:
        genre_scores: dict[str, float] = defaultdict(float)
        for movie in user_rated[:top_k]:
            rating = float(movie.get("rating", 0.0))
            genres = [g.strip() for g in movie.get("genres", "").split("|") if g.strip()]
            for genre in genres:
                genre_scores[genre] += rating

        return dict(sorted(genre_scores.items(), key=lambda x: x[1], reverse=True))

    def compute_genre_profile_from_recs(self, recs: List[Dict]) -> Dict[str, float]:
        genre_scores: dict[str, float] = defaultdict(float)
        for movie in recs:
            pred = float(movie.get("predicted_rating", 0.0))
            genres = [g.strip() for g in movie.get("genres", "").split("|") if g.strip()]
            for genre in genres:
                genre_scores[genre] += pred

        return dict(sorted(genre_scores.items(), key=lambda x: x[1], reverse=True))

    def filter_recommendations(
        self, recs: List[Dict], include_genres: list[str], year_range: tuple[int, int]
    ) -> List[Dict]:
        if not recs:
            return []

        min_year, max_year = year_range
        filtered = []
        for rec in recs:
            title = rec.get("title", "")
            year = self._extract_year_from_title(title)
            if year is None or year < min_year or year > max_year:
                continue

            if include_genres:
                genres = {g.strip() for g in rec.get("genres", "").split("|") if g.strip()}
                if not genres.intersection(set(include_genres)):
                    continue

            filtered.append(rec)

        return filtered

    def cold_start_recommendations(
        self,
        n: int = 10,
        include_genres: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> List[Dict]:
        include_genres = include_genres or []
        if year_range is None:
            years = self._all_movie_years()
            year_range = (min(years), max(years)) if years else (1900, 2100)

        agg = (
            self.ratings_df.groupby("movieId")
            .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
            .reset_index()
        )

        ranked = self.movies_df.merge(agg, on="movieId", how="left").fillna(
            {"avg_rating": 0.0, "rating_count": 0}
        )
        ranked["popularity_score"] = ranked["avg_rating"] * (ranked["rating_count"] ** 0.2)
        ranked = ranked.sort_values(
            by=["popularity_score", "rating_count", "avg_rating"],
            ascending=[False, False, False],
        )

        recs = [
            {
                "movieId": int(row.movieId),
                "title": row.title,
                "genres": row.genres,
                "predicted_rating": round(float(row.avg_rating), 2),
            }
            for row in ranked.itertuples(index=False)
        ]

        filtered = self.filter_recommendations(recs, include_genres, year_range)
        return filtered[:n]

    def explain_recommendation(self, rec: Dict, user_genre_profile: Dict[str, float]) -> str:
        rec_genres = [g.strip() for g in rec.get("genres", "").split("|") if g.strip()]
        matched = [g for g in rec_genres if g in user_genre_profile]
        if matched:
            top_match = matched[:2]
            return f"Matches your preference for {', '.join(top_match)}."
        return "High predicted rating from the collaborative filtering model."

    def _all_movie_years(self) -> List[int]:
        years = []
        for title in self.movies_df["title"].astype(str).tolist():
            year = self._extract_year_from_title(title)
            if year is not None:
                years.append(year)
        return years

    def similar_movies(self, movie_id: int, n: int = 10) -> list[dict]:
        try:
            inner_iid = self.trainset.to_inner_iid(movie_id)
        except ValueError:
            return []

        if not hasattr(self.algo, "get_neighbors") or not hasattr(self.algo, "sim"):
            raise ValueError(
                "similar_movies requires an item-based KNN model "
                "with a similarity matrix (e.g. KNNBasic with user_based=False)."
            )

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

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    @staticmethod
    def _extract_year_from_title(title: str) -> int | None:
        if len(title) < 6:
            return None
        if title[-1] == ")" and title[-6] == "(" and title[-5:-1].isdigit():
            return int(title[-5:-1])
        return None
