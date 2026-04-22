import pandas as pd
import streamlit as st

from recommender.config import (
    KNN_MODEL_PATH,
    LINKS_PATH,
    MOVIES_PATH,
    RATINGS_PATH,
    SVD_MODEL_PATH,
    TAGS_PATH,
)
from recommender.data.loaders import Load_Data
from recommender.models.persistence import Model_Store
from recommender.models.training import load_svd_from_disk, train_and_save_svd
from recommender.recommender.service import Recommender_Service


@st.cache_data
def get_raw_data(_loader: Load_Data):
    movies_df = _loader.load_movie()
    ratings_df = _loader.load_rating()
    return movies_df, ratings_df


@st.cache_resource
def get_service():
    loader = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    movies_df, ratings_df = get_raw_data(loader)

    if not SVD_MODEL_PATH.exists():
        train_and_save_svd(use_full_trainset=True)

    algo = load_svd_from_disk()
    rating_trainset, _ = loader.load_rating_dataset()

    service = Recommender_Service(
        algo=algo,
        trainset=rating_trainset,
        movies_df=movies_df,
        ratings_df=ratings_df,
    )

    knn_service = None
    if KNN_MODEL_PATH.exists():
        knn_algo = Model_Store(KNN_MODEL_PATH).load_model()
        knn_service = Recommender_Service(
            algo=knn_algo, trainset=rating_trainset, movies_df=movies_df, ratings_df=ratings_df
        )

    return service, knn_service, ratings_df, movies_df


def _collect_genres(movies_df: pd.DataFrame) -> list[str]:
    genres = set()
    for raw in movies_df["genres"].fillna("").astype(str):
        for genre in raw.split("|"):
            genre = genre.strip()
            if genre:
                genres.add(genre)
    return sorted(genres)


def run_app():
    st.title("Movie Recommender (SVD + MovieLens)")
    st.sidebar.header("Settings")

    with st.spinner("Loading data and model..."):
        service, knn_service, ratings_df, movies_df = get_service()

    user_ids = sorted(ratings_df["userId"].unique())
    selected_user = st.sidebar.selectbox("Select userId", user_ids, index=0)

    n_recs = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10)
    n_history = st.sidebar.slider(
        "Top-rated movies to inspect", min_value=5, max_value=30, value=10
    )

    all_genres = _collect_genres(movies_df)
    selected_genres = st.sidebar.multiselect("Filter by genres", all_genres, default=[])

    years = service._all_movie_years()
    if years:
        min_year, max_year = min(years), max(years)
    else:
        min_year, max_year = 1900, 2100
    selected_year_range = st.sidebar.slider(
        "Release year range", min_value=min_year, max_value=max_year, value=(min_year, max_year)
    )

    st.write(f"Showing up to **{n_recs}** recommendations for user **{selected_user}**.")

    user_rated = service.get_user_rated_movies(selected_user)
    genre_profile = {}

    if user_rated:
        st.subheader(f"User {selected_user}: top rated movies")
        user_rated_df = pd.DataFrame(user_rated[:n_history])
        st.dataframe(user_rated_df[["title", "genres", "rating"]], width="stretch")

        genre_profile = service.compute_genre_profile(user_rated, top_k=n_history)
        if genre_profile:
            st.subheader("Preferred genres")
            st.markdown(f"Top genres: {', '.join(list(genre_profile.keys())[:4])}")
            genre_df = pd.DataFrame(
                {"genre": list(genre_profile.keys()), "score": list(genre_profile.values())}
            )
            st.bar_chart(genre_df.set_index("genre"))
    else:
        st.info("User not in the model trainset. Cold-start recommendations will be used.")

    if st.button("Get Recommendations"):
        recs = service.recommend_top_n_movie_for_user(user_id=selected_user, n=max(2 * n_recs, 20))
        recs = service.filter_recommendations(recs, selected_genres, selected_year_range)

        if not recs:
            recs = service.cold_start_recommendations(
                n=n_recs, include_genres=selected_genres, year_range=selected_year_range
            )
            st.warning("Used cold-start recommendations (popular movies).")
        else:
            recs = recs[:n_recs]

        if not recs:
            st.warning("No recommendations for the current filters.")
            return

        for rec in recs:
            rec["why"] = service.explain_recommendation(rec, genre_profile)

        recs_df = pd.DataFrame(recs)
        st.subheader("Recommended movies")
        st.dataframe(
            recs_df[["title", "genres", "predicted_rating", "why"]], width="stretch"
        )

        rec_genre_profile = service.compute_genre_profile_from_recs(recs)
        if rec_genre_profile:
            st.subheader("Genres in recommendations")
            rec_genre_df = pd.DataFrame(
                {"genre": list(rec_genre_profile.keys()), "score": list(rec_genre_profile.values())}
            )
            st.bar_chart(rec_genre_df.set_index("genre"))

    # --- Similar Movies ---
    st.divider()
    st.subheader("Find Similar Movies")

    if knn_service is None:
        st.warning("KNN model not found. Run `scripts/train_model.py` to enable this feature.")
    else:
        movie_titles = sorted(movies_df["title"].tolist())
        selected_title = st.selectbox("Select a movie", movie_titles)
        n_similar = st.slider("Number of similar movies", min_value=3, max_value=20, value=10)

        if st.button("Find Similar"):
            movie_id = int(movies_df.loc[movies_df["title"] == selected_title, "movieId"].iloc[0])
            similar = knn_service.similar_movies(movie_id=movie_id, n=n_similar)

            if not similar:
                st.info("No similar movies found (movie may not be in training data).")
            else:
                st.dataframe(
                    pd.DataFrame(similar)[["title", "genres", "similarity"]],
                    width="stretch",
                )


if __name__ == "__main__":
    run_app()
