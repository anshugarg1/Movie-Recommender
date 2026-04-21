import pandas as pd
import streamlit as st

from recommender.config import MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH, SVD_MODEL_PATH, KNN_MODEL_PATH
from recommender.data.loaders import Load_Data
from recommender.models.persistence import Model_Store
from recommender.recommender.service import Recommender_Service


@st.cache_data
def get_raw_data(_obj_load_data: Load_Data):
    movies_df = _obj_load_data.load_movie()
    ratings_df = _obj_load_data.load_rating()
    return movies_df, ratings_df

@st.cache_resource
def get_service():
    obj_load_data = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    movies_df, ratings_df = get_raw_data(obj_load_data)
    rating_trainset, _ = obj_load_data.load_rating_dataset()

    svd_algo = Model_Store(SVD_MODEL_PATH).load_model()
    svd_service = Recommender_Service(algo=svd_algo, trainset=rating_trainset, movies_df=movies_df)

    knn_service = None
    if KNN_MODEL_PATH.exists():
        knn_algo = Model_Store(KNN_MODEL_PATH).load_model()
        knn_service = Recommender_Service(algo=knn_algo, trainset=rating_trainset, movies_df=movies_df)

    return svd_service, knn_service, ratings_df, movies_df


def run_app():
    svd_service, knn_service, ratings_df, movies_df = get_service()

    st.title("Movie Recommender (SVD, MovieLens)")
    st.sidebar.header("Settings")

    user_ids = sorted(ratings_df["userId"].unique())
    selected_user = st.sidebar.selectbox("Select userId", user_ids, index=0)
    n_recs = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10)
    n_history = st.sidebar.slider("Top-rated movies to show", min_value=5, max_value=30, value=10)

    st.write(f"Showing top **{n_recs}** recommendations for user **{selected_user}**.")

    # --- User's top-rated movies ---
    user_rated = svd_service.get_user_rated_movies(selected_user)

    if user_rated:
        st.subheader(f"User {selected_user}: top rated movies")
        user_rated_df = pd.DataFrame(user_rated[:n_history])
        st.dataframe(user_rated_df[["title", "genres", "rating"]])

        genre_profile = svd_service.compute_genre_profile(user_rated, top_k=n_history)
        if genre_profile:
            st.subheader("User's preferred genres")
            top_genres = ", ".join(list(genre_profile.keys())[:4])
            st.markdown(f"**Most liked genres:** {top_genres}")
            genre_df = (
                pd.DataFrame({"genre": list(genre_profile.keys()), "score": list(genre_profile.values())})
                .sort_values("score", ascending=False)
            )
            st.bar_chart(genre_df.set_index("genre"))
        else:
            st.info("No genre information available for this user's top movies.")
    else:
        st.info("This user has no ratings in the training data.")

    # --- Recommendations ---
    if st.button("Get Recommendations"):
        recs = svd_service.recommend_top_n_movie_for_user(user_id=selected_user, n=n_recs)

        if not recs:
            st.warning("No recommendations available for this user.")
            return

        st.subheader("Recommended movies")
        st.dataframe(pd.DataFrame(recs)[["title", "genre", "pred_rating"]])

        rec_genre_profile = svd_service.compute_genre_profile_from_recs(recs)
        if rec_genre_profile:
            st.subheader("Genres in recommended movies")
            rec_genre_df = (
                pd.DataFrame({"genre": list(rec_genre_profile.keys()), "score": list(rec_genre_profile.values())})
                .sort_values("score", ascending=False)
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
                st.dataframe(pd.DataFrame(similar)[["title", "genres", "similarity"]])


if __name__ == "__main__":
    run_app()
