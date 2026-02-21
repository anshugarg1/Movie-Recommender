import pandas as pd
import streamlit as st

from recommender.config import MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH,SVD_MODEL_PATH
from recommender.data.loaders import Load_Data
from recommender.models.persistence import Model_Store
from recommender.recommender.service import Recommender_Service


@st.cache_data
def get_raw_data(_obj_load_data: Load_Data):
    """Load movies and ratings once and cache them."""
    movies_df = _obj_load_data.load_movie()
    ratings_df = _obj_load_data.load_rating()
    return movies_df, ratings_df

@st.cache_resource
def get_service():
    """
    Create and cache the RecommenderService.
    This ensures the model is loaded only once per session.
    """
    obj_load_data = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    movies_df, ratings_df = get_raw_data(obj_load_data)
    obj_model_store = Model_Store(SVD_MODEL_PATH)
    algo = obj_model_store.load_model()
    rating_trainset, rating_testset = obj_load_data.load_rating_dataset()

    service = Recommender_Service(
        algo=algo,
        trainset=rating_trainset,
        movies_df=movies_df,
    )
    return service, ratings_df, algo, movies_df

# ---------- UI logic ----------
def run_app():
    service, ratings_df, algo, movies_df = get_service()

    st.title("🎬 Movie Recommender (SVD, MovieLens)")
    st.sidebar.header("Settings")

    # Available users from ratings data
    user_ids = sorted(ratings_df["userId"].unique())
    default_user = user_ids[0]

    selected_user = st.sidebar.selectbox(
        "Select userId", user_ids, index=user_ids.index(default_user)
    )

    n_recs = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10)
    print(n_recs)

    n_history = st.sidebar.slider(
        "Number of user's top-rated movies to show",
        min_value=5,
        max_value=30,
        value=10,
    )

    st.write(
        f"Showing top **{n_recs}** recommendations for user **{selected_user}** "
        "(based on SVD model)."
    )

    # --- User's own top-rated movies ---
    user_rated = service.get_user_rated_movies(selected_user)

    if user_rated:
        st.subheader(f"⭐ User {selected_user}: top rated movies are:")
        user_rated_df = pd.DataFrame(user_rated[:n_history])
        st.dataframe(user_rated_df[["title", "genres", "rating"]])
                # --- Genre preference based on top-rated movies ---
        genre_profile = service.compute_genre_profile(user_rated, top_k=n_history)

        if genre_profile:
            st.subheader("🎭 User's preferred genres (based on top-rated movies)")

            # Show as text, e.g. "Drama, Action, Thriller..."
            top_genres = list(genre_profile.keys())
            pretty_genres = ", ".join(top_genres[:4])
            st.markdown(f"**Most liked genres:** {pretty_genres}")

            # Also show a small bar chart for more detail
            genre_df = (
                pd.DataFrame(
                    {"genre": list(genre_profile.keys()),
                     "score": list(genre_profile.values())}
                )
                .sort_values("score", ascending=False)
            )
            st.bar_chart(
                genre_df.set_index("genre")  # Streamlit takes index as x-axis
            )
        else:
            st.info("No genre information available for this user's top movies.")

    else:
        st.info("This user has no ratings in the training data.")

    if st.button("Get Recommendations"):
        recs = service.recommend_top_n_movie_for_user(user_id=selected_user, n=n_recs)

        if not recs:
            st.warning("No recommendations available for this user (maybe unknown user).")
            return

        recs_df = pd.DataFrame(recs)
        st.subheader("Recommended movies")
        st.dataframe(recs_df[["title", "genre", "pred_rating"]])

        # --- Genre profile of recommended movies ---
        rec_genre_profile = service.compute_genre_profile_from_recs(recs)

        if rec_genre_profile:
            st.subheader("📊 Genres in recommended movies")

            rec_genre_df = (
                pd.DataFrame(
                    {
                        "genre": list(rec_genre_profile.keys()),
                        "score": list(rec_genre_profile.values()),
                    }
                )
                .sort_values("score", ascending=False)
            )

            st.bar_chart(rec_genre_df.set_index("genre"))
        else:
            st.info("No genre information available for recommended movies.")


if __name__ == "__main__":
    run_app()
