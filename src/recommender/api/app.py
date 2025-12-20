import pandas as pd
import streamlit as st

from recommender.config import MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH,SVD_MODEL_PATH
from recommender.data.loaders import Load_Data
from recommender.models.persistence import Model_Store
from recommender.recommender.service import Recommender_Service


@st.cache_data
def get_raw_data():
    """Load movies and ratings once and cache them."""
    obj_load_data = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    movies_df = obj_load_data.load_movie()
    ratings_df = obj_load_data.load_rating()
    return movies_df, ratings_df

@st.cache_resource
def get_service():
    """
    Create and cache the RecommenderService.
    This ensures the model is loaded only once per session.
    """
    movies_df, ratings_df = get_raw_data()
    obj_model_store = Model_Store(SVD_MODEL_PATH)
    algo = obj_model_store.load_model()
    rating_trainset = Load_Data.load_rating_dataset()

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

    st.write(
        f"Showing top **{n_recs}** recommendations for user **{selected_user}** "
        "(based on SVD model)."
    )

    if st.button("Get Recommendations"):
        recs = service.recommend_top_n_movie_for_user(algo, movies_df, selected_user, n=n_recs)

        if not recs:
            st.warning("No recommendations available for this user (maybe unknown user).")
            return

        recs_df = pd.DataFrame(recs)
        st.subheader("Recommended movies")
        st.dataframe(recs_df[["title", "genres", "predicted_rating"]])


if __name__ == "__main__":
    run_app()