import pandas as pd

from recommender.recommender.service import Recommender_Service


class DummyTrainset:
    def to_inner_uid(self, user_id):
        raise ValueError("unused in this test")


def _service():
    movies_df = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Toy Story (1995)", "Heat (1995)", "Arrival (2016)"],
            "genres": ["Adventure|Animation", "Action|Crime", "Drama|Sci-Fi"],
        }
    )
    ratings_df = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4, 1, 2],
            "movieId": [1, 1, 2, 2, 3, 3],
            "rating": [4.0, 5.0, 3.5, 4.0, 4.5, 5.0],
        }
    )
    return Recommender_Service(
        algo=None, trainset=DummyTrainset(), movies_df=movies_df, ratings_df=ratings_df
    )


def test_compute_genre_profile_from_recs_uses_predicted_rating():
    service = _service()
    recs = [
        {"title": "A", "genres": "Drama|Action", "predicted_rating": 4.5},
        {"title": "B", "genres": "Drama", "predicted_rating": 4.0},
    ]

    profile = service.compute_genre_profile_from_recs(recs)

    assert profile["Drama"] == 8.5
    assert profile["Action"] == 4.5


def test_filter_recommendations_by_genre_and_year():
    service = _service()
    recs = [
        {"title": "Toy Story (1995)", "genres": "Adventure|Animation", "predicted_rating": 4.8},
        {"title": "Arrival (2016)", "genres": "Drama|Sci-Fi", "predicted_rating": 4.7},
    ]

    filtered = service.filter_recommendations(
        recs, include_genres=["Drama"], year_range=(2000, 2020)
    )

    assert len(filtered) == 1
    assert filtered[0]["title"] == "Arrival (2016)"


def test_cold_start_recommendations_schema():
    service = _service()
    recs = service.cold_start_recommendations(
        n=2, include_genres=["Drama"], year_range=(2000, 2020)
    )

    assert len(recs) <= 2
    for rec in recs:
        assert "movieId" in rec
        assert "title" in rec
        assert "genres" in rec
        assert "predicted_rating" in rec
