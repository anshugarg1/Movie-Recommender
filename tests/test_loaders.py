import pandas as pd

from recommender.data.loaders import Load_Data


def test_loader_reads_all_sources(tmp_path):
    movies = pd.DataFrame({"movieId": [1], "title": ["A (1995)"], "genres": ["Drama"]})
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [1, 1, 1, 1],
            "rating": [4.0, 5.0, 3.5, 4.5],
            "timestamp": [1, 2, 3, 4],
        }
    )
    tags = pd.DataFrame({"userId": [1], "movieId": [1], "tag": ["great"], "timestamp": [1]})
    links = pd.DataFrame({"movieId": [1], "imdbId": [1], "tmdbId": [1]})

    movies_path = tmp_path / "movies.csv"
    ratings_path = tmp_path / "ratings.csv"
    tags_path = tmp_path / "tags.csv"
    links_path = tmp_path / "links.csv"

    movies.to_csv(movies_path, index=False)
    ratings.to_csv(ratings_path, index=False)
    tags.to_csv(tags_path, index=False)
    links.to_csv(links_path, index=False)

    loader = Load_Data(movies_path, ratings_path, tags_path, links_path)

    assert not loader.load_movie().empty
    assert not loader.load_rating().empty
    assert not loader.load_tag().empty
    assert not loader.load_link().empty


def test_loader_builds_rating_split(tmp_path):
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4],
            "movieId": [1, 2, 1, 2, 1, 2, 1, 2],
            "rating": [4.0, 4.5, 3.0, 3.5, 4.0, 5.0, 2.5, 3.0],
            "timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    dummy = pd.DataFrame({"movieId": [1], "title": ["A (1995)"], "genres": ["Drama"]})
    movies_path = tmp_path / "movies.csv"
    ratings_path = tmp_path / "ratings.csv"
    tags_path = tmp_path / "tags.csv"
    links_path = tmp_path / "links.csv"

    dummy.to_csv(movies_path, index=False)
    ratings.to_csv(ratings_path, index=False)
    pd.DataFrame({"userId": [], "movieId": [], "tag": [], "timestamp": []}).to_csv(
        tags_path, index=False
    )
    pd.DataFrame({"movieId": [], "imdbId": [], "tmdbId": []}).to_csv(links_path, index=False)

    loader = Load_Data(movies_path, ratings_path, tags_path, links_path)
    trainset, testset = loader.load_rating_dataset(test_size=0.25, random_state=42)

    assert trainset.n_ratings > 0
    assert len(testset) > 0
