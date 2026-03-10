from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


class Load_Data:
    def __init__(self, movie_path, rating_path, tag_path, link_path):
        self.movie_path = Path(movie_path)
        self.rating_path = Path(rating_path)
        self.tag_path = Path(tag_path)
        self.link_path = Path(link_path)

    def load_movie(self):
        return pd.read_csv(self.movie_path)

    def load_rating(self):
        return pd.read_csv(self.rating_path)

    def load_tag(self):
        return pd.read_csv(self.tag_path)

    def load_link(self):
        return pd.read_csv(self.link_path)

    def load_all_data(self):
        return (
            self.load_movie(),
            self.load_rating(),
            self.load_tag(),
            self.load_link(),
        )

    def load_rating_dataset(self, test_size=0.25, random_state=42):
        reader = Reader(rating_scale=(1, 5))
        ratings = self.load_rating()[["userId", "movieId", "rating"]]
        data = Dataset.load_from_df(ratings, reader)
        trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
        return trainset, testset

    def load_full_trainset(self):
        reader = Reader(rating_scale=(1, 5))
        ratings = self.load_rating()[["userId", "movieId", "rating"]]
        data = Dataset.load_from_df(ratings, reader)
        return data.build_full_trainset()
