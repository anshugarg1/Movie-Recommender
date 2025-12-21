import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

movie_path = './data/ml-latest-small/ml-latest-small/movies.csv'
rating_path = './data/ml-latest-small/ml-latest-small/ratings.csv'
tag_path = './data/ml-latest-small/ml-latest-small/tags.csv'
link_path = './data/ml-latest-small/ml-latest-small/links.csv'

class Load_Data:
    def __init__(self, movie_path, rating_path, tag_path, link_path):
        self.movie_path = movie_path
        self.rating_path = rating_path
        self.tag_path = tag_path
        self.link_path = link_path

    def load_movie(self):    
        movie_data = pd.read_csv(self.movie_path)
        print(movie_data[:4])
        return movie_data

    def load_rating(self):
        rating_data = pd.read_csv(self.rating_path)
        print(rating_data[:4])
        return rating_data
    
    def load_tag(self):
        tag_data = pd.read_csv()
        print(tag_data[:4])
        return tag_data

    def load_link(self):
        link_data = pd.read_csv()
        print(link_data[:4])
        return link_data

    def load_rating_dataset(self):
        reader = Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(self.load_rating()[['userId','movieId','rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.25)
        print(type(trainset))
        return trainset, testset


    # def create_anti_test_set(self, trainset):
    #     anti_testset = trainset.build_anti_testset()
    #     # print(anti_testset)
    #     return anti_testset

