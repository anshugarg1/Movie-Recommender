from .top_n_recomm import get_top_n_for_user

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
        Optional: return movies the user already rated (for UI display).
        """
        return

    def similar_movies(self, movie_id: int, n: int = 10) -> list[dict]:
        """
        Optional: if you use item-based KNN, return similar movies with similarity score.
        """
        return