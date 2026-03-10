from surprise import KNNBasic, SVD
from surprise.model_selection import GridSearchCV

def cf_item_based(trainset, testset):    
    #Item based collaborative filtering - KNN
    sim_dict_2 = {'name':'cosine', 'user_based':False}
    algo_item = KNNBasic(sim_options=sim_dict_2)
    algo_item.fit(trainset)
    pred_item = algo_item.test(testset)
    return pred_item

def cf_user_based(trainset, testset):
    #User based collaborative filtering - KNN
    sim_dict = {'name':'cosine', 'user_based':True}
    algo_user = KNNBasic(sim_options=sim_dict)
    algo_user.fit(trainset)
    pred_user = algo_user.test(testset)
    return pred_user

def cf_svd(trainset, testset):      
    #SVD approach
    algo_svd = SVD(random_state=42)
    algo_svd.fit(trainset)
    pred_svd = algo_svd.test(testset)
    return pred_svd, algo_svd


def tune_svd(data):
    param_grid = {
        "n_factors": [50, 100],
        "n_epochs": [20, 30],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.1],
    }

    grid = GridSearchCV(
        SVD,
        param_grid,
        measures=["rmse", "mae"],
        cv=3,
        n_jobs=-1,
        joblib_verbose=0,
    )
    grid.fit(data)
    return grid.best_score["rmse"], grid.best_params["rmse"]
