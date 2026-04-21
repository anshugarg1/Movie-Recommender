from surprise import KNNBasic, SVD

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
    algo_svd = SVD(random_state = 42)
    algo_svd.fit(trainset)
    pred_svd = algo_svd.test(testset)
    return pred_svd, algo_svd
