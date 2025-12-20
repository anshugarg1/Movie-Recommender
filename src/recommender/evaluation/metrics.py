from surprise.accuracy import rmse
from collections import defaultdict

def eval_res(pred_user, pred_item, pred_svd):
    #Evaluation
    print("Evaluation User based CR RMSE")
    rmse(pred_user)

    print("Evaluation Item based CR RMSE")
    rmse(pred_item)

    print("Evaluation SVD RMSE")
    rmse(pred_svd)

def precision_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))
    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (est, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, true_r) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    # Return the average precision@k for all users
    return sum(prec for prec in precisions.values()) / len(precisions)
