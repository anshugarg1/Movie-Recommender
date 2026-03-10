from collections import defaultdict

from surprise.accuracy import rmse


def _group_predictions_by_user(predictions):
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))
    return user_est_true


def precision_at_k(predictions, k=10, threshold=3.5):
    user_est_true = _group_predictions_by_user(predictions)
    precisions = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold and est >= threshold) for est, true_r in user_ratings[:k]
        )
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0.0

    return sum(precisions.values()) / len(precisions) if precisions else 0.0


def recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = _group_predictions_by_user(predictions)
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rel_and_rec_k = sum(
            (true_r >= threshold and est >= threshold) for est, true_r in user_ratings[:k]
        )
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0.0

    return sum(recalls.values()) / len(recalls) if recalls else 0.0


def map_at_k(predictions, k=10, threshold=3.5):
    user_est_true = _group_predictions_by_user(predictions)
    ap_scores = []

    for _, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        hits = 0
        precision_sum = 0.0
        relevant_total = sum(true_r >= threshold for _, true_r in top_k)
        if relevant_total == 0:
            ap_scores.append(0.0)
            continue

        for idx, (est, true_r) in enumerate(top_k, start=1):
            if est >= threshold and true_r >= threshold:
                hits += 1
                precision_sum += hits / idx

        ap_scores.append(precision_sum / relevant_total)

    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def evaluate_predictions(predictions, k=10, threshold=3.5):
    return {
        "rmse": rmse(predictions, verbose=False),
        "precision_at_k": precision_at_k(predictions, k=k, threshold=threshold),
        "recall_at_k": recall_at_k(predictions, k=k, threshold=threshold),
        "map_at_k": map_at_k(predictions, k=k, threshold=threshold),
    }
