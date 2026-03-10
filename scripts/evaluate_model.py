from recommender.config import LINKS_PATH, MOVIES_PATH, RATINGS_PATH, TAGS_PATH
from recommender.data.loaders import Load_Data
from recommender.evaluation.metrics import evaluate_predictions
from recommender.models.algorithms import cf_svd


def main():
    loader = Load_Data(MOVIES_PATH, RATINGS_PATH, TAGS_PATH, LINKS_PATH)
    trainset, testset = loader.load_rating_dataset()

    predictions, _ = cf_svd(trainset, testset)
    metrics = evaluate_predictions(predictions, k=10, threshold=3.5)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")


if __name__ == "__main__":
    main()
