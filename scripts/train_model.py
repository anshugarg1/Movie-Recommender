from surprise import SVD

from recommender.models.training import train_and_save_svd


def train_and_save_model() -> SVD:
    return train_and_save_svd(use_full_trainset=True)


def main():
    print("Training SVD model on full ratings dataset...")
    train_and_save_model()
    print("Training finished.")
    print("Model saved.")


if __name__ == "__main__":
    main()
