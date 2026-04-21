# Movie Recommender System

A collaborative filtering movie recommender built on the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) dataset using the [Surprise](https://surpriselib.com/) library and a [Streamlit](https://streamlit.io/) web UI.

## Features

- **Personalised recommendations** — SVD model predicts top-N unseen movies for a selected user
- **User genre profile** — bar chart of genres weighted by the user's own ratings
- **Similar movies** — item-based KNN cosine similarity for "movies like this one"
- **CLI entry point** — quick terminal use via `main.py`

## Project structure

```
Movie-Recommender/
├── data/
│   ├── ml-latest-small/   # MovieLens raw CSVs
│   └── models/            # Saved model dumps (SVD + KNN)
├── scripts/
│   └── train_model.py     # Train & save both models
├── src/recommender/
│   ├── api/app.py         # Streamlit web app
│   ├── config.py          # Paths & environment config
│   ├── data/loaders.py    # CSV loading + Surprise dataset builder
│   ├── evaluation/metrics.py  # RMSE + Precision@K
│   ├── models/
│   │   ├── algorithms.py  # cf_user_based, cf_item_based, cf_svd
│   │   └── persistence.py # Model save/load via surprise.dump
│   └── recommender/
│       ├── service.py     # Recommender_Service (recs, genre profile, similar movies)
│       └── top_n_recomm.py # get_top_n_for_user
├── main.py                # CLI: train SVD, prompt for user, print top-10
└── pyproject.toml
```

## Setup

```bash
# Install dependencies (requires Python >= 3.11)
pip install -e .
```

## Train models

```bash
cd Movie-Recommender
python scripts/train_model.py
```

This trains and saves both the SVD model (`data/models/svd_model.dump`) and the KNN item-based model (`data/models/knn_item_model.dump`).

## Run the web app

```bash
streamlit run src/recommender/api/app.py
```

## Run the CLI

```bash
python main.py
```

## Algorithms

| Model | Use case |
|---|---|
| SVD | Personalised top-N recommendations |
| KNN item-based (cosine) | Similar movies lookup |
