# Movie Recommender

Movie recommendation app built on MovieLens data using collaborative filtering with `surprise` SVD.

## Setup

```powershell
uv sync
```

## Train model

```powershell
uv run python scripts/train_model.py
```

## Run app

```powershell
uv run streamlit run src/recommender/api/app.py
```

If no model exists at `data/models/svd_model.dump`, the app auto-trains one on startup.

## Evaluate model

```powershell
uv run python scripts/evaluate_model.py
```

## Tune SVD hyperparameters

```powershell
uv run python scripts/tune_svd.py
```

## Quality checks

```powershell
uv run ruff check .
uv run black --check .
uv run pytest
```
