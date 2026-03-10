from pathlib import Path
from surprise import dump

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "svd_model.dump"


class Model_Store:
    def __init__(self, model_path: Path):
        self.model_path = model_path

    def save_model(self, algo):
        """Save trained model parameters."""
        dump.dump(str(self.model_path), algo=algo)

    def load_model(self):
        """Load a trained model."""
        predictions, algo = dump.load(str(self.model_path))
        return algo
