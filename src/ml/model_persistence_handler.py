import os
import joblib
from typing import Any


from src.config import MODELS_DIR


class ModelPersistenceHandler:
    DEFAULT_MODEL_FILENAME = os.path.join(MODELS_DIR, 'trained_model.joblib')

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model_joblib(self, model, filename='trained_model.joblib') -> None:
        """
        Save a model using joblib.
        :param model: The model to be saved.
        :param filename: The name of the file where the model will be saved.
        """
        model_path = os.path.join(self.models_dir, filename)
        joblib.dump(model, model_path)
        print(f"Model saved as {model_path}")

    def load_model_joblib(self, filename='trained_model.joblib') -> Any:
        """
        Load a model using joblib.
        :param filename: The name of the file where the model is saved.
        :return: The loaded model.
        """
        model_path = os.path.join(self.models_dir, filename)
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
