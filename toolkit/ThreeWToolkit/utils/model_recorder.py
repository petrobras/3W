import logging
from pathlib import Path
from ..core.base_models import BaseModels, BaseTorchModels, BaseSkLearnModels
from ..constants import CHECKPOINT_DIR
import torch
import pickle

logger = logging.getLogger(__name__)


class ModelRecorder:
    """Utility for saving and loading models to/from CHECKPOINT_DIR."""

    @staticmethod
    def save_model(model: BaseModels, filename: str | Path) -> Path:
        """
        Save a model to CHECKPOINT_DIR. Supports PyTorch and scikit-learn.

        Parameters:
            model: Trained model object.
            filename: File name (saved inside CHECKPOINT_DIR).

        Returns:
            Full path where model was saved.
        """
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINT_DIR / Path(filename).name

        if isinstance(model, BaseTorchModels):
            try:
                torch.save(model.state_dict(), path)
            except Exception as e:
                raise RuntimeError(f"Error saving PyTorch model: {e}")

        elif isinstance(model, BaseSkLearnModels):
            try:
                with open(path, "wb") as f:
                    pickle.dump(model, f)
            except Exception as e:
                raise RuntimeError(f"Error saving Pickle model: {e}")

        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        logger.info("Model saved to %s", path)
        return path

    @staticmethod
    def load_model(filename: str | Path, model: BaseModels | None = None) -> BaseModels:
        """
        Load a model from CHECKPOINT_DIR. Supports PyTorch (.pt, .pth) and Pickle (.pkl).

        Parameters:
            filename: File name (looked up in CHECKPOINT_DIR if not absolute).
            model: Model instance for loading PyTorch weights (optional).

        Returns:
            Loaded model.
        """
        path = Path(filename)
        if not path.is_absolute():
            path = CHECKPOINT_DIR / path.name

        ext = path.suffix.lower()

        if ext in [".pt", ".pth"]:
            try:
                if model is None:
                    return torch.load(path)
                state_dict = torch.load(path)
                if isinstance(model, BaseTorchModels):
                    model.load_state_dict(state_dict)
                return model
            except Exception as e:
                raise RuntimeError(f"Error loading PyTorch model: {e}")

        elif ext in [".pkl", ".pickle"]:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Error loading Pickle model: {e}")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info("Model loaded from %s", path)
