import logging
from pathlib import Path
from ..core.base_models import BaseModels
from ..models.sklearn_models import SklearnModels
from ..models.torch_models import TorchModels
from ..constants import CHECKPOINT_DIR

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
        path = Path(filename)
        if not path.is_absolute():
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            path = CHECKPOINT_DIR / Path(filename).name

        model.save(path)  # Call the model's save method
        logger.info("Model saved to %s", path)
        return path

    @staticmethod
    def load_model(filename: str | Path, model_type: type[BaseModels] | None = None) -> BaseModels:
        """
        Load a model from CHECKPOINT_DIR. Supports PyTorch (.pt, .pth) and Pickle (.pkl).

        Parameters:
            filename: File name (looked up in CHECKPOINT_DIR if not absolute).
            model_type: Model class for loading (optional, can be inferred from file extension).

        Returns:
            Loaded model.
        """
        path = Path(filename)
        if not path.is_absolute():
            path = CHECKPOINT_DIR / path.name

        if model_type is not None:
            model = model_type.load(path)
        else: # Infer model type from file extension
            if path.suffix in {".pt", ".pth"}:
                logger.debug("Inferring model type as TorchModels based on file extension %s", path.suffix)
                model = TorchModels.load(path)  # Call the model's load method, which will use this utility
            elif path.suffix in {".pkl", ".pickle"}:
                logger.debug("Inferring model type as SklearnModels based on file extension %s", path.suffix)
                model = SklearnModels.load(path)  # Call the model's load method, which will use this utility
            else:
                raise ValueError(
                    "Unsupported file extension for loading model. Supported: .pt, .pth, .pkl, .pickle"
                )
        logger.info("Model loaded from %s", path)
        return model
