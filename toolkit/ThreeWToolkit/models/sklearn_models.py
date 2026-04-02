from typing import Mapping
import logging

from pathlib import Path
import pickle
from pydantic import Field, PrivateAttr, field_validator

from sklearn.base import BaseEstimator

from ..core.base_models import ModelsConfig, BaseModels

logger = logging.getLogger(__name__)

class SklearnModelsConfig(ModelsConfig):
    """Sklearn model configuration. Use with SklearnTrainer for training."""

    model_type: type[BaseEstimator] = Field(
        ..., description="Type of sklearn model to use. Must be one of the supported models."
    )

    model_params: dict[str, int | float | str | bool | None] = Field(
        default_factory=dict, description="Model-specific hyperparameters."
    )
    _target: type = PrivateAttr(default_factory=lambda: SklearnModels)

    @field_validator("model_params")
    @classmethod
    def check_model_params(cls, model_params, info):
        """Validate that model_type is supported."""
        try: # try to instantiate the model with given parameters
            info.data["model_type"](**model_params)
        except Exception as e:
            raise ValueError(f"Invalid model_params: {e}")
        return model_params


class SklearnModels(BaseModels):
    """Sklearn model wrapper. Use SklearnTrainer for training."""

    def __init__(self, config: SklearnModelsConfig):
        self.config: SklearnModelsConfig = config

        self.model: BaseEstimator = self.config.model_type(**self.config.model_params)

        self._feature_names: list[str] | None = None

    @property
    def model_name(self) -> str:
        return self.model.__class__.__name__

    def save(self, filename: str | Path) -> Path:
        """Save model to disk."""
        path = Path(filename) # ensure path
        if path.suffix and path.suffix not in {".pkl", ".pickle"}:
            raise ValueError(
                "Sklearn models must be saved with .pkl or .pickle extension"
            )
        elif not path.suffix:
            path = path.with_suffix(".pkl")
            logger.warning(
                "No file extension provided. Saving sklearn model with .pkl extension: %s", path
            )

        with path.open("wb") as f:
            pickle.dump(self, f)

        return path


    @classmethod
    def load(cls, filename: str | Path) -> "SklearnModels":
        """Load model from disk."""
        path = Path(filename)
        with path.open("rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise ValueError(f"Loaded object is not a SklearnModels instance: {type(obj)}")
            return obj

    def get_params(self) -> Mapping[str, object]:
        return self.model.get_params()

    def set_params(self, **params: object) -> None:
        self.model.set_params(**params)
