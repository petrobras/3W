from typing import Mapping, Any
import logging

from typing import Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt
from pathlib import Path
import pickle
from pydantic import Field, PrivateAttr, field_validator, ValidationInfo

from ..core.base_models import ModelsConfig, BaseModels

logger = logging.getLogger(__name__)


@runtime_checkable
class SklearnModelProtocol(Protocol):
    """Protocol for sklearn models. Used for type hinting."""

    def fit(self, x, y, **fit_params): ...
    def score(self, x, y, **score_params) -> float: ...
    def predict(self, x) -> npt.NDArray[np.number]: ...
    def get_params(self) -> Mapping[str, object]: ...
    def set_params(self, **params: object) -> None: ...


@runtime_checkable
class SklearnModelWithPredictProbaProtocol(SklearnModelProtocol, Protocol):
    """Protocol for sklearn models that support predict_proba. Used for type hinting."""

    def predict_proba(self, x) -> npt.NDArray[np.number]: ...


class SklearnModelsConfig(ModelsConfig):
    """Configuration for scikit-learn models."""

    model_type: (
        type[SklearnModelProtocol] | type[SklearnModelWithPredictProbaProtocol]
    ) = Field(
        ...,
        description="Type of sklearn model to use. Must be one of the supported models.",
    )

    model_params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific hyperparameters."
    )
    _target: type = PrivateAttr(default_factory=lambda: SklearnModels)

    @field_validator("model_params")
    @classmethod
    def check_model_params(
        cls, model_params: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        """Validate that model_type is supported."""
        try:  # try to instantiate the model with given parameters
            info.data["model_type"](**model_params)
        except Exception as e:
            raise ValueError(f"Invalid model_params: {e}")
        return model_params


class SklearnModels(BaseModels):
    """Sklearn model wrapper. Use SklearnTrainer for training."""

    def __init__(self, config: SklearnModelsConfig):
        """Initialize SklearnModels with given configuration.
        Args:
            config: SklearnModelsConfig instance containing model configuration.
        """
        self.config: SklearnModelsConfig = config

        self.model: SklearnModelProtocol = self.config.model_type(
            **self.config.model_params
        )

        self._feature_names: list[str] | None = None

    @property
    def model_name(self) -> str:
        """Get the name of the model class.
        Returns:
            Name of the underlying model class.
        """
        return self.model.__class__.__name__

    def save(self, filename: str | Path) -> Path:
        """Save model to disk.
        Args:
            filename: Path to save the model. Must have .pkl or .pickle extension.\
                    If no extension is provided, .pkl will be used by default.
        Returns:
            Path to the saved model.
        """
        path = Path(filename)  # ensure path
        if path.suffix and path.suffix not in {".pkl", ".pickle"}:
            raise ValueError(
                "Sklearn models must be saved with .pkl or .pickle extension"
            )
        elif not path.suffix:
            path = path.with_suffix(".pkl")
            logger.warning(
                "No file extension provided. Saving sklearn model with .pkl extension: %s",
                path,
            )

        with path.open("wb") as f:
            pickle.dump(self, f)

        return path

    @classmethod
    def load(cls, filename: str | Path) -> "SklearnModels":
        """Load model from disk.
        Args:
            filename: Path to the saved model.
        Returns:
            SklearnModels instance loaded from disk.
        """
        path = Path(filename)
        with path.open("rb") as f:
            obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise ValueError(
                    f"Loaded object is not a SklearnModels instance: {type(obj)}"
                )
            return obj

    def get_params(self) -> Mapping[str, object]:
        """Return the parameters of the underlying model.
        Returns:
            A mapping of parameter names to their values.
        """
        return self.model.get_params()

    def set_params(self, **params: object) -> None:
        """Set the parameters of the underlying model.
        Args:
            **params: Parameter names and their new values.
        """
        self.model.set_params(**params)
