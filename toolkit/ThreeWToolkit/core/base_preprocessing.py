from pydantic import BaseModel
from abc import ABC, abstractmethod
from .base_instantiable import Instantiable


class BasePreprocessingConfig(BaseModel, Instantiable):
    """Base configuration for preprocessing steps."""

    target_: type["BasePreprocessing"]


class BasePreprocessing(ABC):
    """Base class for preprocessing steps."""

    def __init__(self, config: BasePreprocessingConfig):
        self.config = config

    @abstractmethod
    def transform(self, data: dict) -> dict:
        """Transform the data using the fitted preprocessing step
        or directly if no fi1ting is needed."""
        raise NotImplementedError("Subclasses must implement the transform method.")

    def fit(self, data: dict) -> None:
        """If needed, fit the preprocessing step to the data.
        By default, this does nothing, as some methods won't need fitting."""
        pass

    def compute(self) -> None:
        pass
