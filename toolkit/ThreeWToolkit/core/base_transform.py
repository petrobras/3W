from pydantic import BaseModel
import pandas as pd
from abc import ABC, abstractmethod
from .base_instantiable import Instantiable
from typing import Any


class BaseTransformConfig(BaseModel, Instantiable):
    """Base configuration for general transformation steps."""

    target_: type["BaseTransform"]


class BaseTransform(ABC):
    """Base class for general transformation steps."""

    def __init__(self, config: BaseTransformConfig):
        self.config = config

    @abstractmethod
    def fit(self, dataset: Any) -> None:
        """If needed, fit the transformation step to the data.
        By default, this does nothing, as some methods won't need fitting."""
        pass

    @abstractmethod
    def transform(self, dataset: Any) -> pd.DataFrame:
        """Transform the data using the fitted transformation step
        or directly if no fitting is needed."""
        raise NotImplementedError("Subclasses must implement the transform method.")

    def fit_and_transform(self, dataset: Any) -> pd.DataFrame:
        """Convenience method to fit and transform the data in one step."""
        self.fit(dataset)
        return self.transform(dataset)
