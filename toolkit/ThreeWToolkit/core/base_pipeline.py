"""Definition for the base pipeline class."""

from abc import ABC
from pydantic import BaseModel
from .base_instantiable import Instantiable


class BasePipelineConfig(BaseModel, Instantiable):
    """Base configuration for pipelines."""

    target_: type["BasePipeline"]


class BasePipeline(ABC):
    """Base class for pipelines."""

    def run(self):
        """Run the pipeline."""
        raise NotImplementedError("Subclasses must implement the run method.")
