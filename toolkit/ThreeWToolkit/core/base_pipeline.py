"""Definition for the base pipeline class."""

from abc import ABC
from pydantic import BaseModel, Field
from typing import Any
from .base_instantiable import Instantiable
from .base_trainer import TrainingResult


class PipelineResult(BaseModel):
    """Container for pipeline execution results."""

    training_result: TrainingResult | None = None
    assessment_output: Any | None = None
    experiment_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BasePipelineConfig(BaseModel, Instantiable):
    """Base configuration for pipelines."""

    target_: type["BasePipeline"]


class BasePipeline(ABC):
    """Base class for pipelines."""

    def run(self):
        """Run the pipeline."""
        raise NotImplementedError("Subclasses must implement the run method.")
