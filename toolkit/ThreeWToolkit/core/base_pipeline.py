"""Definition for the base pipeline class."""

from abc import ABC
from pydantic import BaseModel, Field
from typing import Any
from .base_instantiable import Instantiable
from .base_trainer import CrossValidationResult, TrainingResult
from .base_assessment import AssessmentOutput


class PipelineResult(BaseModel):
    """Container for pipeline execution results."""

    training_result: TrainingResult | CrossValidationResult | None = Field(
        default=None, description="Result from training phase."
    )
    assessment_output: AssessmentOutput | None = Field(
        default=None, description="Output from assessment phase."
    )
    experiment_name: str = Field(default="", description="Name of the experiment.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the pipeline run."
    )


class BasePipelineConfig(BaseModel, Instantiable):
    """Base configuration for pipelines."""

    _target: type["BasePipeline"]


class BasePipeline(ABC):
    """Base class for pipelines."""

    def run(self):
        """Run the pipeline."""
        raise NotImplementedError("Subclasses must implement the run method.")
