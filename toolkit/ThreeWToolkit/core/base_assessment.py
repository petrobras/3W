"""Definition for the base assessment class."""

from abc import ABC

from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from .base_instantiable import Instantiable
from ..core.enums import TaskTypeEnum
from ..core.base_trainer import TrainingHistory


class AssessmentOutput(BaseModel):
    """Output container for model assessment results."""

    model_name: str = Field(..., description="Name of the model.")
    task_type: TaskTypeEnum = Field(..., description="Type of ML task.")
    timestamp: str = Field(..., description="Timestamp of the assessment.")

    true_values: np.ndarray | None = Field(
        default=None, description="Ground truth values."
    )
    predictions: np.ndarray | None = Field(
        default=None, description="Model predictions."
    )
    metrics: dict[str, float] | None = Field(
        default=None, description="Computed metrics."
    )
    training_history: TrainingHistory | None = Field(
        default=None, description="Training history from the trainer."
    )

    experiment_dir: str | None = Field(
        default=None, description="Directory where experiment results are saved."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAssessmentConfig(BaseModel, Instantiable):
    """Base configuration for assessments."""

    _target: type["BaseAssessment"]


class BaseAssessment(ABC):
    """Base class for assessments."""
