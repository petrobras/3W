"""Definition for the base assessment class."""

from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import numpy as np

from .base_instantiable import Instantiable
from ..core.enums import TaskTypeEnum
from ..core.base_trainer import TrainingHistory


class AssessmentOutput(BaseModel):
    """Output container for model assessment results."""

    model_name: str
    task_type: TaskTypeEnum
    timestamp: str

    predictions: np.ndarray | None = None
    true_values: np.ndarray | None = None
    metrics: dict[str, float] | None = None
    training_history: TrainingHistory | None = None

    config: dict[str, Any] = Field(default_factory=dict)
    experiment_dir: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAssessmentConfig(BaseModel, Instantiable):
    """Base configuration for assessments."""

    _target: type["BaseAssessment"]


class BaseAssessment(ABC):
    """Base class for assessments."""
