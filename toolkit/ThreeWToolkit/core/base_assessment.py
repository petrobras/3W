"""Definition for the base assessment class."""

from pydantic import BaseModel, ConfigDict, Field
from ..core.enums import TaskTypeEnum
import numpy as np
from typing import Any
from abc import ABC
from .base_instantiable import Instantiable


class AssessmentOutput(BaseModel):
    """Output container for model assessment results."""

    model_name: str
    task_type: TaskTypeEnum
    timestamp: str

    predictions: np.ndarray | None = None
    true_values: np.ndarray | None = None
    metrics: dict[str, float] | None = None
    training_history: dict[str, Any] | None = None

    config: dict[str, Any] = Field(default_factory=dict)
    experiment_dir: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseAssessmentConfig(BaseModel, Instantiable):
    """Base configuration for assessments."""

    target_: type["BaseAssessment"]


class BaseAssessment(ABC):
    """Base class for assessments."""
