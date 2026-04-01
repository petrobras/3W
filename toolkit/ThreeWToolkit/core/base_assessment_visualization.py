"""Definition for the base assessment visualization class."""

from abc import ABC
from .base_instantiable import Instantiable
from pydantic import BaseModel


class BaseAssessmentVisualizationConfig(BaseModel, Instantiable):
    """Base configuration for assessment visualizations."""

    target_: type["BaseAssessmentVisualization"]


class BaseAssessmentVisualization(ABC):
    def __init__(self, config: BaseAssessmentVisualizationConfig):
        self.config = config
