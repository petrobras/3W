from abc import ABC
from pydantic import BaseModel

class AssessmentVisualizationConfig(BaseModel):
    pass

class BaseAssessmentVisualization(ABC):
    def __init__(self, config: AssessmentVisualizationConfig):
        self.config = config