from abc import ABC
from pydantic import BaseModel


class AssessmentConfig(BaseModel):
    pass


class BaseAssessment(ABC):
    def __init__(self, config: AssessmentConfig):
        self.config = config
