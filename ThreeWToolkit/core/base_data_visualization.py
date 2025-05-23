from abc import ABC
from pydantic import BaseModel

class DataVisualizationConfig(BaseModel):
    pass

class BaseDataVisualization(ABC):
    def __init__(self, config: DataVisualizationConfig):
        self.config = config