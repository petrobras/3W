from abc import ABC
from pydantic import BaseModel

class MetricsConfig(BaseModel):
    pass

class BaseMetrics(ABC):
    def __init__(self, config: MetricsConfig):
        self.config = config