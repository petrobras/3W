from abc import ABC
from pydantic import BaseModel

class TimeSeriesHoldoutConfig(BaseModel):
    pass

class BaseTimeSeriesHoldout(ABC):
    def __init__(self, config: TimeSeriesHoldoutConfig):
        self.config = config