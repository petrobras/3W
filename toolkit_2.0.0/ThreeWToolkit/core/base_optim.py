from abc import ABC
from pydantic import BaseModel


class OptimConfig(BaseModel):
    pass


class BaseOptim(ABC):
    def __init__(self, config: OptimConfig):
        self.config = config
