from abc import ABC
from pydantic import BaseModel

class ModelDevelopmentConfig(BaseModel):
    pass

class BaseModelDevelopment(ABC):
    def __init__(self, config: ModelDevelopmentConfig):
        self.config = config