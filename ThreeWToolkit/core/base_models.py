from abc import ABC
from pydantic import BaseModel

class ModelsConfig(BaseModel):
    pass

class BaseModels(ABC):
    def __init__(self, config: ModelsConfig):
        self.config = config