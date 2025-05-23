from abc import ABC
from pydantic import BaseModel

class PreprocessingConfig(BaseModel):
    pass

class BasePreprocessing(ABC):
    def __init__(self, config: PreprocessingConfig):
        self.config = config