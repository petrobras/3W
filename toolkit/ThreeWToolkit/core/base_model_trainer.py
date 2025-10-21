from abc import ABC
from pydantic import BaseModel


class ModelTrainerConfig(BaseModel):
    pass


class BaseModelTrainer(ABC):
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
