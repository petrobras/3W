from abc import ABC
from pydantic import BaseModel


class ModelTrainerConfig(BaseModel):
    pass


class BaseModelTrainer(ABC):
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the base model trainer.

        Args:
            config (ModelTrainerConfig): Configuration for the model trainer.
        """
        self.config = config
