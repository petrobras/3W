from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path
from typing import Any


class ModelTrainerConfig(BaseModel):
    pass


class BaseModelTrainer(ABC):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model.

        For PyTorch models:
            train(train_loader: DataLoader, val_loader: DataLoader, ...)

        For Sklearn models:
            train(x_train: np.ndarray, y_train: np.ndarray, ...)
        """
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> dict:
        pass

    @abstractmethod
    def save(self, filepath: Path):
        pass

    @abstractmethod
    def load(self, filepath: Path) -> Any:
        pass
