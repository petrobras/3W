from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


class ModelTrainerConfig(BaseModel):
    pass


class BaseModelTrainer(ABC):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None

    @abstractmethod
    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def evaluate(self, x, y, metrics) -> dict:
        pass

    @abstractmethod
    def save_checkpoint(self, filepath: Path):
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: Path):
        pass
