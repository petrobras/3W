from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from ..core.base_model_trainer import BaseModelTrainer, ModelTrainerConfig
from ..core.enums import (
    OptimizersEnum,
    CriterionEnum,
)
from ..models.mlp import MLPConfig, MLP, LabeledSubset
from ..models.sklearn_models import SklearnModelsConfig, SklearnModels
from ..utils import ModelRecorder
from typing import Any


class TrainerConfig(ModelTrainerConfig):
    optimizer: OptimizersEnum
    criterion: CriterionEnum
    batch_size: int
    epochs: int
    seed: int
    config_model: MLPConfig | SklearnModelsConfig
    learning_rate: float
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelTrainer(BaseModelTrainer):
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.lr = config.learning_rate
        self.device = config.device
        self.model = self._get_model(config.config_model)
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.seed = config.seed

    def create_dataloader(self, x, y, shuffle: bool):
        dataset = LabeledSubset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _get_model(self, config_model):
        match config_model:
            case MLPConfig():
                self.optimizer = self._get_optimizer(self.config.optimizer)
                self.criterion = self._get_fn_cost(self.config.criterion)
                return MLP(config_model).to(self.device)
            case SklearnModelsConfig():
                return SklearnModels(config_model)
            case _:
                raise ValueError(f"Unknown model config: {config_model}")

    def _get_optimizer(self, optimizer_enum: OptimizersEnum) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model.
        """
        if optimizer_enum == OptimizersEnum.ADAM:
            return optim.Adam(
                params=self.model.parameters(),
                lr=self.lr,
            )
        elif optimizer_enum == OptimizersEnum.ADAMW:
            return optim.AdamW(
                params=self.model.parameters(),
                lr=self.lr,
            )
        elif optimizer_enum == OptimizersEnum.SGD:
            return optim.SGD(params=self.model.parameters(), lr=self.lr)
        elif optimizer_enum == OptimizersEnum.RMSPROP:
            return optim.RMSprop(params=self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_enum}")

    def _get_fn_cost(self, criterion_enum: CriterionEnum) -> Callable:
        if criterion_enum == CriterionEnum.CROSS_ENTROPY:
            return nn.CrossEntropyLoss()
        elif criterion_enum == CriterionEnum.BINARY_CROSS_ENTROPY:
            return nn.BCEWithLogitsLoss()
        elif criterion_enum == CriterionEnum.MSE:
            return nn.MSELoss()
        elif criterion_enum == CriterionEnum.MAE:
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_enum}")

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        **kwargs,
    ):
        # PyTorch models
        if hasattr(self.model, "model") and isinstance(
            getattr(self.model, "model"), torch.nn.Module
        ):
            train_loader = DataLoader(
                LabeledSubset(x_train, y_train),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = None
            if x_val is not None and y_val is not None:
                val_loader = DataLoader(
                    LabeledSubset(x_val, y_val),
                    batch_size=self.batch_size,
                    shuffle=False,
                )
            self.model.train(train_loader, val_loader, **kwargs)
        else:
            # SkLearn models
            self.model.train(x_train, y_train, **kwargs)
        pass

    def evaluate(self, x, y, metrics: list[Callable], **kwargs) -> dict:
        """
        Generic evaluate method for both PyTorch and Sklearn models.
        """
        return self.model.evaluate(x, y, metrics, **kwargs)

    def test(self, x, y, **kwargs) -> Any:
        """
        Generic test method for both PyTorch and Sklearn models.
        """
        if hasattr(self.model, "model") and isinstance(
            getattr(self.model, "model"), torch.nn.Module
        ):
            test_loader = DataLoader(
                LabeledSubset(x, y),
                batch_size=self.batch_size,
                shuffle=False,
            )
            return self.model.test(test_loader, **kwargs)
        else:
            # For sklearn, test is usually just evaluate
            return self.model.evaluate(x, y, kwargs.get("metrics", []), **kwargs)

    def predict(self, x, **kwargs) -> Any:
        """
        Generic predict method for both PyTorch and Sklearn models.
        """
        if hasattr(self.model, "model") and isinstance(
            getattr(self.model, "model"), torch.nn.Module
        ):
            pred_loader = (
                DataLoader(
                    x,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                if not isinstance(x, DataLoader)
                else x
            )
            return self.model.predict(pred_loader, **kwargs)
        else:
            return self.model.predict(x, **kwargs)

    def save_weights(self, filepath: Path):
        """
        Saves a checkpoint of the model.
        """
        ModelRecorder.save_best_model(model=self.model, filename=filepath)

    def load_weights(self, filepath: Path) -> model:
        """
        Loads a checkpoint of the model.
        """
        state_dict = ModelRecorder.load_model(filename=filepath)
        # pass state dict to self.model if its a pytorch instance
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(state_dict)

        return self.model
