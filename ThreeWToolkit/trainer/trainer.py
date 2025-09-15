from typing import Callable, Any
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from pydantic import field_validator
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from ..core.base_model_trainer import BaseModelTrainer, ModelTrainerConfig
from ..core.enums import (
    OptimizersEnum,
    CriterionEnum,
)
from ..models.mlp import MLPConfig, MLP
from ..models.sklearn_models import SklearnModelsConfig, SklearnModels
from ..utils import ModelRecorder
from torch.utils.data import TensorDataset


class TrainerConfig(ModelTrainerConfig):
    """
    Configuration for the ModelTrainer, including training hyperparameters and model config.

    Args:
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
        learning_rate (float): Learning rate for optimizer.
        config_model (MLPConfig | SklearnModelsConfig): Model configuration object.
        optimizer (str): "adam", "adamw", "sgd", "rmsprop" (default: "adam").
        criterion (str): "cross_entropy", "binary_cross_entropy", "mse", "mae" (default: "cross_entropy").
        device (str): Device to use (default: 'cuda' if available, else 'cpu').
        metrics (list[Callable] | None): List of metrics to evaluate the model.
        cross_validation (bool | None): Whether to use cross-validation (default: None).
        n_splits (int | None): Number of splits for cross-validation (default: None).
        shuffle_train (bool): Whether to shuffle the training data (default: True).

    Example:
        >>> trainer_config = TrainerConfig(
        ...     batch_size=32,
        ...     epochs=10,
        ...     seed=42,
        ...     learning_rate=0.001,
        ...     config_model=MLPConfig(...),
        ...     optimizer=OptimizersEnum.ADAM,
        ...     criterion=CriterionEnum.MSE,
        ...     device='cuda',
        ...     metrics=[mean_squared_error, r2_score],
        ...     cross_validation=True,
        ...     n_splits=5
        ... )
    """

    batch_size: int
    epochs: int
    seed: int
    learning_rate: float
    config_model: MLPConfig | SklearnModelsConfig
    criterion: str = "cross_entropy"
    optimizer: str = "adam"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    metrics: list[Callable] | None = None
    cross_validation: bool | None = None
    n_splits: int = 5
    shuffle_train: bool = True

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v):
        if v <= 0:
            raise ValueError("epochs must be > 0")
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v):
        if v <= 0:
            raise ValueError("learning_rate must be > 0")
        return v

    @field_validator("n_splits")
    @classmethod
    def check_n_splits(cls, v, values):
        # values is a ValidationInfo object in Pydantic v2
        cross_val = (
            values.data.get("cross_validation")
            if hasattr(values, "data")
            else values.get("cross_validation")
        )
        if cross_val and v is not None and v <= 1:
            raise ValueError("n_splits must be > 1 for cross-validation")
        return v

    @field_validator("optimizer")
    @classmethod
    def check_optimizer(cls, v):
        valid = {o.value for o in OptimizersEnum}
        if v not in valid:
            raise ValueError(f"optimizer must be one of {valid}")
        return v

    @field_validator("criterion")
    @classmethod
    def check_criterion(cls, v):
        valid = {c.value for c in CriterionEnum}
        if v not in valid:
            raise ValueError(f"criterion must be one of {valid}")
        return v

    @field_validator("device")
    @classmethod
    def check_device(cls, v):
        valid = {"cpu", "cuda"}
        if v not in valid:
            raise ValueError("device must be 'cpu' or 'cuda'")
        return v


class ModelTrainer(BaseModelTrainer):
    """
    Generic model trainer supporting both PyTorch and scikit-learn models.

    Args:
        config (TrainerConfig): Configuration for the trainer and model.

    Example:
        >>> trainer = ModelTrainer(trainer_config)
        >>> trainer.train(x_train, y_train, x_val, y_val)
        >>> results = trainer.test(x_test, y_test, metrics=[...])
    """

    def __init__(self, config: TrainerConfig) -> None:
        """
        Initialize the ModelTrainer.

        Args:
            config (TrainerConfig): Trainer configuration object.
        """
        self.config = config
        self.lr = config.learning_rate
        self.device = config.device
        self.model = self._get_model(config.config_model)
        # Only create optimizer and criterion for PyTorch models
        if isinstance(self.model, MLP):
            self.optimizer = self._get_optimizer(self.config.optimizer)
            self.criterion = self._get_fn_cost(self.config.criterion)
        else:
            self.optimizer = None
            self.criterion = None
        self.cross_validation = config.cross_validation
        if self.config.cross_validation:
            self.n_splits = config.n_splits
        self.metrics = config.metrics
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.seed = config.seed
        self.shuffle_train = config.shuffle_train

    def _get_model(
        self, config_model: MLPConfig | SklearnModelsConfig
    ) -> MLP | SklearnModels:
        """
        Instantiate the model based on the configuration.

        Args:
            config_model (MLPConfig | SklearnModelsConfig): Model configuration object.

        Returns:
            MLP | SklearnModels: Instantiated model.

        Raises:
            ValueError: If the model config is unknown.
        """
        match config_model:
            case MLPConfig():
                return MLP(config_model).to(self.device)
            case SklearnModelsConfig():
                return SklearnModels(config_model)
            case _:
                raise ValueError(f"Unknown model config: {config_model}")

    def _get_optimizer(self, optimizer: str) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model.

        Args:
            optimizer (str): Optimizer type.

        Returns:
            torch.optim.Optimizer: Instantiated optimizer.

        Raises:
            ValueError: If the optimizer is unknown.

        Example:
            >>> optimizer = trainer._get_optimizer(OptimizersEnum.ADAM)
        """
        model_params = self.model.get_params()
        if optimizer == OptimizersEnum.ADAM.value:
            return optim.Adam(
                params=model_params,
                lr=self.lr,
            )
        elif optimizer == OptimizersEnum.ADAMW.value:
            return optim.AdamW(
                params=model_params,
                lr=self.lr,
            )
        elif optimizer == OptimizersEnum.SGD.value:
            return optim.SGD(params=model_params, lr=self.lr)
        elif optimizer == OptimizersEnum.RMSPROP.value:
            return optim.RMSprop(params=model_params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _get_fn_cost(self, criterion: str | None) -> Callable:
        """
        Get the loss function based on the criterion enum.

        Args:
            criterion (str | None): Loss function type.

        Returns:
            Callable: Loss function.

        Raises:
            ValueError: If the criterion is unknown.

        Example:
            >>> loss_fn = trainer._get_fn_cost("mse")
        """
        if criterion == CriterionEnum.CROSS_ENTROPY.value:
            return nn.CrossEntropyLoss()
        elif criterion == CriterionEnum.BINARY_CROSS_ENTROPY.value:
            return nn.BCEWithLogitsLoss()
        elif criterion == CriterionEnum.MSE.value:
            return nn.MSELoss()
        elif criterion == CriterionEnum.MAE.value:
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def _create_dataloader(
        self, x: Any, y: Any = None, shuffle: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader from input features and labels.

        Args:
            x (np.ndarray | torch.Tensor): Input features.
            y (np.ndarray | torch.Tensor): Labels.

        Returns:
            DataLoader: PyTorch DataLoader for the dataset.

        Example:
            >>> loader = trainer.create_dataloader(X, y, shuffle=True)
        """
        X_tensor = torch.tensor(x.values, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.tensor(y.values, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            y_tensor = torch.empty_like(X_tensor)
            dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame | pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | pd.Series | None = None,
        **kwargs,
    ) -> None:
        """
        Train the model using the provided data.

        Args:
            x_train (pd.DataFrame): Windowed training data.
            y_train (pd.DataFrame): Windowed training labels.

        Example:
            >>> trainer.train(windowed_data)
        """
        if self.cross_validation:
            # Save model initial state dict
            initial_state_dict = (
                self.model.state_dict() if isinstance(self.model, MLP) else None
            )
            self.history = []
            for fold, (train_idx, val_idx) in enumerate(
                TimeSeriesSplit(n_splits=self.n_splits).split(x_train, y_train)
            ):
                print(f"Training fold {fold + 1}/{self.n_splits}")
                # Reset model to initial state before each fold
                if isinstance(self.model, MLP) and initial_state_dict is not None:
                    self.model.load_state_dict(initial_state_dict)
                x_train_fold = self._select_rows(x_train, train_idx)
                y_train_fold = self._select_rows(y_train, train_idx)
                x_val_fold = self._select_rows(x_train, val_idx)
                y_val_fold = self._select_rows(y_train, val_idx)
                fold_history = self.call_trainer(
                    x_train_fold,
                    y_train_fold,
                    x_val=x_val_fold,
                    y_val=y_val_fold,
                    **kwargs,
                )
                self.history.append(fold_history)
        elif x_val is not None and y_val is not None:
            self.history = [
                self.call_trainer(
                    x_train,
                    y_train,
                    x_val=x_val,
                    y_val=y_val,
                    **kwargs,
                )
            ]
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                shuffle=self.shuffle_train,
            )
            self.history = [
                self.call_trainer(
                    x_train,
                    y_train,
                    x_val=x_val,
                    y_val=y_val,
                    **kwargs,
                )
            ]

    def _select_rows(self, data, idx):
        if hasattr(data, "iloc"):
            return data.iloc[idx]
        else:
            return data[idx]

    def call_trainer(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Any = None,
        y_val: Any = None,
        **kwargs,
    ) -> dict[str, list[Any]] | None:
        if isinstance(self.model, MLP):
            train_loader = self._create_dataloader(
                x_train, y_train, shuffle=self.shuffle_train
            )

            if x_val is not None and y_val is not None:
                val_loader = self._create_dataloader(x_val, y_val, shuffle=False)
            else:
                val_loader = None

            # Only pass optimizer/criterion if not None
            optimizer = (
                self.optimizer
                if self.optimizer is not None
                else torch.optim.Adam(self.model.parameters(), lr=self.lr)
            )
            criterion = self.criterion if self.criterion is not None else nn.MSELoss()
            return self.model.fit(
                train_loader,
                self.epochs,
                optimizer,
                criterion,
                val_loader,
                self.metrics,
                self.device,
            )
        else:
            return self.model.fit(x_train, y_train, **kwargs)

    def test(self, x: Any, y: Any, metrics: list[Callable], **kwargs) -> Any:
        """
        Evaluate the model on test data.

        Args:
            x (np.ndarray | torch.Tensor): Test features.
            y (np.ndarray | torch.Tensor): Test labels.
            metrics (list[Callable]): List of metric functions for evaluation.
            **kwargs: Additional arguments for the model's test/evaluate method.

        Returns:
            Any: Test results (loss, metrics, etc.).

        Example:
            >>> test_loss, test_metrics = trainer.test(X_test, y_test, metrics=[mean_squared_error])
        """
        if isinstance(self.model, MLP):
            test_loader = self._create_dataloader(x, y, shuffle=False)
            criterion = self.criterion if self.criterion is not None else nn.MSELoss()
            return self.model.test(test_loader, criterion, metrics, self.device)
        else:
            return self.model.evaluate(x, y, metrics)

    def predict(self, x: Any, **kwargs) -> Any:
        """
        Generate predictions using the trained model.

        Args:
            x (Dataloader): Input features.
            **kwargs: Additional arguments for the model's predict method.

        Returns:
            Any: Model predictions.

        Example:
            >>> preds = trainer.predict(X_test)
        """
        if isinstance(self.model, MLP):
            pred_loader = self._create_dataloader(x, shuffle=False)
            return self.model.predict(pred_loader, self.device, **kwargs)
        else:
            return self.model.predict(x, **kwargs)

    def save(self, filepath: Path) -> None:
        """
        Save a checkpoint of the model to the specified filepath.

        Args:
            filepath (Path): Path to save the model checkpoint.

        Example:
            >>> trainer.save(Path('best_model.pth'))
        """
        ModelRecorder.save_best_model(model=self.model, filename=filepath)

    def load(self, filepath: Path) -> MLP | SklearnModels:
        """
        Load a checkpoint of the model from the specified filepath.

        Args:
            filepath (Path): Path to the model checkpoint.

        Returns:
            MLP | SklearnModels: The loaded model instance.

        Example:
            >>> model = trainer.load(Path('best_model.pth'))
        """
        state_dict = ModelRecorder.load_model(filename=filepath)
        if isinstance(self.model, MLP):
            self.model.load_state_dict(state_dict)
        return self.model
