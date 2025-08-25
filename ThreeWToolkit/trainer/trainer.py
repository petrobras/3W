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
    """
    Configuration for the ModelTrainer, including training hyperparameters and model config.

    Args:
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
        learning_rate (float): Learning rate for optimizer.
        config_model (MLPConfig | SklearnModelsConfig): Model configuration object.
        optimizer (OptimizersEnum): Optimizer type (default: ADAM).
        criterion (CriterionEnum): Loss function type (default: CROSS_ENTROPY).
        device (str): Device to use (default: 'cuda' if available, else 'cpu').

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
        ... )
    """

    batch_size: int
    epochs: int
    seed: int
    learning_rate: float
    config_model: MLPConfig | SklearnModelsConfig
    optimizer: OptimizersEnum = OptimizersEnum.ADAM
    criterion: CriterionEnum = CriterionEnum.CROSS_ENTROPY
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

    def __init__(self, config: TrainerConfig):
        """
        Initialize the ModelTrainer.

        Args:
            config (TrainerConfig): Trainer configuration object.
        """
        self.config = config
        self.lr = config.learning_rate
        self.device = config.device
        self.model = self._get_model(config.config_model)
        self.optimizer = self._get_optimizer(self.config.optimizer)
        self.criterion = self._get_fn_cost(self.config.criterion)
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.seed = config.seed

    def create_dataloader(self, x, y, shuffle: bool):
        """
        Create a DataLoader from input features and labels.

        Args:
            x (np.ndarray | torch.Tensor): Input features.
            y (np.ndarray | torch.Tensor): Labels.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader for the dataset.

        Example:
            >>> loader = trainer.create_dataloader(X, y, shuffle=True)
        """
        dataset = LabeledSubset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _get_model(self, config_model):
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

    def _get_optimizer(self, optimizer_enum: OptimizersEnum) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model.

        Args:
            optimizer_enum (OptimizersEnum): Optimizer type.

        Returns:
            torch.optim.Optimizer: Instantiated optimizer.

        Raises:
            ValueError: If the optimizer is unknown.

        Example:
            >>> optimizer = trainer._get_optimizer(OptimizersEnum.ADAM)
        """
        model_params = self.model.get_params()
        if optimizer_enum == OptimizersEnum.ADAM:
            return optim.Adam(
                params=model_params,
                lr=self.lr,
            )
        elif optimizer_enum == OptimizersEnum.ADAMW:
            return optim.AdamW(
                params=model_params,
                lr=self.lr,
            )
        elif optimizer_enum == OptimizersEnum.SGD:
            return optim.SGD(params=model_params, lr=self.lr)
        elif optimizer_enum == OptimizersEnum.RMSPROP:
            return optim.RMSprop(params=model_params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_enum}")

    def _get_fn_cost(self, criterion_enum: CriterionEnum | None) -> Callable:
        """
        Get the loss function based on the criterion enum.

        Args:
            criterion_enum (CriterionEnum | None): Loss function type.

        Returns:
            Callable: Loss function.

        Raises:
            ValueError: If the criterion is unknown.

        Example:
            >>> loss_fn = trainer._get_fn_cost(CriterionEnum.MSE)
        """
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
        metrics: list[Callable] | None = None,
        **kwargs,
    ):
        """
        Train the model using the provided data.

        Args:
            x_train (np.ndarray | torch.Tensor): Training features.
            y_train (np.ndarray | torch.Tensor): Training labels.
            x_val (np.ndarray | torch.Tensor, optional): Validation features.
            y_val (np.ndarray | torch.Tensor, optional): Validation labels.
            **kwargs: Additional arguments for the model's fit method.

        Example:
            >>> trainer.train(X_train, y_train, X_val, y_val)
        """
        if isinstance(self.model, MLP):
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
            self.model.fit(
                train_loader,
                self.epochs,
                self.optimizer,
                self.criterion,
                val_loader,
                metrics,
                self.device,
            )
        else:
            self.model.fit(x_train, y_train, **kwargs)

    def test(self, x, y, metrics, **kwargs) -> Any:
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
            # For PyTorch initialize loader
            test_loader = DataLoader(
                LabeledSubset(x, y),
                batch_size=self.batch_size,
                shuffle=False,
            )
            return self.model.test(test_loader, self.criterion, metrics, self.device)
        else:
            # For sklearn call evaluate with x, y
            return self.model.evaluate(x, y, metrics)

    def predict(self, x, **kwargs) -> Any:
        """
        Generate predictions using the trained model.

        Args:
            x (np.ndarray | torch.Tensor | DataLoader): Input features or DataLoader.
            **kwargs: Additional arguments for the model's predict method.

        Returns:
            Any: Model predictions.

        Example:
            >>> preds = trainer.predict(X_test)
        """
        if isinstance(self.model, MLP):
            pred_loader = (
                DataLoader(
                    x,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                if not isinstance(x, DataLoader)
                else x
            )
            return self.model.predict(pred_loader, self.device, **kwargs)
        else:
            return self.model.predict(x, **kwargs)

    def save(self, filepath: Path):
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

    @property
    def history(self):
        """
        Access the training history (train/val loss per epoch).

        Returns:
            dict: Dictionary with keys 'train_loss' and 'val_loss'.
        """
        if isinstance(self.model, MLP):
            return self.model.history
        else:
            raise AttributeError("This model does not track training history.")
