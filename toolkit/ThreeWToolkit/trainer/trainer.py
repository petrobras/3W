import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from typing import Callable, Any
from pydantic import field_validator

from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..core.base_step import BaseStep
from ..core.base_model_trainer import ModelTrainerConfig
from ..core.enums import OptimizersEnum, CriterionEnum, TaskType
from ..models.mlp import MLPConfig
from ..models.sklearn_models import SklearnModelsConfig
from ..assessment.model_assess import ModelAssessment, ModelAssessmentConfig


class TrainerConfig(ModelTrainerConfig):
    """Configuration class for the ModelTrainer.

    This class defines all the hyperparameters and settings needed to configure
    the training process for machine learning models. It extends the base
    ModelTrainerConfig with specific training-related parameters.

    Args:
        batch_size (int): The number of samples per batch during training.
            Must be greater than 0.
        epochs (int): The total number of training epochs to run.
            Must be greater than 0.
        seed (int): Random seed for reproducibility across training runs.
        learning_rate (float): The learning rate for the optimizer.
            Must be greater than 0.
        config_model (MLPConfig | SklearnModelsConfig): Configuration object
            for the specific model type to be trained.
        criterion (str, optional): Loss function to use during training.
            Options: "cross_entropy", "binary_cross_entropy", "mse", "mae".
            Defaults to "cross_entropy".
        optimizer (str, optional): Optimization algorithm to use.
            Options: "adam", "adamw", "sgd", "rmsprop".
            Defaults to "adam".
        device (str, optional): Computing device for training.
            Options: "cpu", "cuda". Defaults to "cuda" if available, else "cpu".
        cross_validation (bool | None, optional): Whether to use k-fold
            cross-validation during training. Defaults to None (disabled).
        n_splits (int, optional): Number of folds for cross-validation.
            Only used when cross_validation is True. Defaults to 5.
        test_size (float, optional): Proportion of dataset reserved for testing.
            Defaults to 0.2.
        val_size (float, optional): Proportion of training data used for validation.
            Defaults to 0.3.
        shuffle_train (bool, optional): Whether to shuffle training data
            before each epoch. Defaults to True.

    Raises:
        ValueError: If any validation constraint is violated (e.g., negative
            batch_size, invalid optimizer name, etc.).

    Example:
        >>> config = TrainerConfig(
        ...     batch_size=32,
        ...     epochs=100,
        ...     seed=42,
        ...     learning_rate=0.001,
        ...     config_model=MLPConfig(...),
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
    cross_validation: bool | None = None
    n_splits: int = 5
    test_size: float = 0.2
    val_size: float = 0.3
    shuffle_train: bool = True

    @field_validator("batch_size")
    @classmethod
    def check_batch_size(cls, v):
        """Validate that batch_size is positive.

        Args:
            v (int): The batch size value to validate.

        Returns:
            int: The validated batch size.

        Raises:
            ValueError: If batch_size is not greater than 0.
        """
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    @field_validator("epochs")
    @classmethod
    def check_epochs(cls, v):
        """Validate that epochs is positive.

        Args:
            v (int): The number of epochs to validate.

        Returns:
            int: The validated number of epochs.

        Raises:
            ValueError: If epochs is not greater than 0.
        """
        if v <= 0:
            raise ValueError("epochs must be > 0")
        return v

    @field_validator("learning_rate")
    @classmethod
    def check_learning_rate(cls, v):
        """Validate that learning_rate is positive.

        Args:
            v (float): The learning rate value to validate.

        Returns:
            float: The validated learning rate.

        Raises:
            ValueError: If learning_rate is not greater than 0.
        """
        if v <= 0:
            raise ValueError("learning_rate must be > 0")
        return v

    @field_validator("n_splits")
    @classmethod
    def check_n_splits(cls, v, values):
        """Validate n_splits when cross-validation is enabled.

        Args:
            v (int): The number of splits to validate.
            values: The validation context containing other field values.

        Returns:
            int: The validated number of splits.

        Raises:
            ValueError: If n_splits is not greater than 1 when cross_validation is True.
        """
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
        """Validate that optimizer is from the supported list.

        Args:
            v (str): The optimizer name to validate.

        Returns:
            str: The validated optimizer name.

        Raises:
            ValueError: If optimizer is not in the supported list.
        """
        valid = {o.value for o in OptimizersEnum}
        if v not in valid:
            raise ValueError(f"optimizer must be one of {valid}")
        return v

    @field_validator("criterion")
    @classmethod
    def check_criterion(cls, v):
        """Validate that criterion is from the supported list.

        Args:
            v (str): The criterion name to validate.

        Returns:
            str: The validated criterion name.

        Raises:
            ValueError: If criterion is not in the supported list.
        """
        valid = {c.value for c in CriterionEnum}
        if v not in valid:
            raise ValueError(f"criterion must be one of {valid}")
        return v

    @field_validator("device")
    @classmethod
    def check_device(cls, v):
        """Validate that device is either 'cpu' or 'cuda'.

        Args:
            v (str): The device name to validate.

        Returns:
            str: The validated device name.

        Raises:
            ValueError: If device is not 'cpu' or 'cuda'.
        """
        valid = {"cpu", "cuda"}
        if v not in valid:
            raise ValueError("device must be 'cpu' or 'cuda'")
        return v


class ModelTrainer(BaseStep):
    def __init__(self, config: TrainerConfig) -> None:
        """Initialize the ModelTrainer with the given configuration.

        Args:
            config (TrainerConfig): Configuration object containing all
                training parameters and model settings.
        """
        """Initialize the ModelTrainer with the given configuration.

        Args:
            config (TrainerConfig): Configuration object containing all
                training parameters and model settings.
        """
        self.config = config
        self.lr = config.learning_rate
        self.device = config.device

        self.cross_validation = config.cross_validation
        self.n_splits = config.n_splits if config.cross_validation else None
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.seed = config.seed
        self.test_size = config.test_size
        self.val_size = config.val_size
        self.shuffle_train = config.shuffle_train
        self.history: list = []

    def pre_process(self, data: Any) -> dict[str, Any]:
        """Standardize input data format.

        Args:
            data: Input data (dict or tuple/list).

        Returns:
            Standardized data dictionary.
        """
        if isinstance(data, dict):
            processed_data = data.copy()
        else:
            if hasattr(data, "__iter__") and len(data) >= 2:
                processed_data = {"x_train": data[0], "y_train": data[1]}
                if len(data) >= 4:
                    processed_data.update({"x_val": data[2], "y_val": data[3]})
                if len(data) >= 5:
                    processed_data["kwargs"] = data[4]
            else:
                raise ValueError(
                    "Input data must be dict or iterable with (x_train, y_train)"
                )

        # Validate required keys
        required_keys = ["x_train", "y_train"]
        missing_keys = [k for k in required_keys if k not in processed_data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Ensure optional keys exist
        for key in ["x_val", "y_val", "kwargs"]:
            if key not in processed_data:
                processed_data[key] = None if key != "kwargs" else {}

        return processed_data

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Main logic of the step.

        Performs the actual model training using the provided data.

        Args:
            data (dict[str, Any]): Preprocessed data containing training information.

        Returns:
            dict[str, Any]: Data with training results added.
        """
        # Extract training parameters
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_val = data.get("x_val")
        y_val = data.get("y_val")
        kwargs = data.get("kwargs", {})

        # Perform training
        self.train(x_train, y_train, x_val, y_val, **kwargs)

        # Add training results to data
        data["model"] = self.model
        data["history"] = self.history
        data["trainer"] = self

        return data

    def post_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Standardizes the output of the step.

        Performs any final processing and ensures output format consistency.

        Args:
            data (dict[str, Any]): Data with training results.

        Returns:
            dict[str, Any]: Final processed data ready for next pipeline step.
        """
        # Ensure all expected outputs are present
        expected_outputs = ["model", "history", "trainer"]
        for key in expected_outputs:
            if key not in data:
                raise RuntimeError(
                    f"Training step failed to produce expected output: {key}"
                )

        # Add metadata about the training step
        data["training_completed"] = True
        data["model_type"] = type(self.model).__name__
        data["config"] = self.config

        return data

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame | pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | pd.Series | None = None,
        **kwargs,
    ) -> None:
        """Train model using configured strategy.

        Handles three scenarios:
        1. Cross-validation with k-fold splits
        2. Validation data provided
        3. Auto-split for validation

        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features (optional).
            y_val: Validation labels (optional).
            **kwargs: Additional training parameters.
        """
        if self.cross_validation:
            self._train_with_cross_validation(x_train, y_train, **kwargs)
        elif x_val is not None and y_val is not None:
            self._train_with_validation(x_train, y_train, x_val, y_val, **kwargs)
        else:
            self._train_with_auto_split(x_train, y_train, **kwargs)

    def _train_with_cross_validation(
        self, x_train: Any, y_train: Any, **kwargs
    ) -> None:
        """Train using k-fold cross-validation.

        Args:
            x_train: Training features.
            y_train: Training labels.
            **kwargs: Additional training parameters.
        """
        self.history = []
        self.models_per_fold = []

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.config.shuffle_train,
            random_state=self.seed,
        )
        splits = list(skf.split(x_train, y_train))

        pbar = tqdm(
            enumerate(splits),
            total=self.n_splits,
            desc="[Pipeline] Training Fold 1",
            unit="fold",
            colour="#0a2c53",
        )

        for fold, (train_idx, val_idx) in pbar:
            pbar.set_description_str(f"[Pipeline] Training Fold {fold + 1}")

            # Split data
            x_train_fold = self._select_rows(x_train, train_idx)
            y_train_fold = self._select_rows(y_train, train_idx)
            x_val_fold = self._select_rows(x_train, val_idx)
            y_val_fold = self._select_rows(y_train, val_idx)

            # Train on fold
            fold_history = self._call_training_strategy(
                x_train_fold, y_train_fold, x_val_fold, y_val_fold, **kwargs
            )
            self.history.append(fold_history)
            self.models_per_fold.append(self.model)

    def _train_with_validation(
        self, x_train: Any, y_train: Any, x_val: Any, y_val: Any, **kwargs
    ) -> None:
        """Train using provided validation data.

        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            **kwargs: Additional training parameters.
        """
        self.history = [
            self._call_training_strategy(x_train, y_train, x_val, y_val, **kwargs)
        ]

    def _train_with_auto_split(self, x_train: Any, y_train: Any, **kwargs) -> None:
        """Train with automatic train/validation split.

        Args:
            x_train: Training features.
            y_train: Training labels.
            **kwargs: Additional training parameters.
        """
        x_train, y_train, x_val, y_val = self._holdout(x_train, y_train)
        self.history = [
            self._call_training_strategy(x_train, y_train, x_val, y_val, **kwargs)
        ]

    def _call_training_strategy(
        self, x_train: Any, y_train: Any, x_val: Any = None, y_val: Any = None, **kwargs
    ) -> dict[str, list[Any]] | None:
        """Delegate training to the strategy.

        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features (optional).
            y_val: Validation labels (optional).
            **kwargs: Additional training parameters.

        Returns:
            Training history or None.
        """
        # Prepare kwargs for strategy
        strategy_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle_train,
            "device": self.device,
            **kwargs,
        }

        self.model = self.config.config_model.setup()

        training_strategy = self.model.get_training_strategy()
        strategy = training_strategy()

        # Reinitialize optimizer if model requires it
        if strategy.requires_optimizer():
            strategy_kwargs["optimizer"] = self._get_optimizer(
                self.model, self.config.optimizer
            )

        if strategy.requires_criterion():
            strategy_kwargs["criterion"] = self._get_fn_cost(self.config.criterion)

        return strategy.train(
            self.model, x_train, y_train, x_val, y_val, **strategy_kwargs
        )

    def _holdout(self, X, Y, test_size=None):
        """Split dataset using holdout.

        Args:
            X: Features.
            Y: Targets.
            test_size: Fraction for validation split.

        Returns:
            Tuple of (X_train, Y_train, X_val, Y_val).
        """
        if test_size is None:
            test_size = self.val_size

        X_train, X_val, Y_train, Y_val = train_test_split(
            X,
            Y,
            test_size=test_size,
            shuffle=self.shuffle_train,
            random_state=self.seed,
        )
        return X_train, Y_train, X_val, Y_val

    def _get_optimizer(self, model: Any, optimizer: str) -> torch.optim.Optimizer:
        """Create and return the specified PyTorch optimizer.

        Args:
            optimizer (str): Name of the optimizer to create. Must be one of:
                "adam", "adamw", "sgd", "rmsprop".

        Returns:
            torch.optim.Optimizer: The configured optimizer instance with
                model parameters and learning rate set.

        Raises:
            ValueError: If the optimizer name is not recognized.
        """
        model_params = model.get_params()
        if optimizer == OptimizersEnum.ADAM.value:
            return optim.Adam(params=model_params, lr=self.lr)
        elif optimizer == OptimizersEnum.ADAMW.value:
            return optim.AdamW(params=model_params, lr=self.lr)
        elif optimizer == OptimizersEnum.SGD.value:
            return optim.SGD(params=model_params, lr=self.lr)
        elif optimizer == OptimizersEnum.RMSPROP.value:
            return optim.RMSprop(params=model_params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _get_fn_cost(self, criterion: str | None) -> Callable:
        """Create and return the specified loss function.

        Args:
            criterion (str | None): Name of the loss function to create. Must be
                one of: "cross_entropy", "binary_cross_entropy", "mse", "mae".

        Returns:
            Callable: The configured loss function instance.

        Raises:
            ValueError: If the criterion name is not recognized.
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

    def _select_rows(self, data, idx):
        """Select rows from data using provided indices.

        Handles both pandas DataFrames/Series and other indexable data structures.

        Args:
            data: The data structure to index (DataFrame, Series, or array-like).
            idx: The indices to select.

        Returns:
            The selected subset of data.
        """
        if hasattr(data, "iloc"):
            return data.iloc[idx]
        else:
            return data[idx]

    def assess(
        self,
        x_test: Any,
        y_test: Any,
        assessment_config: ModelAssessmentConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate the trained model using ModelAssessment.

        This is a convenience method that creates a ModelAssessment instance
        and evaluates the current model on the provided test data.

        Args:
            x_test (Any): Test input features for evaluation.
            y_test (Any): Test target labels for evaluation.
            assessment_config (ModelAssessmentConfig | None, optional):
                Configuration for the assessment process. If None, creates
                a default configuration based on the task type.
            **kwargs: Additional arguments passed to ModelAssessment.evaluate().

        Returns:
            dict[str, Any]: Dictionary containing evaluation results with
                metrics, predictions, and other assessment information.

        Example:
            >>> # Basic assessment with default configuration
            >>> results = trainer.assess(X_test, y_test)
            >>> print(f"Accuracy: {results['accuracy']}")
            >>>
            >>> # Custom assessment configuration
            >>> config = ModelAssessmentConfig(
            ...     metrics=["accuracy", "f1", "precision", "recall"],
            ...     task_type=TaskType.CLASSIFICATION
            ... )
            >>> results = trainer.assess(X_test, y_test, assessment_config=config)
        """
        if assessment_config is None:
            # Create default assessment configuration based on task type
            assessment_config = ModelAssessmentConfig(
                metrics=(
                    ["accuracy"]
                    if self._is_classification_task()
                    else ["explained_variance"]
                ),
                task_type=(
                    TaskType.CLASSIFICATION
                    if self._is_classification_task()
                    else TaskType.REGRESSION
                ),
            )

        assessor = ModelAssessment(assessment_config)
        assessor._setup_metrics()
        return assessor.evaluate(self.model, x_test, y_test, **kwargs)

    def _is_classification_task(self) -> bool:
        """Determine if the current task is classification based on the loss function.

        Uses a heuristic approach by checking if the configured criterion
        is typically used for classification tasks.

        Returns:
            bool: True if this appears to be a classification task,
                False if it appears to be a regression task.

        Note:
            This is a heuristic and may not always be accurate. For more
            precise control, specify the task_type explicitly in the
            assessment_config parameter of the assess() method.
        """
        classification_criteria = {
            CriterionEnum.CROSS_ENTROPY.value,
            CriterionEnum.BINARY_CROSS_ENTROPY.value,
        }
        return self.config.criterion in classification_criteria
