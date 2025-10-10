import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from pathlib import Path
from typing import Callable, Any
from torch.utils.data import DataLoader
from pydantic import field_validator

from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..core.base_step import BaseStep
from ..core.base_model_trainer import ModelTrainerConfig
from ..core.enums import OptimizersEnum, CriterionEnum, TaskType
from ..models.mlp import MLPConfig, MLP
from ..models.sklearn_models import SklearnModelsConfig, SklearnModels
from ..utils import ModelRecorder
from torch.utils.data import TensorDataset
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
    """Simplified model trainer focused on training with delegated evaluation.

    This class handles the training process for both PyTorch neural networks (MLP)
    and scikit-learn models. It supports regular training, cross-validation, and
    provides convenient methods for model persistence and evaluation.

    The trainer is designed to be lightweight and focused solely on training,
    with evaluation capabilities delegated to the ModelAssessment class for
    better separation of concerns.

    Args:
        config (TrainerConfig): Configuration object containing all training
            parameters and model configuration.

    Attributes:
        config (TrainerConfig): The training configuration.
        lr (float): Learning rate from configuration.
        device (str): Computing device ('cpu' or 'cuda').
        model (MLP | SklearnModels): The instantiated model to train.
        optimizer (torch.optim.Optimizer | None): PyTorch optimizer (None for sklearn models).
        criterion (Callable | None): Loss function (None for sklearn models).
        cross_validation (bool | None): Whether cross-validation is enabled.
        n_splits (int): Number of folds for cross-validation.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
        shuffle_train (bool): Whether to shuffle training data.
        history (list): Training history from completed training runs.

    Example:
        Basic usage:
        >>> config = TrainerConfig(
        ...     batch_size=32,
        ...     epochs=100,
        ...     seed=42,
        ...     learning_rate=0.001,
        ...     config_model=MLPConfig(...)
        ... )
        >>> trainer = ModelTrainer(config)
        >>> trainer.train(X_train, y_train, X_val, y_val)
        >>>
        >>> # Save the trained model
        >>> trainer.save(Path("model.pth"))
        >>>
        >>> # Evaluate using ModelAssessment
        >>> results = trainer.assess(X_test, y_test)

        Cross-validation usage:
        >>> config = TrainerConfig(
        ...     batch_size=32,
        ...     epochs=100,
        ...     seed=42,
        ...     learning_rate=0.001,
        ...     config_model=MLPConfig(...),
        ...     cross_validation=True,
        ...     n_splits=5
        ... )
        >>> trainer = ModelTrainer(config)
        >>> trainer.train(X_train, y_train)  # No validation set needed
    """

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
        self.model = self._get_model(config.config_model)

        # Only create optimizer and criterion for PyTorch models
        if isinstance(self.model, MLP):
            self.optimizer: torch.optim.Optimizer | None = self._get_optimizer(
                self.config.optimizer
            )
            self.criterion: Callable[..., Any] | None = self._get_fn_cost(
                self.config.criterion
            )
        else:
            self.optimizer = None
            self.criterion = None

        self.cross_validation = config.cross_validation
        if self.config.cross_validation:
            self.n_splits = config.n_splits
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.seed = config.seed
        self.test_size = config.test_size
        self.val_size = config.val_size
        self.shuffle_train = config.shuffle_train
        self.history: list = []

    def pre_process(self, data: Any) -> dict[str, Any]:
        """Standardizes the input of the step.

        Validates and standardizes the input data format for training.

        Args:
            data: Input data that should contain training data and optionally validation data.
                  Can be a dict or any structure containing the required training data.

        Returns:
            dict[str, Any]: Standardized data dictionary with required keys.

        Raises:
            ValueError: If required training data is missing.
        """
        if isinstance(data, dict):
            processed_data = data.copy()
        else:
            # If data is not a dict, assume it's a tuple/list with (x_train, y_train, ...)
            if hasattr(data, "__iter__") and len(data) >= 2:
                processed_data = {"x_train": data[0], "y_train": data[1]}
                if len(data) >= 4:
                    processed_data.update({"x_val": data[2], "y_val": data[3]})
                if len(data) >= 5:
                    processed_data["kwargs"] = data[4]
            else:
                raise ValueError(
                    "Input data must be a dict or iterable with at least (x_train, y_train)"
                )

        # Validate required keys
        required_keys = ["x_train", "y_train"]
        missing_keys = [key for key in required_keys if key not in processed_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in input data: {missing_keys}")

        # Ensure optional keys exist with None defaults
        optional_keys = ["x_val", "y_val", "kwargs"]
        for key in optional_keys:
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

    def _get_model(
        self, config_model: MLPConfig | SklearnModelsConfig
    ) -> MLP | SklearnModels:
        """Instantiate the appropriate model based on the configuration.

        Args:
            config_model (MLPConfig | SklearnModelsConfig): Model configuration
                object that determines which type of model to create.

        Returns:
            MLP | SklearnModels: The instantiated model, moved to the appropriate
                device if it's a PyTorch model.

        Raises:
            ValueError: If the model configuration type is not recognized.
        """
        match config_model:
            case MLPConfig():
                return MLP(config_model).to(self.device)
            case SklearnModelsConfig():
                return SklearnModels(config_model)
            case _:
                raise ValueError(f"Unknown model config: {config_model}")

    def _get_optimizer(self, optimizer: str) -> torch.optim.Optimizer:
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
        model_params = self.model.get_params()
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

    def _create_dataloader(
        self, x: Any, y: Any = None, shuffle: bool = False
    ) -> DataLoader:
        """Create a PyTorch DataLoader from pandas DataFrame/Series.

        Converts pandas data structures to PyTorch tensors and wraps them
        in a DataLoader for batch processing during training.

        Args:
            x (Any): Input features as pandas DataFrame or compatible structure.
            y (Any, optional): Target labels as pandas DataFrame/Series or
                compatible structure. If None, creates empty tensors.
            shuffle (bool, optional): Whether to shuffle data in the DataLoader.
                Defaults to False.

        Returns:
            DataLoader: PyTorch DataLoader configured with the specified
                batch size and shuffle setting.
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
        """Train the model using the provided training data.

        This method handles three training scenarios:
        1. Cross-validation: Uses k-fold cross-validation with stratified splits
        2. Validation provided: Uses provided validation data
        3. Auto-split: Automatically splits training data for validation

        The training history is stored in self.history for later analysis.

        Args:
            x_train (pd.DataFrame): Training input features.
            y_train (pd.DataFrame | pd.Series): Training target labels.
            x_val (pd.DataFrame | None, optional): Validation input features.
                If None and cross_validation is False, data will be auto-split.
            y_val (pd.DataFrame | pd.Series | None, optional): Validation target
                labels. If None and cross_validation is False, data will be auto-split.
            **kwargs: Additional keyword arguments passed to the underlying
                model's fit method.

        Note:
            - For cross-validation, x_val and y_val are ignored
            - Model state is reset between cross-validation folds for PyTorch models
            - Training history is stored as a list of fold histories (cross-validation)
            or a single-element list (regular training)
        """
        if self.cross_validation:
            self.history = []

            # Create stratified k-fold splits
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
            splits = list(skf.split(x_train, y_train))

            # Store initial model configuration for resetting between folds
            initial_config = self.config.config_model

            # Perform stratified k-fold cross-validation with progress bar
            pbar = tqdm(
                enumerate(splits),
                total=self.n_splits,
                desc="[Pipeline] Training Fold 1",
                unit="fold",
                colour="#0a2c53",
            )
            for fold, (train_idx, val_idx) in pbar:
                # Updates the bar description for the current fold
                pbar.set_description_str(f"[Pipeline] Training Fold {fold + 1}")

                # Reinitialize model for each fold (fresh start)
                if isinstance(self.model, MLP):
                    self.model = self._get_model(initial_config)
                    # Reinitialize optimizer with new model parameters
                    self.optimizer = self._get_optimizer(self.config.optimizer)

                # Split data for current fold
                x_train_fold = self._select_rows(x_train, train_idx)
                y_train_fold = self._select_rows(y_train, train_idx)
                x_val_fold = self._select_rows(x_train, val_idx)
                y_val_fold = self._select_rows(y_train, val_idx)

                # Train on current fold and store history
                fold_history = self.call_trainer(
                    x_train_fold,
                    y_train_fold,
                    x_val=x_val_fold,
                    y_val=y_val_fold,
                    **kwargs,
                )
                self.history.append(fold_history)

        elif x_val is not None and y_val is not None:
            # Use provided validation data
            self.history = [
                self.call_trainer(x_train, y_train, x_val=x_val, y_val=y_val, **kwargs)
            ]
        else:
            # Automatically split training data for validation
            x_train, y_train, x_val, y_val = self.holdout(x_train, y_train)
            self.history = [
                self.call_trainer(x_train, y_train, x_val=x_val, y_val=y_val, **kwargs)
            ]

    def holdout(self, X, Y, test_size=None):
        """
        Split any dataset into two subsets using holdout.

        Args:
            X (array-like): Features.
            Y (array-like): Targets.
            test_size (float, optional): Fraction of data for the second split.
                If None, uses self.val_size. Should be between 0 and 1.

        Returns:
            Tuple: (X_train, Y_train, X_holdout, Y_holdout)
        """
        if test_size is None:
            test_size = self.val_size

        X_train, X_holdout, Y_train, Y_holdout = train_test_split(
            X,
            Y,
            test_size=test_size,
            shuffle=self.shuffle_train,
            random_state=self.seed,
        )
        return X_train, Y_train, X_holdout, Y_holdout

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

    def call_trainer(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Any = None,
        y_val: Any = None,
        **kwargs,
    ) -> dict[str, list[Any]] | None:
        """Call the appropriate training method based on model type.

        Dispatches training to either PyTorch neural network training loop
        or scikit-learn model fitting, handling the different interfaces
        transparently.

        Args:
            x_train (Any): Training input features.
            y_train (Any): Training target labels.
            x_val (Any, optional): Validation input features.
            y_val (Any, optional): Validation target labels.
            **kwargs: Additional arguments passed to the model's fit method.

        Returns:
            dict[str, list[Any]] | None: Training history dictionary for PyTorch
                models (containing loss/metric trajectories), or None for
                scikit-learn models.
        """
        if isinstance(self.model, MLP):
            # PyTorch model training
            train_loader = self._create_dataloader(
                x_train, y_train, shuffle=self.shuffle_train
            )
            val_loader = (
                self._create_dataloader(x_val, y_val, shuffle=False)
                if x_val is not None and y_val is not None
                else None
            )

            # Use configured optimizer/criterion or fallback to defaults
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
                device=self.device,
            )
        else:
            # Scikit-learn model training
            return self.model.fit(x_train, y_train, **kwargs)

    def save(self, filepath: Path) -> None:
        """Save the trained model to disk.

        Uses the ModelRecorder utility to save model checkpoints in a
        consistent format across different model types.

        Args:
            filepath (Path): The file path where the model should be saved.
                The extension should be appropriate for the model type
                (.pth for PyTorch models, .pkl for scikit-learn models).

        Note:
            The saved model can be loaded later using the load() method.
        """
        ModelRecorder.save_best_model(model=self.model, filename=filepath)

    def load(self, filepath: Path) -> MLP | SklearnModels:
        """Load a previously saved model from disk.

        Loads model weights/parameters from the specified file and applies
        them to the current model instance.

        Args:
            filepath (Path): The file path from which to load the model.
                Should contain a model saved with the save() method.

        Returns:
            MLP | SklearnModels: The model instance with loaded weights/parameters.

        Note:
            For PyTorch models, this loads the state dict. For scikit-learn
            models, this loads the entire model state.
        """
        state_dict = ModelRecorder.load_model(filename=filepath)
        if isinstance(self.model, MLP):
            self.model.load_state_dict(state_dict)
        return self.model

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
