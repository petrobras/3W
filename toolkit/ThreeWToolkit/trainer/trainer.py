import pandas as pd
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Mapping, TypeAlias
from pydantic import Field, field_validator, model_validator
from dataclasses import dataclass, field

from tqdm.auto import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ..core.base_step import BaseStep
from ..core.base_models import BaseModels
from ..core.protocols import SupportsOptimizerParams
from ..core.base_model_trainer import ModelTrainerConfig
from ..core.enums import OptimizersEnum, CriterionEnum, TaskTypeEnum
from ..models.mlp import MLPConfig
from ..models.sklearn_models import SklearnModelsConfig
from ..assessment.model_assess import (
    AssessmentInput,
    AssessmentOutput,
    ModelAssessment,
    ModelAssessmentConfig,
)

ArrayLike: TypeAlias = pd.DataFrame | pd.Series | np.ndarray

logger = logging.getLogger(__name__)


class TrainerConfig(ModelTrainerConfig):
    """
    Configuration object for the training pipeline.

    Defines hyperparameters, data splitting strategy, optimization settings,
    and task-specific behavior.

    Attributes:
        batch_size (int): Number of samples per batch. Must be > 0.
        epochs (int): Number of training epochs. Must be > 0.
        seed (int): Random seed for reproducibility.
        learning_rate (float): Optimizer learning rate. Must be > 0.
        config_model (MLPConfig | SklearnModelsConfig): Model configuration.
        criterion (str): Loss function name. Must exist in CriterionEnum.
        optimizer (str): Optimizer name. Must exist in OptimizersEnum.
        device (str): "cpu" or "cuda". If "cuda", GPU must be available.
        cross_validation (bool | None): Enable k-fold cross-validation.
        n_splits (int): Number of folds when cross_validation=True. Must be > 1.
        test_size (float): Test split proportion (0 < test_size < 1).
        val_size (float): Validation split proportion (0 < val_size < 1).
        shuffle_train (bool): Shuffle training data.
        task_type (TaskTypeEnum): Classification or regression.
        use_class_weights (bool): Enable class weighting.
        class_weight_strategy (str): "balanced" or "manual".
        manual_class_weights (dict[int, float] | None): Required if strategy="manual".
        deterministic (bool): Enable deterministic CUDA behavior.

    Raises:
        ValueError: If configuration values are inconsistent or invalid.
    """

    batch_size: int = Field(default=32, gt=0)
    epochs: int = Field(default=50, gt=0)
    seed: int = Field(default=42)
    learning_rate: float = Field(default=1e-3, gt=0)

    config_model: MLPConfig | SklearnModelsConfig = Field(...)

    criterion: str = Field(default="cross_entropy")
    optimizer: str = Field(default="adam")

    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")

    cross_validation: bool | None = Field(default=None)
    n_splits: int = Field(default=5, gt=1)

    test_size: float = Field(default=0.2, gt=0, lt=1)
    val_size: float = Field(default=0.3, gt=0, lt=1)

    shuffle_train: bool = Field(default=True)

    task_type: TaskTypeEnum = Field(default=TaskTypeEnum.CLASSIFICATION)

    use_class_weights: bool = Field(default=False)
    class_weight_strategy: str = Field(default="balanced")
    manual_class_weights: dict[int, float] | None = Field(default=None)

    deterministic: bool = Field(default=False)

    @field_validator("optimizer")
    @classmethod
    def check_optimizer(cls: type["TrainerConfig"], value: str) -> str:
        """Validate that optimizer is from the supported list.

        Args:
            cls (TrainerConfig): The class reference.
            value (str): The optimizer name to validate.

        Returns:
            str: The validated optimizer name.

        Raises:
            ValueError: If optimizer is not in the supported list.
        """
        valid = {o.value for o in OptimizersEnum}
        if value not in valid:
            raise ValueError(f"optimizer must be one of {valid}")
        return value

    @field_validator("criterion")
    @classmethod
    def check_criterion(cls: type["TrainerConfig"], value: str) -> str:
        """Validate that criterion is from the supported list.

        Args:
            cls (TrainerConfig): The class reference.
            value (str): The criterion name to validate.

        Returns:
            str: The validated criterion name.

        Raises:
            ValueError: If criterion is not in the supported list.
        """
        valid = {c.value for c in CriterionEnum}
        if value not in valid:
            raise ValueError(f"criterion must be one of {valid}")
        return value

    @field_validator("device")
    @classmethod
    def check_device(cls: type["TrainerConfig"], value: str) -> str:
        """
        Validate computation device selection.

        Ensures that the device is either 'cpu' or 'cuda'.
        If 'cuda' is selected, verifies that CUDA is available.

        Args:
            value (str): Device name.

        Returns:
            str: Validated device name.

        Raises:
            ValueError: If device is invalid or CUDA is unavailable.
        """
        valid = {"cpu", "cuda"}
        if value not in valid:
            raise ValueError("device must be 'cpu' or 'cuda'")

        if value == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA selected but no GPU is available")

        return value

    @field_validator("class_weight_strategy")
    @classmethod
    def validate_class_weight_strategy(cls, value: str) -> str:
        """
        Validate class weight strategy.

        Ensures that the strategy is either 'balanced' or 'manual'.

        Args:
            value (str): Strategy name.

        Returns:
            str: Validated strategy name.

        Raises:
            ValueError: If strategy is not supported.
        """
        valid = {"balanced", "manual"}
        if value not in valid:
            raise ValueError(f"class_weight_strategy must be one of {valid}")
        return value

    @model_validator(mode="after")
    def validate_class_weights(self) -> "TrainerConfig":
        """
        Validate manual class weights configuration.

        Ensures that when class weighting is enabled and the strategy
        is set to 'manual', a valid dictionary of class weights is provided.

        The dictionary must:
            - Have integer keys (class labels)
            - Have strictly positive float values (weights)

        Returns:
            TrainerConfig: The validated configuration instance.

        Raises:
            ValueError: If manual weights are missing or invalid.
        """
        if self.use_class_weights and self.class_weight_strategy == "manual":
            if not self.manual_class_weights:
                raise ValueError(
                    "manual_class_weights must be provided when strategy='manual'"
                )

            if any(v <= 0 for v in self.manual_class_weights.values()):
                raise ValueError("manual_class_weights values must be > 0")

        return self


@dataclass
class TrainInput:
    """
    Container for training input data.

    This dataclass standardizes the data interface passed to the training step,
    avoiding the use of positional indexing and improving type safety.

    Attributes:
        x_train (pd.DataFrame | np.ndarray): Training input features.
        y_train (pd.DataFrame | np.ndarray): Training target labels.
        x_val (pd.DataFrame | np.ndarray | None): Optional validation input features.
        y_val (pd.DataFrame | np.ndarray | None): Optional validation target labels.
    """

    x_train: pd.DataFrame | np.ndarray
    y_train: pd.DataFrame | np.ndarray

    x_val: pd.DataFrame | np.ndarray | None = None
    y_val: pd.DataFrame | np.ndarray | None = None


@dataclass
class TrainOutput:
    """
    Container for training output results.

    This dataclass encapsulates the artifacts produced by the training process
    and metadata describing the training execution.

    Attributes:
        history (dict[str, list[float]]): Training history containing tracked loss values per epoch.
        model_type (str | None): Name of the trained model class.
        trainer_config (TrainerConfig | None): Trainer configuration used during training.
        training_completed (bool): Flag indicating whether training finished successfully.
    """

    models: list[BaseModels] = field(default_factory=list)
    x_train: list[np.ndarray] = field(default_factory=list)
    y_train: list[np.ndarray] = field(default_factory=list)
    x_val: list[np.ndarray] = field(default_factory=list)
    y_val: list[np.ndarray] = field(default_factory=list)

    model_name: str | None = None
    trainer_config: TrainerConfig | None = None
    training_completed: bool = False

    history: dict[str, list[float]] | None = None


class ModelTrainer(BaseStep):
    def __init__(self, config: TrainerConfig) -> None:
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
        self.history = self._get_artifacts_history_dict()

    def pre_process(self, data: TrainInput) -> TrainInput:
        """
        Validate and standardize the training input.

        This method ensures that the input follows the expected
        TrainInput structure and validates basic consistency rules.

        Args:
            data (TrainInput): Input object expected to be a TrainInput instance.

        Returns:
            TrainInput: Validated training input container.

        Raises:
            TypeError: If the input is not a TrainInput instance.
            ValueError: If mandatory fields are missing or inconsistent.
        """
        # Case 1: Already a TrainInput instance
        if isinstance(data, TrainInput):
            train_input = data
        else:
            raise TypeError(
                "Trainer input must be a TrainInput instance containing at least 'x_train' and 'y_train'."
            )

        # Basic consistency validation
        if train_input.x_train is None or train_input.y_train is None:
            raise ValueError("x_train and y_train must not be None.")

        # XOR check: both validation sets must be provided together
        if (train_input.x_val is None) ^ (train_input.y_val is None):
            raise ValueError(
                "x_val and y_val must be provided together or both be None."
            )

        return train_input

    def run(self, data: TrainInput) -> TrainOutput:
        """
        Execute the training process.

        This method performs the actual model training using the
        preprocessed training data.

        Args:
            data (TrainInput): Validated training input container.

        Returns:
            TrainOutput: Object containing training artifacts and results.
        """
        self._set_seed()

        logger.info("Training started")

        # Perform training
        self.train(
            x_train=data.x_train,
            y_train=data.y_train,
            x_val=data.x_val,
            y_val=data.y_val,
        )

        return TrainOutput(
            models=self.history["models"],
            x_train=self.history["x_train"],
            y_train=self.history["y_train"],
            x_val=self.history["x_val"],
            y_val=self.history["y_val"],
        )

    def post_process(self, data: TrainOutput) -> TrainOutput:
        """
        Finalize and enrich training output.

        This method attaches metadata related to the training process
        and ensures output consistency before passing it to the next
        pipeline step.

        Args:
            data (TrainOutput): Training output container.

        Returns:
            TrainOutput: Finalized training output with metadata added.
        """
        # Add metadata about the training step
        data.model_name = self.history["models"][0].model_name
        data.trainer_config = self.config
        data.history = {k: self.history[k] for k in ["train_losses", "val_losses"]}

        return data

    def train(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
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
        """
        if self.cross_validation:
            self._train_with_cross_validation(x_train, y_train)
        elif x_val is not None and y_val is not None:
            self._train_with_validation(x_train, y_train, x_val, y_val)
        else:
            self._train_with_auto_split(x_train, y_train)

    def _train_with_cross_validation(
        self, x_train: ArrayLike, y_train: ArrayLike
    ) -> None:
        """Train using k-fold cross-validation.

        Args:
            x_train: Training features.
            y_train: Training labels.
        """
        if self.config.task_type == TaskTypeEnum.CLASSIFICATION:
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.config.shuffle_train,
                random_state=self.seed,
            )
        else:
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=self.config.shuffle_train,
                random_state=self.seed,
            )

        splits = list(kf.split(x_train, y_train))

        logger.info("Cross-validation enabled | n_splits=%d", self.n_splits)

        pbar = tqdm(
            enumerate(splits),
            total=self.n_splits,
            desc="[Pipeline] Training Fold 1",
            unit="fold",
            colour="#0a2c53",
        )

        for fold, (train_idx, val_idx) in pbar:
            logger.info("Fold %d/%d started", fold + 1, self.n_splits)
            pbar.set_description_str(f"[Pipeline] Training Fold {fold + 1}")

            # Split data
            x_train_fold = self._select_rows(x_train, train_idx)
            y_train_fold = self._select_rows(y_train, train_idx)
            x_val_fold = self._select_rows(x_train, val_idx)
            y_val_fold = self._select_rows(y_train, val_idx)

            # Train on fold
            results = self._call_training_strategy(
                x_train_fold, y_train_fold, x_val_fold, y_val_fold
            )

            self._update_train_history(
                results, x_train_fold, y_train_fold, x_val_fold, y_val_fold
            )

    def _train_with_validation(
        self, x_train: ArrayLike, y_train: ArrayLike, x_val: ArrayLike, y_val: ArrayLike
    ) -> None:
        """Train using provided validation data.

        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
        """
        logger.info("Using provided validation set")

        results = self._call_training_strategy(x_train, y_train, x_val, y_val)

        self._update_train_history(results, x_train, y_train, x_val, y_val)

    def _train_with_auto_split(self, x_train: ArrayLike, y_train: ArrayLike) -> None:
        """Train with automatic train/validation split.

        Args:
            x_train: Training features.
            y_train: Training labels.
        """
        logger.info("Using auto split set")

        x_train, x_val, y_train, y_val = self._holdout(x_train, y_train)

        results = self._call_training_strategy(x_train, y_train, x_val, y_val)

        self._update_train_history(results, x_train, y_train, x_val, y_val)

    def _call_training_strategy(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        x_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
    ) -> dict[str, list[float]]:
        """Delegate training to the strategy.

        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features (optional).
            y_val: Validation labels (optional).

        Returns:
            Training history or None.
        """
        # Prepare kwargs for strategy
        strategy_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle_train,
            "device": self.device,
        }

        model = self.config.config_model.setup()

        training_strategy = model.get_training_strategy()
        strategy = training_strategy()

        # Reinitialize optimizer if model requires it
        if strategy.requires_optimizer:
            strategy_kwargs["optimizer"] = self._get_optimizer(
                model, self.config.optimizer
            )

        if strategy.requires_criterion:
            strategy_kwargs["criterion"] = self._get_fn_cost(
                self.config.criterion,
                y_train=y_train,
            )

        logger.info("Training started")

        return strategy.train(model, x_train, y_train, x_val, y_val, **strategy_kwargs)

    def _holdout(
        self, X: ArrayLike, y: ArrayLike, test_size=None
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Split dataset using holdout.

        Args:
            X: Features.
            y: Targets.
            test_size: Fraction for validation split.

        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        if test_size is None:
            test_size = self.val_size

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=self.shuffle_train,
            random_state=self.seed,
        )
        return X_train, X_val, y_train, y_val

    def _get_optimizer(
        self, model: BaseModels, optimizer: str
    ) -> torch.optim.Optimizer:
        """Create and return the specified PyTorch optimizer.

        Args:
            model (BaseModels): PyTorch model instance.
            optimizer (str): Name of the optimizer to create. Must be one of:
                "adam", "adamw", "sgd", "rmsprop".

        Returns:
            torch.optim.Optimizer: The configured optimizer instance with
                model parameters and learning rate set.

        Raises:
            ValueError: If the optimizer name is not recognized.
        """
        if not isinstance(model, SupportsOptimizerParams):
            raise TypeError(
                f"Optimizer can only be created for models exposing "
                f"'get_params()'. Received: {type(model).__name__}"
            )

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

    def _get_fn_cost(
        self, criterion: str | None, y_train: ArrayLike | None = None
    ) -> Callable:
        """
        Create and return the configured loss function.

        This method instantiates the appropriate PyTorch loss function based on
        the selected criterion and training configuration.

        When class weighting is enabled (`use_class_weights=True`) and the task
        is classification, the method automatically computes and injects class
        weights into the loss function using the training labels.

        Supported behaviors:
            - Multi-class classification:
                Uses `nn.CrossEntropyLoss(weight=class_weights)`
            - Binary classification:
                Uses `nn.BCEWithLogitsLoss(pos_weight=class_weights)`
            - Regression:
                Uses `nn.MSELoss` or `nn.L1Loss`

        Args:
            criterion (str | None):
                Name of the loss function. Must be one of:
                {"cross_entropy", "binary_cross_entropy", "mse", "mae"}.

            y_train (ArrayLike | None):
                Training labels used to compute class weights when
                class weighting is enabled. Required only for classification
                tasks with `use_class_weights=True`.

        Returns:
            Callable:
                Instantiated PyTorch loss function ready for training.

        Raises:
            ValueError:
                If the specified criterion name is not supported.

            ValueError:
                If class weighting is enabled but `y_train` is not provided.

        Notes:
            - `weight` in CrossEntropyLoss applies per-class weighting for
            multi-class classification.

            - `pos_weight` in BCEWithLogitsLoss controls positive class weighting
            in binary classification and expects a tensor of size [1] or
            matching output dimensions.

            - For regression losses, class weighting is ignored.
        """
        class_weights = None

        if (
            self.config.use_class_weights
            and self._is_classification_task()
            and y_train is not None
        ):
            class_weights = self._compute_class_weights(y_train)

        if criterion == CriterionEnum.CROSS_ENTROPY.value:
            return nn.CrossEntropyLoss(weight=class_weights)
        elif criterion == CriterionEnum.BINARY_CROSS_ENTROPY.value:
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif criterion == CriterionEnum.MSE.value:
            return nn.MSELoss()
        elif criterion == CriterionEnum.MAE.value:
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def _select_rows(self, data: ArrayLike, idx: int):
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
        x: ArrayLike,
        y: ArrayLike,
        assessment_config: ModelAssessmentConfig | None = None,
    ) -> AssessmentOutput:
        """Evaluate the trained model using ModelAssessment.

        This is a convenience method that creates a ModelAssessment instance
        and evaluates the current model on the provided test data.

        Args:
            x (ArrayLike): Input features for evaluation.
            y (ArrayLike): Target labels for evaluation.
            assessment_config (ModelAssessmentConfig | None, optional):
                Configuration for the assessment process. If None, creates
                a default configuration based on the task type.

        Returns:
            AssessmentOutput: Enriched assessment output.

        Example:
            >>> # Basic assessment with default configuration
            >>> results = trainer.assess(X_test, y_test)
            >>> print(f"Accuracy: {results['accuracy']}")
            >>>
            >>> # Custom assessment configuration
            >>> config = ModelAssessmentConfig(
            ...     metrics=["accuracy", "f1", "precision", "recall"],
            ...     task_type=TaskTypeEnum.CLASSIFICATION
            ... )
            >>> results = trainer.assess(X_test, y_test, assessment_config=config)

        Note:
            When cross-validation is enabled, all trained fold models
            are passed to the assessment pipeline. The assessment strategy
            is responsible for aggregating predictions if needed.
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
                    TaskTypeEnum.CLASSIFICATION
                    if self._is_classification_task()
                    else TaskTypeEnum.REGRESSION
                ),
            )

        assess_input = AssessmentInput(
            models=self.history["models"],
            x=x,
            y=y,
            dataset_split=assessment_config.dataset_split,
            x_train_folds=self.history["x_train"],
            y_train_folds=self.history["y_train"],
            x_val_folds=self.history["x_val"],
            y_val_folds=self.history["y_val"],
        )

        assessor = ModelAssessment(assessment_config)
        return assessor.evaluate(assess_input)

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
        return self.config.task_type == TaskTypeEnum.CLASSIFICATION

    def _update_train_history(self, results, x_train, y_train, x_val, y_val):
        """
        Update internal training history with artifacts from a training run.

        This method collects and stores outputs produced by the training strategy,
        including trained models, loss values, and the corresponding dataset
        partitions used during training. It is called after each training execution,
        including each fold during cross-validation.

        Args:
            results (dict[str, Any]): Dictionary returned by the training strategy.
                Expected keys may include:
                    - "model": Trained model instance.
                    - "train_loss": Training loss value.
                    - "val_loss": Validation loss value.
            x_train (Any): Training feature subset used in the current training run.
            y_train (Any): Training target subset used in the current training run.
            x_val (Any): Validation feature subset used in the current training run.
            y_val (Any): Validation target subset used in the current training run.

        Note:
            The method does not enforce strict key presence in `results` in order
            to remain compatible with different training strategy implementations.
            Only available artifacts are stored.

        Side Effects:
            Updates the internal `self.history` dictionary by appending new
            training artifacts and dataset splits.
        """
        if "model" in results:
            self.history["models"].append(results["model"])

        if "train_loss" in results:
            self.history["train_losses"].append(results["train_loss"])

        if "val_loss" in results:
            self.history["val_losses"].append(results["val_loss"])

        self.history["x_train"].append(x_train)
        self.history["y_train"].append(y_train)
        self.history["x_val"].append(x_val)
        self.history["y_val"].append(y_val)

    def _get_artifacts_history_dict(self) -> Mapping[str, list]:
        """
        Initialize and return the training artifacts history structure.

        This method creates the base dictionary used to store training artifacts
        throughout the execution of the training pipeline. The structure supports
        both single-run training and cross-validation by storing multiple entries
        per key.

        Returns:
            Mapping[str, list]: Initialized history dictionary containing empty
            lists for:
                - "models": Trained model instances.
                - "train_losses": Training loss values per run or fold.
                - "val_losses": Validation loss values per run or fold.
                - "x_train": Training feature subsets.
                - "y_train": Training target subsets.
                - "x_val": Validation feature subsets.
                - "y_val": Validation target subsets.

        Note:
            Each list grows dynamically as training progresses, with one entry
            added per training execution or fold.
        """
        return {
            "models": [],
            "train_losses": [],
            "val_losses": [],
            "x_train": [],
            "x_val": [],
            "y_train": [],
            "y_val": [],
        }

    def _set_seed(self) -> None:
        """
        Set global random seeds to improve experiment reproducibility.

        This method configures random number generators for NumPy and PyTorch
        (CPU and CUDA backends) to ensure deterministic behavior across
        training runs when possible.

        It also adjusts CuDNN backend settings to favor deterministic
        operations over performance-optimized but non-deterministic kernels.

        Notes:
            - Setting `torch.backends.cudnn.deterministic = True` may reduce
            training performance but improves reproducibility.
            - Some GPU operations remain non-deterministic depending on
            hardware and PyTorch version.
            - For full reproducibility, dataset shuffling and data loader
            worker seeds should also be controlled.

        Side Effects:
            Modifies global random generator states for NumPy and PyTorch,
            affecting all subsequent random operations in the current process.
        """
        seed = self.seed

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _compute_class_weights(self, y: ArrayLike) -> torch.Tensor:
        """
        Compute class weights for imbalanced classification problems.

        This method generates class weighting tensors to be used by loss
        functions such as CrossEntropyLoss in order to mitigate class
        imbalance during training.

        Two strategies are supported:

        - Manual:
            Uses user-provided weights defined in the trainer configuration.

        - Automatic ("balanced"):
            Computes weights using scikit-learn's `compute_class_weight`
            with the "balanced" strategy, which assigns higher weights
            to underrepresented classes.

        Args:
            y (ArrayLike): Target labels used to compute class distribution.
                Can be a NumPy array, pandas Series/DataFrame, or any array-like
                structure containing class indices.

        Returns:
            torch.Tensor: Tensor containing class weights ordered by class label,
                with dtype float32 and moved to the configured training device.

        Raises:
            ValueError: If manual weighting strategy is selected but
                `manual_class_weights` is not provided in the configuration.

        Notes:
            - For manual strategy, class weights are sorted by class label key
            to ensure correct alignment with model output indices.
            - For automatic strategy, class labels are inferred directly from `y`.
            - Returned tensor is placed on the same device used for training
            (CPU or GPU).
            - This method assumes a classification task with discrete labels.

        Example:
            Automatic balancing:
            >>> weights = trainer._compute_class_weights(y_train)
            >>> criterion = nn.CrossEntropyLoss(weight=weights)

            Manual balancing:
            >>> config.manual_class_weights = {0: 1.0, 1: 3.5}
            >>> weights = trainer._compute_class_weights(y_train)
        """
        if self.config.class_weight_strategy == "manual":
            if self.config.manual_class_weights is None:
                raise ValueError("manual_class_weights must be provided")

            weights = self.config.manual_class_weights
            return torch.tensor(
                [weights[k] for k in sorted(weights.keys())],
                dtype=torch.float32,
                device=self.device,
            )

        # automatic balanced
        y_np = np.asarray(y)

        classes = np.unique(y_np)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_np,
        )

        return torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=self.device,
        )
