"""Base trainer class with framework-agnostic training logic."""

import random
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from .base_dataset import BaseDataset
from .base_models import BaseModels
from .base_instantiable import Instantiable
from .dataset_outputs import DatasetOutputs

logger = logging.getLogger(__name__)


class TrainingResult(BaseModel):
    """
    Container for training results.

    Attributes:
        model: Trained model instance.
        history: Training history with metrics per epoch/iteration.
        train_dataset_size: Number of events in training dataset.
        val_dataset_size: Number of events in validation dataset (0 if no validation).
        metadata: Additional metadata about training.
    """

    model: BaseModels
    history: dict[str, Any]
    train_dataset_size: int
    val_dataset_size: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseTrainerConfig(BaseModel, Instantiable):
    """Base configuration for all trainers."""

    seed: int = Field(default=42, description="Random seed for reproducibility")
    use_class_weights: bool = Field(
        default=False,
        description="Whether to compute class weights for imbalanced data",
    )
    class_weight_strategy: str = Field(
        default="balanced", description="Class weight strategy: 'balanced' or 'manual'"
    )
    manual_class_weights: dict[int, float] | None = Field(
        default=None, description="Manual class weights (required if strategy='manual')"
    )


class BaseTrainer(ABC):
    """
    Abstract base trainer with framework-agnostic logic.

    This class provides common functionality for all trainers:
    - Dataset validation
    - Random seed setting
    - Class weight computation
    - Training flow orchestration

    Subclasses (TorchTrainer, SklearnTrainer) implement framework-specific
    data preparation and training execution.

    Usage:
        # Subclass must implement abstract methods
        class MyTrainer(BaseTrainer):
            def _prepare_data_for_training(self, dataset):
                # Convert dataset to framework-specific format
                ...

            def _execute_training(self, model, train_data, val_data):
                # Run training loop
                ...

        # Use the trainer
        trainer = MyTrainer(config)
        result = trainer.train(train_dataset, val_dataset)
    """

    model: BaseModels  # Set by subclass during training

    def __init__(self, config: BaseTrainerConfig):
        """
        Initialize the base trainer.

        Args:
            config: Trainer configuration with common parameters.
        """
        self.config = config
        logger.info("Initialized %s with seed=%d", self.__class__.__name__, config.seed)

    def train(
        self, train_dataset: BaseDataset, val_dataset: BaseDataset | None = None
    ) -> TrainingResult:
        """
        Train a model on the provided datasets.

        This method orchestrates the training process:
        1. Validate datasets
        2. Set random seeds for reproducibility
        3. Prepare data (framework-specific)
        4. Execute training (framework-specific)
        5. Return results

        Args:
            train_dataset: Training dataset (already preprocessed/transformed).
            val_dataset: Optional validation dataset.

        Returns:
            TrainingResult containing trained model, history, and metadata.

        Raises:
            ValueError: If datasets are incompatible or invalid.

        Example:
            >>> train_ds = ParquetDatasetConfig(split="train", ...).build()
            >>> val_ds = ParquetDatasetConfig(split="val", ...).build()
            >>> trainer = TorchTrainer(config)
            >>> result = trainer.train(train_ds, val_ds)
            >>> print(f"Training loss: {result.history['train_loss'][-1]}")
        """
        logger.info(
            "Starting training | train_size=%d | val_size=%s",
            len(train_dataset),
            len(val_dataset) if val_dataset else "None",
        )

        # 1. Validate datasets
        self._validate_datasets(train_dataset, val_dataset)

        # 2. Set random seeds
        self._set_random_seeds(self.config.seed)

        # 3. Prepare data (framework-specific)
        logger.info("Preparing training data...")
        train_data = self._prepare_data_for_training(train_dataset)
        val_data = None
        if val_dataset is not None:
            logger.info("Preparing validation data...")
            val_data = self._prepare_data_for_training(val_dataset)

        # 4. Execute training (framework-specific)
        logger.info("Executing training...")
        history = self._execute_training(train_data, val_data)

        # 5. Return results
        result = TrainingResult(
            model=self.model,
            history=history,
            train_dataset_size=len(train_dataset),
            val_dataset_size=len(val_dataset) if val_dataset else 0,
            metadata={
                "trainer_type": self.__class__.__name__,
                "seed": self.config.seed,
            },
        )

        self._training_result = result

        logger.info("Training completed successfully")
        return result

    def _validate_datasets(
        self, train_dataset: BaseDataset, val_dataset: BaseDataset | None
    ) -> None:
        """
        Validate that train and validation datasets are compatible.

        Checks:
        - Datasets are not empty
        - First events have consistent structure (signal columns, label presence)
        - If validation dataset provided, it matches training dataset structure

        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.

        Raises:
            ValueError: If datasets are invalid or incompatible.
        """
        # Check train dataset not empty
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")

        # Get first event to check structure
        first_train_event = train_dataset[0]

        if not isinstance(first_train_event, DatasetOutputs):
            raise ValueError(
                f"Dataset must return DatasetOutputs, got {type(first_train_event)}"
            )

        # If validation dataset provided, check compatibility
        if val_dataset is not None:
            if len(val_dataset) == 0:
                raise ValueError("Validation dataset is empty")

            first_val_event = val_dataset[0]

            # Check signal columns match
            train_cols = set(first_train_event.signal.columns)
            val_cols = set(first_val_event.signal.columns)

            if train_cols != val_cols:
                raise ValueError(
                    f"Train and val datasets have different signal columns. "
                    f"Train: {train_cols}, Val: {val_cols}"
                )

            # Check both have labels or both don't
            train_has_label = first_train_event.label is not None
            val_has_label = first_val_event.label is not None

            if train_has_label != val_has_label:
                raise ValueError(
                    "Train and val datasets must both have labels or both not have labels"
                )

        logger.info("Dataset validation passed")

    def _set_random_seeds(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.

        Sets seeds for:
        - Python random module
        - NumPy
        - PyTorch (CPU and CUDA)

        Args:
            seed: Random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # For deterministic behavior (may impact performance)
            if hasattr(self.config, "deterministic") and self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        logger.debug("Random seeds set to %d", seed)

    def _compute_class_weights(self, dataset: BaseDataset) -> dict[int, float]:
        """
        Compute class weights from dataset for handling class imbalance.

        Iterates over dataset to collect all labels, then computes
        weights using sklearn's compute_class_weight with 'balanced' strategy.

        Args:
            dataset: Dataset to compute weights from.

        Returns:
            Dictionary mapping class labels to weights.

        Raises:
            ValueError: If dataset has no labels or strategy is invalid.

        Example:
            >>> weights = trainer._compute_class_weights(train_dataset)
            >>> # {0: 1.5, 1: 0.75}  # Class 0 is underrepresented
        """
        if self.config.class_weight_strategy == "manual":
            if self.config.manual_class_weights is None:
                raise ValueError(
                    "manual_class_weights must be provided when strategy='manual'"
                )
            return self.config.manual_class_weights

        # Collect all labels from dataset
        all_labels: list[int | float] = []
        for event in dataset:
            if event.label is None:
                raise ValueError("Cannot compute class weights: dataset has no labels")
            all_labels.extend(event.label.values.tolist())

        labels_array = np.array(all_labels)
        unique_classes = np.unique(labels_array)

        # Compute balanced weights
        from sklearn.utils.class_weight import compute_class_weight

        weights = compute_class_weight(
            class_weight="balanced", classes=unique_classes, y=labels_array
        )

        class_weight_dict = {
            int(cls): float(weight) for cls, weight in zip(unique_classes, weights)
        }

        logger.info("Computed class weights: %s", class_weight_dict)
        return class_weight_dict

    @abstractmethod
    def _prepare_data_for_training(self, dataset: BaseDataset) -> Any:
        """
        Prepare dataset for training (framework-specific).

        Subclasses implement this to convert BaseDataset to the format
        required by their framework:
        - TorchTrainer: Convert to DataLoader
        - SklearnTrainer: Convert to (X, y) numpy arrays

        Args:
            dataset: Dataset to prepare.

        Returns:
            Framework-specific data structure ready for training.
        """
        pass

    @abstractmethod
    def _execute_training(
        self, train_data: Any, val_data: Any | None
    ) -> dict[str, Any]:
        """Execute the training loop (framework-specific)."""
        pass

    def evaluate(
        self,
        test_dataset: BaseDataset,
        assessment_config: Any | None = None,
    ) -> Any:
        """Evaluate the trained model using ModelAssessment.

        Args:
            test_dataset: Test dataset for evaluation.
            assessment_config: Optional ModelAssessmentConfig. If None, uses defaults.

        Returns:
            AssessmentOutput with evaluation results.
        """
        from ..assessment.model_assess import ModelAssessment
        from ..core.base_assessment import ModelAssessmentConfig
        from ..core.enums import TaskTypeEnum, DataSplitEnum

        if not hasattr(self, "_training_result") or self._training_result is None:
            raise RuntimeError(
                "Model must be trained before evaluation. Call train() first."
            )

        if assessment_config is None:
            assessment_config = ModelAssessmentConfig(
                metrics=["accuracy"],
                task_type=TaskTypeEnum.CLASSIFICATION,
                dataset_split=DataSplitEnum.TEST,
            )

        assessor = ModelAssessment(
            trainer=self,
            training_result=self._training_result,
            config=assessment_config,
        )
        return assessor.evaluate(test_dataset)
