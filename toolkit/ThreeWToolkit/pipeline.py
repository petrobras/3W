"""Pipeline for orchestrating the complete ML workflow."""

import logging
from typing import Any
from pydantic import Field, PrivateAttr

from .core.base_pipeline import BasePipeline, BasePipelineConfig, PipelineResult
from .core.base_dataset import BaseDataset, BaseDatasetConfig
from .core.base_trainer import BaseTrainer, BaseTrainerConfig, TrainingResult
from .core.enums import TaskTypeEnum, DataSplitEnum
from .assessment import ModelAssessment, ModelAssessmentConfig

logger = logging.getLogger(__name__)


class PipelineConfig(BasePipelineConfig):
    """
    Configuration for the ML pipeline.

    This config consolidates all component configurations, allowing the
    Pipeline to instantiate all necessary components from their configs.
    """

    # Component configs
    train_dataset_config: BaseDatasetConfig = Field(
        ..., description="Configuration for training dataset."
    )
    trainer_config: BaseTrainerConfig = Field(
        ..., description="Configuration for the model trainer."
    )
    test_dataset_config: BaseDatasetConfig | None = Field(
        default=None, description="Configuration for test dataset (optional)."
    )
    val_dataset_config: BaseDatasetConfig | None = Field(
        default=None, description="Configuration for validation dataset (optional)."
    )
    transform_config: Any | None = Field(
        default=None,
        description="Configuration for data transformation (optional). Should be BaseTransformConfig.",
    )

    # Experiment settings
    experiment_name: str = Field(
        default="experiment", description="Name for this experiment run."
    )

    # Task settings
    task_type: TaskTypeEnum = Field(
        default=TaskTypeEnum.CLASSIFICATION, description="Type of ML task."
    )
    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy"], description="Metrics to compute."
    )
    generate_report: bool = Field(
        default=True, description="Whether to generate assessment report."
    )

    _target: type = PrivateAttr(default_factory=lambda: Pipeline)
    model_config = {"arbitrary_types_allowed": True}


class Pipeline(BasePipeline):
    """
    ML pipeline orchestrating data loading, transformation, training, and evaluation.

    The pipeline follows a simple flow:
    1. Build components from their configs
    2. Load data using a dataset (e.g., ParquetDataset)
    3. Transform data using preprocessing and feature extraction
    4. Train a model using a trainer (TorchTrainer or SklearnTrainer)
    5. Evaluate the model using ModelAssessment

    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline from a configuration.

        All components (datasets, trainer, transform) are built from their
        respective configs during initialization.

        Args:
            config: Pipeline configuration containing all component configs.
        """
        self.config = config

        # Build components from configs
        self.train_dataset: BaseDataset = config.train_dataset_config.build()
        self.trainer: BaseTrainer = config.trainer_config.build()

        self.test_dataset: BaseDataset | None = (
            config.test_dataset_config.build()
            if config.test_dataset_config is not None
            else None
        )
        self.val_dataset: BaseDataset | None = (
            config.val_dataset_config.build()
            if config.val_dataset_config is not None
            else None
        )
        self.transform: Any | None = (
            config.transform_config.build()
            if config.transform_config is not None
            else None
        )

        self._training_result: TrainingResult | None = None
        self._assessment_output: Any | None = None  # AssessmentOutput at runtime

        logger.info("Pipeline initialized | experiment=%s", config.experiment_name)

    def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.

        Returns:
            PipelineResult containing training and assessment outputs.
        """
        logger.info("Starting pipeline execution")

        # Step 1: Fit and transform data
        train_data = self._prepare_data(self.train_dataset, fit=True)
        if train_data is None:
            raise RuntimeError("Failed to prepare training data")

        val_data = self._prepare_data(self.val_dataset) if self.val_dataset else None
        test_data = self._prepare_data(self.test_dataset) if self.test_dataset else None

        # Step 2: Train model
        logger.info("Training model...")
        self._training_result = self.trainer.train(train_data, val_data)
        logger.info("Training complete")

        # Step 3: Evaluate model (if test data provided)
        if test_data is not None:
            logger.info("Evaluating model...")
            self._assessment_output = self._evaluate(test_data)
            logger.info("Evaluation complete")

        return PipelineResult(
            training_result=self._training_result,
            assessment_output=self._assessment_output,
            experiment_name=self.config.experiment_name,
            metadata={
                "train_size": len(self.train_dataset),
                "val_size": len(self.val_dataset) if self.val_dataset else 0,
                "test_size": len(self.test_dataset) if self.test_dataset else 0,
            },
        )

    def _prepare_data(
        self, dataset: BaseDataset | None, fit: bool = False
    ) -> BaseDataset | None:
        """Apply transformation to dataset."""
        if dataset is None:
            return None

        if self.transform is None:
            return dataset

        if fit:
            logger.info("Fitting transform on training data...")
            self.transform.fit(dataset)

        return self.transform.transform(dataset)

    def _evaluate(self, test_data: BaseDataset) -> Any:
        """Run model evaluation."""
        # Lazy import to avoid pydantic/numpy compatibility issue with metrics

        if self._training_result is None:
            raise RuntimeError("Model must be trained before evaluation")

        assessment_config = ModelAssessmentConfig(
            metrics=self.config.metrics,
            task_type=self.config.task_type,
            dataset_split=DataSplitEnum.TEST,
            generate_report=self.config.generate_report,
        )

        assessor = ModelAssessment(
            trainer=self.trainer,
            training_result=self._training_result,
            config=assessment_config,
        )

        return assessor.evaluate(test_data)

    def train(self, val_dataset: BaseDataset | None = None) -> TrainingResult:
        """
        Train the model only.

        Args:
            val_dataset: Optional validation dataset (overrides constructor value).

        Returns:
            TrainingResult from training.
        """
        train_data = self._prepare_data(self.train_dataset, fit=True)
        if train_data is None:
            raise RuntimeError("Failed to prepare training data")

        val_data = val_dataset or self.val_dataset
        if val_data is not None:
            val_data = self._prepare_data(val_data)

        self._training_result = self.trainer.train(train_data, val_data)
        return self._training_result

    def evaluate(self, test_dataset: BaseDataset | None = None) -> Any:
        """
        Evaluate the trained model.

        Args:
            test_dataset: Optional test dataset (overrides constructor value).

        Returns:
            AssessmentOutput from evaluation.

        Raises:
            RuntimeError: If model hasn't been trained.
        """
        if self._training_result is None:
            raise RuntimeError(
                "Model must be trained before evaluation. Call train() first."
            )

        test_data = test_dataset or self.test_dataset
        if test_data is None:
            raise ValueError("No test dataset provided")

        test_data = self._prepare_data(test_data)
        if test_data is None:
            raise ValueError("Test data preparation failed")

        self._assessment_output = self._evaluate(test_data)
        return self._assessment_output

    @property
    def model(self):
        """Get the trained model."""
        if self._training_result is None:
            return None
        return self._training_result.model

    @property
    def training_history(self) -> dict[str, Any] | None:
        """Get training history."""
        if self._training_result is None:
            return None
        return self._training_result.history

    @property
    def assessment_results(self) -> Any | None:
        """Get assessment results."""
        return self._assessment_output
