"""Pipeline for orchestrating the complete ML workflow."""

import logging
from pydantic import Field, PrivateAttr

from .core.base_pipeline import BasePipeline, BasePipelineConfig, PipelineResult
from .core.base_dataset import BaseDataset, BaseDatasetConfig
from .core.base_trainer import BaseTrainer, BaseTrainerConfig, PredictionResult, TrainingResult
from .core.base_transform import BaseTransform, BaseTransformConfig
from .core.base_assessment import AssessmentOutput
from .core.enums import TaskTypeEnum
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
    test_dataset_config: BaseDatasetConfig | None = Field(
        default=None, description="Configuration for test dataset (optional)."
    )
    val_dataset_config: BaseDatasetConfig | None = Field(
        default=None, description="Configuration for validation dataset (optional)."
    )
    transform_config: BaseTransformConfig | None = Field(
        default=None,
        description="Configuration for data transformation (optional). Should be BaseTransformConfig.",
    )
    trainer_config: BaseTrainerConfig = Field(
        ..., description="Configuration for the model trainer."
    )
    assessment_config: ModelAssessmentConfig | None = Field(
        default=None,
        description="Configuration for model assessment (optional). If not provided, defaults will be used.",
    )
    save_results: bool = Field(
        default=False, description="Whether to save the trained model after training."
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
        self.trainer: BaseTrainer = config.trainer_config.build()

        self.train_dataset: BaseDataset = config.train_dataset_config.build()

        self.val_dataset: BaseDataset | None = (
            config.val_dataset_config.build()
            if config.val_dataset_config is not None
            else None
        )
        self.test_dataset: BaseDataset | None = (
            config.test_dataset_config.build()
            if config.test_dataset_config is not None
            else None
        )

        self.transform: BaseTransform | None = (
            config.transform_config.build()
            if config.transform_config is not None
            else None
        )

        self.assessment: ModelAssessment | None = (
            config.assessment_config.build()
            if config.assessment_config is not None
            else None
        )

        logger.info("Pipeline initialized | experiment=%s", config.experiment_name)

    def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.

        Returns:
            PipelineResult containing training and assessment outputs.
        """
        logger.info("Starting pipeline execution")

        training_result = self.train(self.train_dataset, self.val_dataset)
        if self.test_dataset is not None:
            predictions = self.predict(self.test_dataset)
        else:
            logger.debug("No test dataset provided, skipping prediction")
            predictions = None

        assessment_result = self.assess(training_result, predictions)

        if self.config.save_results:
            training_result.model.save(f"{self.config.experiment_name}_model.pth")
            logger.info("Model saved to %s_model.pth", self.config.experiment_name)

        return PipelineResult(
            training_result=training_result,
            assessment_output=assessment_result,
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

    def train(
        self, train_dataset: BaseDataset, val_dataset: BaseDataset | None
    ) -> TrainingResult:
        train_data = self._prepare_data(train_dataset, fit=True)
        if train_data is None:
            raise RuntimeError("Failed to prepare training data")

        val_data = self._prepare_data(val_dataset, fit=False) if val_dataset else None

        logger.info("Training model...")
        training_result = self.trainer.train(train_data, val_data)
        logger.info("Training complete")

        return training_result

    def predict(self, dataset: BaseDataset) -> PredictionResult:
        prepared_data = self._prepare_data(dataset, fit=False)
        if prepared_data is None:
            raise RuntimeError("Failed to prepare data for prediction")

        logger.info("Generating predictions...")
        predictions = self.trainer.predict(prepared_data)
        logger.info("Prediction complete")

        return predictions

    def assess(
            self, training_result: TrainingResult, predictions: PredictionResult | None
    ) -> AssessmentOutput | None:
        """
        Assess the trained model.

        Args:
            training_result: Result from the training phase, containing model and history.
            predictions: Predictions generated from the test dataset.

        Returns:
            AssessmentOutput from evaluation.

        Raises:
            RuntimeError: If model hasn't been trained.
        """
        if self.assessment:
            eval_results = self.assessment.evaluate(training_result, predictions)
            logger.info("Evaluation complete")
            return eval_results
        else:
            logger.debug("No assessment component provided, skipping evaluation")
            return None
