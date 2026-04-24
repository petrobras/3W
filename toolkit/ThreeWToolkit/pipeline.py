"""Pipeline for orchestrating the complete ML workflow."""

import logging
from typing import Literal

from pydantic import Field, PrivateAttr, field_validator, ValidationInfo

from .assessment import ModelAssessment, ModelAssessmentConfig
from .core.base_assessment import AssessmentOutput
from .core.base_dataset import BaseDataset, BaseDatasetConfig
from .core.base_pipeline import BasePipeline, BasePipelineConfig, PipelineResult
from .core.base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    CrossValidationResult,
    PredictionResult,
    TrainingResult,
)
from .core.base_transform import BaseTransform, BaseTransformConfig
from .utils import ModelRecorder
from .utils.data_splitter import KFoldSplitter

logger = logging.getLogger(__name__)


class PipelineConfig(BasePipelineConfig):
    """
    Configuration for the ML pipeline.

    This config consolidates all component configurations, allowing the
    Pipeline to instantiate all necessary components from their configs.
    """

    # Component configs
    task: Literal["training", "cross_validation"] = Field(
        default="training", description="Task to perform on run()"
    )
    train_dataset_config: BaseDatasetConfig = Field(
        ..., description="Configuration for training dataset."
    )
    test_dataset_config: BaseDatasetConfig | None = Field(
        default=None, description="Configuration for test dataset (optional)."
    )
    val_dataset_config: BaseDatasetConfig | None = Field(
        default=None, description="Configuration for validation dataset (optional)."
    )
    pre_transform_config: BaseTransformConfig | None = Field(
        default=None,
        description="Configuration for data preprocessing transformation (optional). Should be BaseTransformConfig.",
    )
    transform_config: BaseTransformConfig | None = Field(
        default=None,
        description="Configuration for data transformation (optional). Should be BaseTransformConfig.",
    )
    trainer_config: BaseTrainerConfig = Field(
        ..., description="Configuration for the model trainer."
    )
    num_folds: int | None = Field(
        default=None,
        gt=1,
        description="Number of folds for cross-validation. Required if task is 'cross_validation'.",
    )
    stratify_by: list[str] = Field(
        default_factory=list,
        description="List of column names to stratify by during cross-validation (optional).",
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

    _target: type = PrivateAttr(default_factory=lambda: Pipeline)
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("num_folds")
    @classmethod
    def validate_num_folds(
        cls, num_folds: int | None, info: ValidationInfo
    ) -> int | None:
        if info.data.get("task") == "cross_validation" and num_folds is None:
            raise ValueError(
                "num_folds must be provided when task is 'cross_validation'"
            )
        elif info.data.get("task") != "cross_validation" and num_folds is not None:
            raise ValueError(
                "num_folds should not be provided when task is not 'cross_validation'"
            )
        return num_folds


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

        self.pre_transform: BaseTransform | None = (
            config.pre_transform_config.build()
            if config.pre_transform_config is not None
            else None
        )

        self.assessment: ModelAssessment | None = (
            config.assessment_config.build()
            if config.assessment_config is not None
            else None
        )

        self.model_recorder: ModelRecorder = ModelRecorder()
        logger.info("Pipeline initialized | experiment=%s", config.experiment_name)

    def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.

        Returns:
            PipelineResult containing training and assessment outputs.
        """
        logger.info("Starting pipeline execution")

        self.pre_process_data()

        # train or cross-validate training
        training_result: TrainingResult | CrossValidationResult
        if self.config.task == "cross_validation":
            if self.config.num_folds is None:
                raise ValueError(
                    "num_folds must be specified for cross-validation task"
                )
            training_result = self.cross_validate(
                self.train_dataset,
                self.config.num_folds,
                self.config.stratify_by,
                random_state=self.trainer.config.seed,
            )
            logger.info(
                "Cross-validation completed with %d folds", self.config.num_folds
            )
        elif self.config.task == "training":
            training_result = self.train(self.train_dataset, self.val_dataset)
            logger.info("Training completed")
        else:
            raise ValueError(f"Unsupported task: {self.config.task}")

        # predict (will run only if test dataset is provided)
        predictions = self.predict(self.test_dataset)

        # assessment step if assessment component is provided
        assessment_result = self.assess(training_result, predictions)

        # model recorder
        if self.config.save_results:
            if self.transform is not None:
                logger.info("Saving fitted transform for future use...")
                transform_path = self.model_recorder.save_transform(
                    self.transform, self.config.experiment_name + "_transform.pkl"
                )
                logger.info("Transform saved to %s", transform_path)

            model_path = self.model_recorder.save_model(
                self.trainer.model, self.config.experiment_name
            )
            logger.info("Model saved to %s", model_path)

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
        """Apply transformation to dataset.
        If fit=True, the transform will be fitted on the provided dataset before transforming.

        Args:
            dataset: The dataset to prepare.
            fit: Whether to fit the transform on the dataset before transforming.
        Returns:
            The transformed dataset, or None if the input dataset is None.
        """
        if dataset is None:
            return None

        if self.transform is None:
            return dataset

        if fit:
            logger.info("Fitting transform on training data...")
            self.transform.fit(dataset)

        return self.transform.transform(dataset)

    def pre_process_data(
        self,
    ) -> None:
        """Apply preprocessing transformation to dataset.
        This is a separate step from the main transform, allowing for distinct preprocessing transformations.

        Args:
            train_dataset: The dataset to preprocess.
            test_dataset: The test dataset to preprocess.
            val_dataset: The validation dataset to preprocess.
        Returns:
            The preprocessed dataset, or None if the input dataset is None.
        """
        if self.pre_transform is None:
            logger.debug(
                "No preprocessing transform provided, skipping preprocessing step"
            )
            return

        logger.info("Fitting preprocessing transform on training data...")
        self.pre_transform.fit(self.train_dataset)

        logger.info("Applying preprocessing transform to training data...")
        self.train_dataset = self.pre_transform.transform(self.train_dataset)

        if self.val_dataset is not None:
            logger.info("Applying preprocessing transform to validation data...")
            self.val_dataset = self.pre_transform.transform(self.val_dataset)

        if self.test_dataset is not None:
            logger.info("Applying preprocessing transform to test data...")
            self.test_dataset = self.pre_transform.transform(self.test_dataset)

        return

    def cross_validate(
        self,
        train_dataset: BaseDataset,
        num_folds: int,
        stratify_by: list[str] | None = None,
        random_state: int | None = None,
    ) -> CrossValidationResult:
        """Perform k-fold cross-validation on the training dataset.
        Args:
            train_dataset: The dataset to perform cross-validation on.
            num_folds: The number of folds to use for cross-validation.
            stratify_by: Optional list of column names to stratify by during splitting.
            random_state: Optional random state for reproducibility of splits.
        Returns:
            CrossValidationResult containing results from all folds and metadata.
        """
        splitter = KFoldSplitter(
            num_splits=num_folds,
            stratify_by=stratify_by or [],
            random_state=random_state,
        )
        training_results = []
        for fold_idx, (train_subset, val_subset) in enumerate(
            splitter.split_data(train_dataset)
        ):
            logger.info("Starting fold %d/%d", fold_idx + 1, num_folds)
            result = self.train(train_subset, val_subset)
            training_results.append(result)
            logger.info("Completed fold %d/%d", fold_idx + 1, num_folds)
        logger.info("Cross-validation completed with %d folds", num_folds)
        return CrossValidationResult(
            fold_results=training_results,
            metadata={
                "num_folds": num_folds,
                "stratify_by": stratify_by or [],
                "trainer_type": self.trainer.__class__.__name__,
                "seed": self.trainer.config.seed,
                "split_random_state": random_state,
            },
        )

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

    def predict(self, dataset: BaseDataset | None) -> PredictionResult | None:
        prepared_data = self._prepare_data(dataset, fit=False)
        if prepared_data is None:
            logger.info("No data to predict on, skipping prediction")
            return None

        logger.info("Generating predictions...")
        predictions = self.trainer.predict(prepared_data)
        logger.info("Prediction complete")
        return predictions

    def assess(
        self,
        training_result: TrainingResult | CrossValidationResult,
        predictions: PredictionResult | None,
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
