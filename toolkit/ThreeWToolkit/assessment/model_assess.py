"""ModelAssessment for evaluating trained models."""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from pydantic import Field, field_validator, PrivateAttr

from ..constants import OUTPUT_DIR
from ..core.enums import TaskTypeEnum

from datetime import datetime
from typing import Callable
from torch.utils.data import DataLoader, TensorDataset

from ..reports.report_generation import ReportGeneration
from ..core.base_models import BaseModels
from ..core.base_dataset import BaseDataset
from ..core.base_trainer import TrainingResult
from ..core.base_assessment import AssessmentOutput
from ..core import BaseAssessmentConfig, BaseAssessment

from ..metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    explained_variance_score,
)
from .strategies.torch_prediction_strategy import TorchPredictionStrategy
from .strategies.sklearn_prediction_strategy import SklearnPredictionStrategy

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Maps task types to supported metrics."""

    def __init__(self):
        self._registry = {
            TaskTypeEnum.CLASSIFICATION: {
                "accuracy": accuracy_score,
                "balanced_accuracy": balanced_accuracy_score,
                "precision": lambda y, p: precision_score(
                    y, p, average="weighted", zero_division=0
                ),
                "recall": lambda y, p: recall_score(
                    y, p, average="weighted", zero_division=0
                ),
                "f1": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
                "average_precision": lambda y, p: (
                    average_precision_score(y, p, average="weighted")
                    if len(np.unique(y)) > 1
                    else 0.0
                ),
            },
            TaskTypeEnum.REGRESSION: {
                "explained_variance": explained_variance_score,
            },
        }

    def resolve(
        self, task_type: TaskTypeEnum, metrics: list[str]
    ) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        """Resolve metric names to callable functions."""
        available = self._registry.get(task_type, {})
        resolved = {}
        for m in metrics:
            if m not in available:
                raise ValueError(f"Metric '{m}' not available for task {task_type}")
            resolved[m] = available[m]
        return resolved


class ModelAssessmentConfig(BaseAssessmentConfig):
    """
    Configuration for model assessment and evaluation.

    Args:
        metrics (list[str]): List of metric names to calculate.
        output_dir (Path): Directory to save assessment results.
        export_results (bool): Whether to export results to CSV files.
        generate_report (bool): Whether to generate LaTeX report using ReportGeneration.
        task_type (TaskTypeEnum): Type of task (TaskTypeEnum.CLASSIFICATION or TaskTypeEnum.REGRESSION).
        batch_size (int): Batch size for PyTorch model predictions.
        device (str): Device for PyTorch computations.
        report_title (str | None): Title for the report.
        report_author (str): Author name for the report.
    """

    metrics: list[str] = Field(
        default=["accuracy", "f1"], description="Metrics to compute."
    )
    output_dir: Path = Field(
        default=Path(OUTPUT_DIR), description="Directory for output files."
    )
    export_results: bool = Field(
        default=True, description="Whether to export results to files."
    )
    generate_report: bool = Field(
        default=False, description="Whether to generate a report."
    )
    task_type: TaskTypeEnum = Field(
        default=TaskTypeEnum.CLASSIFICATION, description="Type of ML task."
    )
    batch_size: int = Field(
        default=64, gt=0, description="Batch size for PyTorch model predictions."
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for PyTorch computations.",
    )
    report_title: str | None = Field(default=None, description="Title for the report.")
    report_author: str = Field(
        default="3W Toolkit Report", description="Author name for the report."
    )
    _target: type = PrivateAttr(default_factory=lambda: ModelAssessment)

    @field_validator("task_type")
    @classmethod
    def validate_task_type(
        cls: type["ModelAssessmentConfig"], task_type: TaskTypeEnum
    ) -> TaskTypeEnum:
        """
        Validate that the task type is supported.

        Args:
            cls (ModelAssessmentConfig): The class reference.
            task_type (TaskTypeEnum): Task type to validate.

        Returns:
            TaskTypeEnum: Validated task type.

        Raises:
            ValueError: If task_type is not supported.
        """
        valid_types = {TaskTypeEnum.CLASSIFICATION, TaskTypeEnum.REGRESSION}
        if task_type not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}")
        return task_type

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls: type["ModelAssessmentConfig"], metrics: list[str]
    ) -> list[str]:
        """
        Validate that the requested metrics are supported.

        Args:
            cls (ModelAssessmentConfig): The class reference.
            metrics (list[str]): List of metric names.

        Returns:
            list[str]: Validated list of metrics.

        Raises:
            ValueError: If any metric is not supported.
        """
        valid_metrics = {
            # Classification metrics
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            # Regression metrics
            "explained_variance",
        }
        invalid_metrics = set(metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}"
            )
        return metrics

    @field_validator("device")
    @classmethod
    def validate_device(cls: type["ModelAssessmentConfig"], device: str) -> str:
        """
        Validate that the computation device is supported.

        Args:
            cls (ModelAssessmentConfig): The class reference.
            device (str): Device name ('cpu' or 'cuda').

        Returns:
            str: Validated device name.

        Raises:
            ValueError: If device is not supported.
        """
        valid_devices = {"cpu", "cuda"}
        if device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        return device


class ModelAssessment(BaseAssessment):
    """Evaluates a trained model from a trainer instance."""

    def __init__(
        self,
        training_result: TrainingResult,
        config: ModelAssessmentConfig,
    ):
        """Initialize ModelAssessment with a trainer and its training result.

        Args:
            trainer: The trainer instance used for training.
            training_result: The result from trainer.train().
            config: Optional assessment configuration.
        """
        self.model = training_result.model
        self.training_history = training_result.history
        self.config = config
        self.results: AssessmentOutput | None = None
        self.metric_registry = MetricRegistry()
        self._report_generator = None

        if self.config.generate_report:
            self._report_generation_class = ReportGeneration

    def evaluate(self, test_dataset: BaseDataset) -> AssessmentOutput:
        """Evaluate the model on a test dataset.

        Args:
            test_dataset: The dataset to evaluate on.

        Returns:
            AssessmentOutput with predictions, metrics, and training history.
        """
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty")

        x_array, y_array = self._prepare_data(test_dataset)

        metric_fns = self.metric_registry.resolve(
            task_type=self.config.task_type, metrics=self.config.metrics
        )

        preds = self._get_predictions(self.model, x_array)
        metrics = {k: float(fn(y_array, preds)) for k, fn in metric_fns.items()}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = self.config.output_dir / f"exp_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        self.results = AssessmentOutput(
            model_name=self.model.model_name,
            task_type=self.config.task_type,
            timestamp=timestamp,
            predictions=preds,
            true_values=y_array,
            metrics=metrics,
            training_history=self.training_history,
            experiment_dir=str(experiment_dir),
        )

        self.experiment_dir = experiment_dir

        if self.config.export_results:
            self._export_results()

        if self.config.generate_report:
            self._generate_report(x_array, y_array)

        logger.info(self.summary())
        return self.results

    def _prepare_data(self, dataset: BaseDataset) -> tuple[np.ndarray, np.ndarray]:
        """Convert BaseDataset to numpy arrays (X, y)."""
        logger.info("Preparing dataset for assessment (size=%d)", len(dataset))

        signals_list = []
        labels_list = []

        for idx in range(len(dataset)):
            event = dataset[idx]
            signal = event.signal.values

            # Each row in signal is a sample
            if signal.ndim == 2:
                signals_list.append(signal)
            else:
                signals_list.append(signal.reshape(1, -1))

            if event.label is not None:
                label = event.label.values
                if hasattr(label, "__len__"):
                    labels_list.extend(label)
                else:
                    labels_list.append(label)

        x_array = np.concatenate(signals_list, axis=0)
        y_array = np.array(labels_list)

        logger.info(
            "Prepared arrays | X.shape=%s | y.shape=%s", x_array.shape, y_array.shape
        )
        return x_array, y_array

    def _get_predictions(self, model: BaseModels, x: np.ndarray) -> np.ndarray:
        """Generate predictions by detecting model type."""
        if isinstance(model, nn.Module):
            strategy: TorchPredictionStrategy | SklearnPredictionStrategy = (
                TorchPredictionStrategy()
            )
            dataset = TensorDataset(
                torch.tensor(x, dtype=torch.float32), torch.zeros(len(x))
            )
            loader = DataLoader(dataset, batch_size=self.config.batch_size)
            return strategy.predict(
                model, self.config.task_type, loader=loader, device=self.config.device
            )
        else:
            strategy = SklearnPredictionStrategy()
            return strategy.predict(model, self.config.task_type, x=x)

    def summary(self) -> str:
        """Generate summary of assessment results."""
        if not self.results:
            return "No evaluation results available."

        lines = [
            "Model Assessment Summary",
            "========================",
            f"Model: {self.results.model_name}",
            f"Task Type: {self.results.task_type.value}",
            f"Timestamp: {self.results.timestamp}",
            "",
            "Metrics:",
        ]

        if self.results.metrics:
            for m, v in self.results.metrics.items():
                lines.append(f"  {m}: {v:.4f}")

        if self.training_history:
            lines.append("")
            lines.append("Training History:")
            if "train_loss" in self.training_history:
                final_loss = self.training_history["train_loss"][-1]
                lines.append(f"  Final train_loss: {final_loss:.4f}")
            if (
                "val_loss" in self.training_history
                and self.training_history["val_loss"]
            ):
                final_val = self.training_history["val_loss"][-1]
                lines.append(f"  Final val_loss: {final_val:.4f}")

        return "\n".join(lines)

    def _export_results(self) -> None:
        """Export results to disk."""
        if self.results is None:
            raise RuntimeError("No results to export.")

        r = self.results

        predictions_df = pd.DataFrame(
            {
                "true_values": r.true_values,
                "predictions": r.predictions,
            }
        )
        predictions_df["model_name"] = r.model_name
        predictions_df["task_type"] = r.task_type.value
        predictions_df["timestamp"] = r.timestamp
        predictions_df.to_csv(
            self.experiment_dir / f"predictions_{r.timestamp}.csv",
            index=False,
        )

        metrics_df = pd.DataFrame([r.metrics])
        metrics_df["model_name"] = r.model_name
        metrics_df["task_type"] = r.task_type.value
        metrics_df["timestamp"] = r.timestamp
        metrics_df.to_csv(
            self.experiment_dir / f"metrics_{r.timestamp}.csv", index=False
        )

        if r.training_history:
            history_df = pd.DataFrame(r.training_history)
            history_df.to_csv(self.experiment_dir / "training_history.csv", index=False)

        logger.info(f"Results exported to {self.experiment_dir}")

    def _generate_report(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """Generate assessment report using ReportGeneration."""
        if not self.config.generate_report:
            return

        if self.results is None:
            raise RuntimeError("No results to generate report.")

        if not hasattr(self, "_report_generation_class"):
            logger.warning("ReportGeneration not available.")
            return

        r = self.results
        title = self.config.report_title or f"Model Assessment Report - {r.model_name}"

        report_generator = self._report_generation_class(
            model=self.model,
            X_train=None,
            y_train=None,
            X_test=x_test,
            y_test=y_test,
            predictions=r.predictions,
            calculated_metrics=r.metrics or {},
            plot_config=None,
            title=title,
            author=self.config.report_author,
            reports_dir=self.experiment_dir,
            export_report_after_generate=True,
        )

        self.report_doc = report_generator.generate_summary_report(format="html")
        logger.info("Report generated at %s", self.experiment_dir)
