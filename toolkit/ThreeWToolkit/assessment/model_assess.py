"""ModelAssessment for evaluating trained models."""

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from pydantic import Field, field_validator, PrivateAttr

from ..constants import OUTPUT_DIR
from ..core.enums import TaskTypeEnum

from datetime import datetime
from typing import Callable

from ..reports.report_generation import ReportGeneration
from ..core.base_trainer import PredictionResult, TrainingResult, CrossValidationResult
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
    matthews_corrcoef,
)

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
                "matthews_corrcoef": matthews_corrcoef,
            },
            TaskTypeEnum.REGRESSION: {
                "explained_variance": explained_variance_score,
            },
        }

    def resolve(
        self, task_type: TaskTypeEnum, metrics: list[str]
    ) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        """Resolve metric names to callable functions.

        Args:
            task_type: Type of ML task (classification or regression).
            metrics: List of metric names to resolve.

        Returns:
            Dictionary mapping metric names to callable score functions.

        Raises:
            ValueError: If a requested metric is not available for the task type.
        """
        available = self._registry.get(task_type, {})
        resolved = {}
        for m in metrics:
            if m not in available:
                raise ValueError(f"Metric '{m}' not available for task {task_type}")
            resolved[m] = available[m]
        return resolved


class ModelAssessmentConfig(BaseAssessmentConfig):
    """Configuration for model assessment and evaluation."""

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
    report_title: str | None = Field(default=None, description="Title for the report.")
    report_author: str = Field(
        default="3W Toolkit Report", description="Author name for the report."
    )
    _target: type = PrivateAttr(default_factory=lambda: ModelAssessment)

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


class ModelAssessment(BaseAssessment):
    """Evaluates a trained model from a trainer instance."""

    def __init__(
        self,
        config: ModelAssessmentConfig,
    ):
        """Initialize ModelAssessment with a trainer and its training result.

        Args:
            trainer: The trainer instance used for training.
            training_result: The result from trainer.train().
            config: Optional assessment configuration.
        """
        self.config = config
        self.metric_registry = MetricRegistry()
        self._cv_summary: dict[str, float] | None = None

        self._report_generator = (
            ReportGeneration if self.config.generate_report else None
        )

    def _compute_cv_summary(
        self, training_results: CrossValidationResult
    ) -> dict[str, float]:
        """Compute final loss mean/std across folds."""
        train_final_losses = np.array(
            [fold.history.train_loss[-1] for fold in training_results.fold_results],
            dtype=float,
        )

        summary = {
            "num_folds": float(len(training_results.fold_results)),
            "train_loss_mean": float(np.mean(train_final_losses)),
            "train_loss_std": float(
                np.std(train_final_losses, ddof=1)
                if len(train_final_losses) > 1
                else 0.0
            ),
        }

        val_final_losses = [
            fold.history.val_loss[-1]
            for fold in training_results.fold_results
            if fold.history.val_loss is not None and len(fold.history.val_loss) > 0
        ]
        if val_final_losses:
            val_array = np.array(val_final_losses, dtype=float)
            summary["val_loss_mean"] = float(np.mean(val_array))
            summary["val_loss_std"] = float(
                np.std(val_array, ddof=1) if len(val_array) > 1 else 0.0
            )
        return summary

    def evaluate(
        self,
        training_results: TrainingResult | CrossValidationResult,
        predictions: PredictionResult | None,
    ) -> AssessmentOutput:
        """Evaluate training results and predictions to compute metrics and generate report.

        Args:
            training_results: The results from the training process, including model and history.
            predictions: The predictions made by the model on the test set.

        Returns:
            AssessmentOutput with predictions, metrics, and training history.
        """
        metric_fns = self.metric_registry.resolve(
            task_type=self.config.task_type, metrics=self.config.metrics
        )

        self._cv_summary = None
        if isinstance(training_results, CrossValidationResult):
            if len(training_results.fold_results) == 0:
                raise ValueError("CrossValidationResult has no fold results.")
            model = training_results.fold_results[-1].model
            training_history = None
            self._cv_summary = self._compute_cv_summary(training_results)
        else:
            model = training_results.model
            training_history = training_results.history

        metrics = None
        if predictions is not None and predictions.y_true is not None:
            metrics = {
                k: float(fn(predictions.y_true, predictions.y_pred))
                for k, fn in metric_fns.items()
            }
        elif not isinstance(training_results, CrossValidationResult):
            raise ValueError(
                "True labels (y_true) are required for metric calculation."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = self.config.output_dir / f"exp_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        self.results = AssessmentOutput(
            model=model,
            task_type=self.config.task_type,
            timestamp=timestamp,
            predictions=predictions.y_pred if predictions is not None else None,
            true_values=predictions.y_true if predictions is not None else None,
            metrics=metrics,
            training_history=training_history,
            experiment_dir=str(experiment_dir),
        )

        self.experiment_dir = experiment_dir

        if self.config.export_results:
            self._export_results()

        if self.config.generate_report:
            self._generate_report()

        logger.info(self.summary())
        return self.results

    def summary(self) -> str:
        """Generate summary of assessment results.

        Returns:
            Formatted string containing assessment summary including model info and metrics.
        """
        if not self.results:
            return "No evaluation results available."

        lines = [
            "Model Assessment Summary",
            "========================",
            f"Model: {type(self.results.model).__name__}",
            f"Task Type: {self.results.task_type.value}",
            f"Timestamp: {self.results.timestamp}",
            "",
            "Metrics:",
        ]

        if self.results.metrics:
            for m, v in self.results.metrics.items():
                lines.append(f"  {m}: {v:.4f}")

        if self._cv_summary is not None:
            lines.append("")
            lines.append("Cross-Validation Summary:")
            lines.append(f"  Folds: {int(self._cv_summary['num_folds'])}")
            lines.append(
                f"  Final train_loss: {self._cv_summary['train_loss_mean']:.4f} ± "
                f"{self._cv_summary['train_loss_std']:.4f}"
            )
            if (
                "val_loss_mean" in self._cv_summary
                and "val_loss_std" in self._cv_summary
            ):
                lines.append(
                    f"  Final val_loss: {self._cv_summary['val_loss_mean']:.4f} ± "
                    f"{self._cv_summary['val_loss_std']:.4f}"
                )

        if self.results.training_history:
            lines.append("")
            lines.append("Training History:")
            final_loss = self.results.training_history.train_loss[-1]
            lines.append(f"  Final train_loss: {final_loss:.4f}")
            if self.results.training_history.val_loss is not None:
                final_val = self.results.training_history.val_loss[-1]
                lines.append(f"  Final val_loss: {final_val:.4f}")

        return "\n".join(lines)

    def _export_results(self) -> None:
        """Export results to disk."""
        if self.results is None:
            raise RuntimeError("No results to export.")

        r = self.results

        if r.true_values is not None and r.predictions is not None:
            predictions_df = pd.DataFrame(
                {
                    "true_values": r.true_values,
                    "predictions": r.predictions,
                }
            )
            predictions_df["model_name"] = type(r.model).__name__
            predictions_df["task_type"] = r.task_type.value
            predictions_df["timestamp"] = r.timestamp
            predictions_df.to_csv(
                self.experiment_dir / f"predictions_{r.timestamp}.csv",
                index=False,
            )

        metrics_df = pd.DataFrame([r.metrics or {}])
        metrics_df["model_name"] = type(r.model).__name__
        metrics_df["task_type"] = r.task_type.value
        metrics_df["timestamp"] = r.timestamp
        metrics_df.to_csv(
            self.experiment_dir / f"metrics_{r.timestamp}.csv", index=False
        )

        if r.training_history:
            history_df = pd.DataFrame(r.training_history)
            history_df.to_csv(self.experiment_dir / "training_history.csv", index=False)

        logger.info(f"Results exported to {self.experiment_dir}")

    def _generate_report(self) -> None:
        """Generate assessment report using ReportGeneration."""
        if self.results is None:
            raise RuntimeError("No results to generate report.")

        if self._report_generator is None:
            logger.warning("ReportGeneration not available.")
            return

        title = (
            self.config.report_title
            or f"Model Assessment Report - {type(self.results.model).__name__} - {self.results.task_type.value}"
        )

        report_generator = self._report_generator(
            model=self.results.model,
            train_len=None,
            test_len=None,
            predictions=pd.Series(self.results.predictions),
            calculated_metrics=self.results.metrics or {},
            plot_config=None,
            title=title,
            author=self.config.report_author,
            reports_dir=self.experiment_dir,
            export_report_after_generate=True,
        )

        self.report_doc = report_generator.generate_summary_report(format="html")
        logger.info("Report generated at %s", self.experiment_dir)
