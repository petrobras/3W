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
from ..core.base_trainer import PredictionResult, TrainingResult
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

        if self.config.generate_report:
            self._report_generator = ReportGeneration

    def evaluate(
        self, training_results: TrainingResult, predictions: PredictionResult | None
    ) -> AssessmentOutput:
        """ Evaluate training results and predictions to compute metrics and generate report.

        Args:
            training_results: The results from the training process, including model and history.
            predictions: The predictions made by the model on the test set.

        Returns:
            AssessmentOutput with predictions, metrics, and training history.
        """
        metric_fns = self.metric_registry.resolve(
            task_type=self.config.task_type, metrics=self.config.metrics
        )
        if predictions is None:
            raise ValueError(
                "True labels (y_true) are required for metric calculation."
            )

        if predictions.y_true is None:
            raise ValueError(
                "True labels (y_true) are required for metric calculation."
            )
        metrics = {
            k: float(fn(predictions.y_true, predictions.y_pred))
            for k, fn in metric_fns.items()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = self.config.output_dir / f"exp_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        self.results = AssessmentOutput(
            model_name=training_results.model.model_name,
            task_type=self.config.task_type,
            timestamp=timestamp,
            predictions=predictions.y_pred,
            true_values=predictions.y_true,
            metrics=metrics,
            training_history=training_results.history,
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

    def _generate_report(self) -> None:
        """Generate assessment report using ReportGeneration."""
        if self.results is None:
            raise RuntimeError("No results to generate report.")

        if not hasattr(self, "_report_generation_class"):
            logger.warning("ReportGeneration not available.")
            return

        title = self.config.report_title or f"Model Assessment Report - {self.results.model_name}"

        report_generator = self._report_generator(
            model=self.results.model_name,
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
