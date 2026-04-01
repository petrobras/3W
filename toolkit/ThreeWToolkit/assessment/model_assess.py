"""ModelAssessment for evaluating trained models."""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datetime import datetime
from typing import Callable
from torch.utils.data import DataLoader, TensorDataset

from ..reports.report_generation import ReportGeneration
from ..core.base_models import BaseModels
from ..core.base_dataset import BaseDataset
from ..core.base_trainer import BaseTrainer, TrainingResult
from ..core.base_assessment import ModelAssessmentConfig, AssessmentOutput
from ..core.enums import TaskTypeEnum
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


class ModelAssessment:
    """Evaluates a trained model from a trainer instance."""

    def __init__(
        self,
        trainer: BaseTrainer,
        training_result: TrainingResult,
        config: ModelAssessmentConfig | None = None,
    ):
        """Initialize ModelAssessment with a trainer and its training result.

        Args:
            trainer: The trainer instance used for training.
            training_result: The result from trainer.train().
            config: Optional assessment configuration.
        """
        self.trainer = trainer
        self.training_result = training_result
        self.model = training_result.model
        self.training_history = training_result.history

        self.config = config or ModelAssessmentConfig(
            metrics=["accuracy"],
            task_type=TaskTypeEnum.CLASSIFICATION,
        )

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
            config=self.config.model_dump(),
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
