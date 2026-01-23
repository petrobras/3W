import shutil
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Any, Callable
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field

from ..core.base_models import BaseModels
from ..core.base_step import BaseStep
from ..core.base_assessment import ModelAssessmentConfig
from ..core.enums import DataSplit, TaskType
from ..metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    explained_variance_score,
)

from pylatex import Document


class MetricRegistry:
    """Registry that maps task types to supported metrics."""

    def __init__(self):
        """Initialize the metric registry."""
        self._registry = {
            TaskType.CLASSIFICATION: {
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
            TaskType.REGRESSION: {
                "explained_variance": explained_variance_score,
            },
        }

    def resolve(
        self, task_type: TaskType, metrics: list[str]
    ) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        """Resolve metric names to callable functions.

        Args:
            task_type (TaskType): Task type.
            metrics (list[str]): List of metric names.

        Returns:
            dict[str, Callable[[np.ndarray, np.ndarray], float]]: Mapping of metric names to functions.

        Raises:
            ValueError: If a metric is not supported for the task type.
        """
        available = self._registry.get(task_type, {})
        resolved = {}
        for m in metrics:
            if m not in available:
                raise ValueError(f"Metric '{m}' not available for task {task_type}")
            resolved[m] = available[m]
        return resolved


@dataclass
class FoldResults:
    """Stores evaluation results for a single fold in cross-validation.

    Attributes:
        fold_index (int): Index of the fold.
        predictions (np.ndarray): Model predictions for the fold.
        true_values (np.ndarray): Ground truth values for the fold.
        metrics (dict[str, float]): Calculated metrics for the fold.
        model_name (str): Name of the evaluated model.
        timestamp (str): ISO timestamp when the fold was evaluated.
        X_train (np.ndarray | None): Training features for the fold.
        y_train (np.ndarray | None): Training labels for the fold.
        X_val (np.ndarray | None): Validation features for the fold.
        y_val (np.ndarray | None): Validation labels for the fold.
    """

    fold_index: int
    predictions: np.ndarray
    true_values: np.ndarray
    metrics: dict[str, float]
    model_name: str
    timestamp: str

    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None


@dataclass
class AggregatedResults:
    """Stores aggregated cross-validation metrics.

    Attributes:
        metrics_mean (dict[str, float]): Mean value of each metric.
        metrics_std (dict[str, float]): Standard deviation of each metric.
        metrics_per_fold (list[dict[str, float]]): Metrics calculated per fold.
        n_folds (int): Number of folds.
    """

    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    metrics_per_fold: list[dict[str, float]]
    n_folds: int


@dataclass
class AssessmentInput:
    """
    Container for model assessment input data.

    This dataclass defines the standardized interface between the
    training and assessment pipeline steps.
    """

    models: list[BaseModels]
    x: pd.DataFrame | np.ndarray
    y: pd.DataFrame | np.ndarray

    dataset_split: DataSplit
    kwargs: dict[str, Any] = field(default_factory=dict)

    # Optional cross-validation data
    x_train_folds: list[np.ndarray] = field(default_factory=list)
    y_train_folds: list[np.ndarray] = field(default_factory=list)
    x_val_folds: list[np.ndarray] = field(default_factory=list)
    y_val_folds: list[np.ndarray] = field(default_factory=list)


@dataclass
class AssessmentOutput:
    """Container for model assessment results.

    Attributes:
        model_name (str): Name of the evaluated model.
        task_type (TaskType): Classification or regression task.
        dataset_split (DataSplit): Dataset split used for evaluation.
        timestamp (str): Evaluation timestamp.
        predictions (np.ndarray | None): Model predictions.
        true_values (np.ndarray | None): Ground truth values.
        metrics (dict[str, float] | None): Calculated metrics.
        fold_results (list[FoldResults]): Results per fold (CV only).
        aggregated_results (AggregatedResults | None): Aggregated CV results.
        config (dict[str, Any]): Serialized assessment configuration.
        is_cross_validation (bool): Whether CV was used.
        experiment_dir (str | None): Output directory for the experiment.
    """

    model_name: str
    task_type: TaskType
    dataset_split: DataSplit
    timestamp: str

    predictions: np.ndarray | None = None
    true_values: np.ndarray | None = None
    metrics: dict[str, float] | None = None

    fold_results: list[FoldResults] = field(default_factory=list)
    aggregated_results: AggregatedResults | None = None

    config: dict[str, Any] = field(default_factory=dict)
    is_cross_validation: bool = False

    experiment_dir: str | None = None


class AssessmentInputValidator:
    """Validates and normalizes input data for model assessment."""

    @staticmethod
    def validate(
        data: AssessmentInput,
        config: ModelAssessmentConfig,
    ) -> AssessmentInput:
        """Validate and normalize assessment input.

        This method ensures that the AssessmentInput object contains all
        required attributes and that their values are consistent with the
        assessment configuration.

        It also applies default values defined in the configuration when
        optional fields are missing.

        Args:
            data (AssessmentInput): Raw assessment input.
            config (ModelAssessmentConfig): Assessment configuration.

        Returns:
            AssessmentInput: Validated and normalized assessment input.

        Raises:
            ValueError: If required fields are missing or inconsistent.
            TypeError: If provided values have incompatible types.
        """
        # Required fields validation
        if not data.models:
            raise ValueError("At least one model must be provided")

        if data.x is None or data.y is None:
            raise ValueError("Both x and y must be provided")

        # Normalize models field
        if not isinstance(data.models, list):
            data.models = [data.models]

        # Dataset split default
        if data.dataset_split is None:
            data.dataset_split = config.dataset_split

        # Kwargs default
        if data.kwargs is None:
            data.kwargs = {}

        # Cross-validation consistency
        fold_fields = (
            data.x_train_folds,
            data.y_train_folds,
            data.x_val_folds,
            data.y_val_folds,
        )

        any_fold_provided = any(len(f) > 0 for f in fold_fields)

        if any_fold_provided:
            if not all(f is not None for f in fold_fields):
                raise ValueError(
                    "All fold inputs (x_train_folds, y_train_folds, "
                    "x_val_folds, y_val_folds) must be provided together"
                )

            n_models = len(data.models)

            if data.x_train_folds and len(data.x_train_folds) != n_models:
                raise ValueError("x_train_folds length must match number of models")

            if data.y_train_folds and len(data.y_train_folds) != n_models:
                raise ValueError("y_train_folds length must match number of models")

            if data.x_val_folds and len(data.x_val_folds) != n_models:
                raise ValueError("x_val_folds length must match number of models")

            if data.y_val_folds and len(data.y_val_folds) != n_models:
                raise ValueError("y_val_folds length must match number of models")

        return data


class MetricsAggregator:
    """Utility class for aggregating cross-validation metrics."""

    @staticmethod
    def aggregate(metrics_per_fold: list[dict[str, float]]) -> AggregatedResults:
        """Aggregate metrics across folds.

        Args:
            metrics_per_fold (list[dict[str, float]]): Metrics per fold.

        Returns:
            AggregatedResults: Mean and standard deviation of metrics.
        """
        metric_names = metrics_per_fold[0].keys()
        mean, std = {}, {}

        for name in metric_names:
            values = [m[name] for m in metrics_per_fold]
            mean[name] = float(np.mean(values))
            std[name] = float(np.std(values))

        return AggregatedResults(
            metrics_mean=mean,
            metrics_std=std,
            metrics_per_fold=metrics_per_fold,
            n_folds=len(metrics_per_fold),
        )


class ModelAssessment(BaseStep):
    """Pipeline step responsible for evaluating trained machine learning models.

    This class provides a unified and framework-agnostic interface for assessing
    model performance. It supports both single-model evaluation and
    cross-validation scenarios, handling:

    - Prediction generation
    - Metric computation
    - Result aggregation
    - CSV export
    - Optional report generation

    The assessment behavior is driven entirely by `ModelAssessmentConfig`,
    allowing flexible extension without modifying this class.

    Args:
        config (ModelAssessmentConfig): Configuration object defining metrics,
            task type, output options, and reporting behavior.

    Attributes:
        config (ModelAssessmentConfig): Assessment configuration.
        results (AssessmentResults | None): Results from the most recent evaluation.
        report_doc (Document | None): Generated report document, if enabled.
        metric_registry (MetricRegistry): Registry used to resolve metric functions.
        experiment_dir (Path | None): Directory where outputs are stored.
    """

    def __init__(self, config: ModelAssessmentConfig):
        """Initialize the model assessment step.

        This constructor initializes internal state, prepares metric resolution,
        and conditionally enables report generation using lazy imports to avoid
        circular dependencies.

        Args:
            config (ModelAssessmentConfig): Assessment configuration object.
        """
        self.config = config
        self.results: AssessmentOutput | None = None
        self.report_doc: Document | None = None

        # Lazily import report generation to avoid circular dependencies
        if self.config.generate_report:
            try:
                from ..reports.report_generation import ReportGeneration

                self._report_generation_class = ReportGeneration
            except ImportError:
                print(
                    "Warning: ReportGeneration class not available. Report generation disabled."
                )
                self.config.generate_report = False

        # Initialize metric registry
        self.metric_registry = MetricRegistry()

    def pre_process(self, data: AssessmentInput) -> AssessmentInput:
        """Validate and normalize assessment input data.

        This method ensures that the input object conforms to the expected
        schema and that all required attributes are present and consistent.
        It also applies default configuration values when necessary.

        Args:
            data (AssessmentInput): Raw assessment input object.

        Returns:
            AssessmentInput: Validated and normalized assessment input.

        Raises:
            TypeError: If the input is not an AssessmentInput instance.
            ValueError: If required fields are missing or inconsistent.
        """
        if not isinstance(data, AssessmentInput):
            raise TypeError(
                "Input to ModelAssessment must be an AssessmentInput instance"
            )

        validated_data = AssessmentInputValidator.validate(
            data=data,
            config=self.config,
        )

        return validated_data

    def run(self, data: AssessmentInput) -> AssessmentOutput:
        """Execute the model assessment pipeline step.

        This method orchestrates the execution of the assessment logic by
        invoking the evaluation routine and returning the structured
        assessment output.

        Args:
            data (AssessmentInput): Preprocessed assessment input.

        Returns:
            AssessmentOutput: Final assessment results.
        """
        return self.evaluate(data)

    def post_process(self, data: AssessmentOutput) -> AssessmentOutput:
        """Finalize and enrich the assessment output.

        This method appends additional metadata and human-readable
        summaries to the assessment output object without altering
        the core evaluation results.

        Args:
            data (AssessmentOutput): Assessment output generated by the
                evaluation stage.

        Returns:
            AssessmentOutput: Enriched assessment output.
        """

        return data

    def evaluate(self, data: AssessmentInput) -> AssessmentOutput:
        """Evaluate one or multiple models on the provided dataset.

        This method orchestrates the full evaluation lifecycle:
        prediction, metric computation, result aggregation, exporting,
        and optional report generation.

        Args:
            data (AssessmentInput): Dictionary containing:
                - models: list of trained models
                - x: input features
                - y: ground truth labels
                - dataset_split: dataset split identifier

        Returns:
            AssessmentOutput: Structured assessment results.

        Raises:
            RuntimeError: If evaluation fails and cleanup is required.
        """
        # Extract models
        models = data.models

        # Convert inputs to NumPy arrays
        X_array = self._to_numpy(data.x)
        y_array = self._to_numpy(data.y).flatten()

        # Resolve metric functions based on task type
        metric_fns = self.metric_registry.resolve(
            task_type=self.config.task_type, metrics=self.config.metrics
        )

        # Determine whether this is a cross-validation scenario
        is_cross_validation = len(models) > 1

        if is_cross_validation:
            results = self._evaluate_cv(
                models, X_array, y_array, metric_fns, data.dataset_split
            )
        else:
            results = self._evaluate_single(
                models[0], X_array, y_array, metric_fns, data.dataset_split
            )

        try:
            # Create output directory for results and reports
            self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            self.experiment_dir: Path = (
                self.config.output_dir / f"exp_{self.experiment_timestamp}"
            )
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            results.experiment_dir = str(self.experiment_dir)

            # Build structured output
            output = AssessmentOutput(
                model_name=models[0].model_name,
                task_type=self.config.task_type,
                dataset_split=data.dataset_split,
                timestamp=self.experiment_timestamp,
                is_cross_validation=is_cross_validation,
                config=self.config.model_dump(),
                experiment_dir=str(self.experiment_dir),
            )

            # Populate results
            output.predictions = results.predictions
            output.true_values = results.true_values
            output.metrics = results.metrics
            output.fold_results = results.fold_results
            output.aggregated_results = results.aggregated_results

            self.results = output

            # Generate LaTeX report if enabled
            if self.config.generate_report:
                self._generate_report(data, output)

            # Export results to CSV files if enabled
            if self.config.export_results:
                self._export_results()

        except Exception as exc:
            if getattr(self, "experiment_dir", None) and self.experiment_dir.exists():
                shutil.rmtree(self.experiment_dir)

            raise RuntimeError(
                "ModelAssessment failed. Experiment directory was fully removed."
            ) from exc

        # Print summary
        print(self.summary())

        return output

    def _evaluate_single(
        self,
        model: BaseModels,
        X: np.ndarray,
        y: np.ndarray,
        metric_fns: dict[str, Callable],
        dataset_split: DataSplit,
    ) -> AssessmentOutput:
        """Evaluate a single trained model.

        Args:
            model (BaseModels): Trained model instance.
            X (np.ndarray): Input features.
            y (np.ndarray): Ground truth labels.
            metric_fns (dict[str, Callable]): Metric functions.
            dataset_split (DataSplit): Dataset split identifier.

        Returns:
            AssessmentOutput: Evaluation results for a single model.
        """
        preds = self._get_predictions(model, X)
        metrics = {k: float(fn(y, preds)) for k, fn in metric_fns.items()}

        return AssessmentOutput(
            model_name=model.model_name,
            task_type=self.config.task_type,
            dataset_split=dataset_split,
            timestamp=pd.Timestamp.now().isoformat(),
            predictions=preds,
            true_values=y,
            metrics=metrics,
            is_cross_validation=False,
            config=self.config.model_dump(),
        )

    def _evaluate_cv(
        self,
        models: list[BaseModels],
        X: np.ndarray,
        y: np.ndarray,
        metric_fns: dict[str, Callable],
        dataset_split: DataSplit,
    ) -> AssessmentOutput:
        """Evaluate multiple models in a cross-validation setting.

        Args:
            models (list[BaseModels]): Models trained on different folds.
            X (np.ndarray): Evaluation features.
            y (np.ndarray): Ground truth labels.
            metric_fns (dict[str, Callable]): Metric functions.
            dataset_split (DataSplit): Dataset split identifier.

        Returns:
            AssessmentOutput: Aggregated cross-validation results.
        """
        fold_results = []
        metrics_per_fold = []

        for idx, model in enumerate(models):
            preds = self._get_predictions(model, X)
            metrics = {k: float(fn(y, preds)) for k, fn in metric_fns.items()}

            fold_results.append(
                FoldResults(
                    fold_index=idx,
                    predictions=preds,
                    true_values=y,
                    metrics=metrics,
                    model_name=model.model_name,
                    timestamp=pd.Timestamp.now().isoformat(),
                )
            )
            metrics_per_fold.append(metrics)

        aggregated = MetricsAggregator.aggregate(metrics_per_fold)

        return AssessmentOutput(
            model_name=models[0].model_name,
            task_type=self.config.task_type,
            dataset_split=dataset_split,
            timestamp=pd.Timestamp.now().isoformat(),
            fold_results=fold_results,
            aggregated_results=aggregated,
            is_cross_validation=True,
            config=self.config.model_dump(),
        )

    def _get_predictions(self, model: BaseModels, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the model's prediction strategy.

        Automatically selects between dataloader-based or array-based
        prediction strategies.

        Args:
            model (BaseModels): Trained model.
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Model predictions.

        Note:
            A dummy target tensor is created when using DataLoader-based
            strategies to satisfy PyTorch Dataset interface requirements.
        """
        strategy = model.get_prediction_strategy()()

        if strategy.requires_dataloader():
            # Create DataLoader when required by the strategy
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32), torch.zeros(len(X))
            )
            loader = DataLoader(dataset, batch_size=self.config.batch_size)

            return strategy.predict(
                model, self.config.task_type, loader=loader, device=self.config.device
            )

        return strategy.predict(model, self.config.task_type, X=X)

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert pandas or array-like data to NumPy format.

        Args:
            data (Any): Input data.

        Returns:
            np.ndarray: Converted NumPy array.
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.values

        return np.asarray(data)

    def summary(self) -> str:
        """Generate a human-readable summary of the assessment results.

        This method builds a formatted text summary containing metadata about
        the evaluation (model name, task type, dataset split, timestamp) and
        all computed metrics. It supports both single evaluation and
        cross-validation scenarios.

        For cross-validation, aggregated metrics (mean ± standard deviation)
        are displayed. For single evaluation, raw metric values are shown.

        Returns:
            str: A formatted summary string suitable for printing or logging.
                If no evaluation has been performed, an informative message
                is returned instead.

        Example:
            >>> assessor.evaluate(model, X_test, y_test)
            >>> print(assessor.summary())
            Model Assessment Summary
            ========================
            Model: RandomForestClassifier
            Task Type: TaskType.CLASSIFICATION
            Dataset Split: DatasetSplit.TEST
            Timestamp: 2024-01-15T10:30:45.123456

            Metrics:
            accuracy: 0.8750
            f1: 0.8542

        Notes:
            - Numeric metric values are formatted with 4 decimal places.
            - Cross-validation metrics are shown as mean ± std.
            - Returns a default message if no results are available.
        """
        if not self.results:
            return "No evaluation results available. Run evaluate() first."

        summary_lines = [
            "Model Assessment Summary",
            "========================",
            f"Model: {self.results.model_name}",
            f"Task Type: {self.results.task_type.value}",
            f"Dataset Split: {self.results.dataset_split}",
            f"Timestamp: {self.results.timestamp}",
            "",
            "Metrics:",
        ]

        # Cross-validation: show aggregated metrics (mean ± std)
        if self.results.is_cross_validation and self.results.aggregated_results:
            for m, v in self.results.aggregated_results.metrics_mean.items():
                std = self.results.aggregated_results.metrics_std[m]
                summary_lines.append(f"{m}: {v:.4f} ± {std:.4f}")
        else:
            # Single evaluation: show raw metric values
            if self.results.metrics:
                for m, v in self.results.metrics.items():
                    summary_lines.append(f"{m}: {v:.4f}")

        return "\n".join(summary_lines)

    def _export_results(self) -> None:
        """Export assessment results to disk.

        This method dispatches the export process according to the evaluation
        strategy. For single evaluation, predictions and metrics are exported.
        For cross-validation, fold-level and aggregated results are exported.

        Raises:
            RuntimeError: If no assessment results are available.
        """
        if self.results is None:
            raise RuntimeError("No assessment results available to export.")

        if self.results.is_cross_validation:
            self._export_cv_results()
        else:
            self._export_single_results()

        print(f"Results exported to {self.experiment_dir}")

    def _export_single_results(self) -> None:
        """Export results from a single evaluation run.

        This method saves:
            - A CSV file with true values and predictions.
            - A CSV file with computed metrics.

        Each exported file includes metadata such as model name, task type,
        dataset split, and timestamp.
        """
        if self.results is None:
            raise RuntimeError("No assessment results available to export.")

        r = self.results

        # Predictions
        predictions_df = pd.DataFrame(
            {
                "true_values": r.true_values,
                "predictions": r.predictions,
            }
        )
        predictions_df["model_name"] = r.model_name
        predictions_df["task_type"] = r.task_type.value
        predictions_df["dataset_split"] = r.dataset_split.value
        predictions_df["timestamp"] = r.timestamp

        predictions_df.to_csv(
            self.experiment_dir / f"predictions_{r.dataset_split.value}.csv",
            index=False,
        )

        # Metrics
        metrics_df = pd.DataFrame([r.metrics])
        metrics_df["model_name"] = r.model_name
        metrics_df["task_type"] = r.task_type.value
        metrics_df["dataset_split"] = r.dataset_split.value
        metrics_df["timestamp"] = r.timestamp

        metrics_df.to_csv(
            self.experiment_dir / f"metrics_{r.dataset_split.value}.csv",
            index=False,
        )

    def _export_cv_results(self) -> None:
        """Export results from a cross-validation evaluation.

        This method saves:
            - Predictions for each fold.
            - Metrics computed per fold.
            - Aggregated metrics (mean and standard deviation).
        """
        if self.results is None:
            raise RuntimeError("No assessment results available to export.")

        r = self.results

        # Predictions per fold
        fold_rows = []
        for fold in r.fold_results:
            for y_true, y_pred in zip(fold.true_values, fold.predictions):
                fold_rows.append(
                    {
                        "fold": fold.fold_index,
                        "true_value": y_true,
                        "prediction": y_pred,
                        "model_name": fold.model_name,
                        "timestamp": fold.timestamp,
                    }
                )

        pd.DataFrame(fold_rows).to_csv(
            self.experiment_dir / "cv_predictions.csv",
            index=False,
        )

        if r.aggregated_results:
            # Metrics per fold
            metrics_per_fold_df = pd.DataFrame(r.aggregated_results.metrics_per_fold)
            metrics_per_fold_df["fold"] = range(r.aggregated_results.n_folds)

            metrics_per_fold_df.to_csv(
                self.experiment_dir / "cv_metrics_per_fold.csv",
                index=False,
            )

            # Aggregated metrics
            aggregated_df = pd.DataFrame(
                {
                    "metric": r.aggregated_results.metrics_mean.keys(),
                    "mean": r.aggregated_results.metrics_mean.values(),
                    "std": r.aggregated_results.metrics_std.values(),
                }
            )

            aggregated_df.to_csv(
                self.experiment_dir / "cv_metrics_aggregated.csv",
                index=False,
            )

    def _generate_report(
        self, input_data: AssessmentInput, output_data: AssessmentOutput
    ) -> None:
        """Generate an assessment report if enabled in the configuration.

        This method acts as a dispatcher for report generation, handling
        both single evaluation and cross-validation scenarios.

        If report generation is disabled or the report generation backend
        is not available, the method exits gracefully.

        Args:
            input_data (AssessmentInput): Assessment input data.
            output_data (AssessmentOutput): Assessment output results.

        Raises:
            RuntimeError: If results are not available.
        """
        if not self.config.generate_report:
            return

        if self.results is None:
            raise RuntimeError("No results available to generate report.")

        if not hasattr(self, "_report_generation_class"):
            print("Warning: ReportGeneration not available.")
            return

        if self.results.is_cross_validation:
            # self._generate_cv_report(input_data, output_data) # TODO: Cross-validation report generation still requires adjustments
            pass
        else:
            self._generate_single_report(input_data, output_data)

    def _generate_single_report(
        self, input_data: AssessmentInput, output_data: AssessmentOutput
    ):
        """Generate a report for a single evaluation run.

        This method builds a report using training and test data, predictions,
        and computed metrics. The report is generated and automatically exported.

        Args:
            input_data (AssessmentInput): Assessment input data.
            output_data (AssessmentOutput): Assessment output results.
        """
        if self.results is None:
            raise RuntimeError("No assessment results available to report.")

        r = self.results

        title = self.config.report_title or f"Model Assessment Report - {r.model_name}"

        if r.metrics:
            report_generator = self._report_generation_class(
                model=input_data.models[0],
                X_train=input_data.x_train_folds[0],
                y_train=input_data.y_train_folds[0],
                X_test=input_data.x,
                y_test=input_data.y,
                predictions=r.predictions,
                calculated_metrics=r.metrics,
                plot_config=None,
                title=title,
                author=self.config.report_author,
                reports_dir=self.experiment_dir,
                export_report_after_generate=True,
            )

            self.report_doc = report_generator.generate_summary_report(format="html")

    def _generate_cv_report(
        self, input_data: AssessmentInput, output_data: AssessmentOutput
    ):
        """Generate a report for a cross-validation evaluation.

        This method aggregates cross-validation metrics (mean and standard
        deviation) and generates a consolidated report covering all folds.

        Args:
            input_data (AssessmentInput): Assessment input data.
            output_data (AssessmentOutput): Assessment output results.
        """
        if self.results is None:
            raise RuntimeError("No assessment results available to report.")

        r = self.results

        if r.aggregated_results:
            aggregated_metrics = {
                f"{k}_mean": v for k, v in r.aggregated_results.metrics_mean.items()
            } | {f"{k}_std": v for k, v in r.aggregated_results.metrics_std.items()}

            title = (
                self.config.report_title or f"Model Assessment Report - {r.model_name}"
            )

            fold_preds = [f_res.predictions for f_res in r.fold_results]

            report_generator = self._report_generation_class(
                model=input_data.models,
                X_train=input_data.x_train_folds,
                y_train=input_data.y_train_folds,
                X_test=input_data.x,
                y_test=input_data.y,
                predictions=fold_preds,
                calculated_metrics=aggregated_metrics,
                plot_config=None,
                title=title,
                author=self.config.report_author,
                reports_dir=self.experiment_dir,
                export_report_after_generate=True,
            )

            self.report_doc = report_generator.generate_summary_report(format="html")
