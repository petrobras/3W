import torch
import numpy as np
import pandas as pd

from typing import Any, Callable, Union
from torch.utils.data import DataLoader, TensorDataset

from ..core.base_step import BaseStep
from ..core.base_assessment import ModelAssessmentConfig
from ..core.enums import TaskType
from ..metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    explained_variance_score,
)

from ..models.mlp import MLP
from ..models.sklearn_models import SklearnModels

from pylatex import Document


class ModelAssessment(BaseStep):
    """Comprehensive model evaluation class for both PyTorch and scikit-learn models.

    This class provides a unified interface for evaluating machine learning models
    across different frameworks. It handles metric calculation, result export,
    and report generation with support for both classification and regression tasks.

    The class automatically adapts its evaluation strategy based on the model type
    (PyTorch MLP vs scikit-learn) and provides flexible output options including
    CSV export and LaTeX report generation.

    Args:
        config (ModelAssessmentConfig): Configuration object containing evaluation
            parameters, output settings, and metric specifications.

    Attributes:
        config (ModelAssessmentConfig): The assessment configuration.
        results (dict[str, Any]): Dictionary storing the latest evaluation results.
        report_doc (Optional[Document]): Generated LaTeX report document.
        metric_functions (dict): Mapping of metric names to their calculation functions.

    Example:
        Basic usage for classification:
        >>> config = ModelAssessmentConfig(
        ...     metrics=["accuracy", "f1", "precision", "recall"],
        ...     task_type=TaskType.CLASSIFICATION,
        ...     export_results=True
        ... )
        >>> assessor = ModelAssessment(config)
        >>> results = assessor.evaluate(model, X_test, y_test)
        >>> print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        >>> print(assessor.summary())

        Usage with report generation:
        >>> config = ModelAssessmentConfig(
        ...     metrics=["explained_variance"],
        ...     task_type=TaskType.REGRESSION,
        ...     generate_report=True,
        ...     report_title="Model Performance Analysis"
        ... )
        >>> assessor = ModelAssessment(config)
        >>> results = assessor.evaluate(model, X_test, y_test)

    Note:
        - The class automatically creates output directories if they don't exist
        - Report generation requires the ReportGeneration class to be available
        - Metric calculations are robust with error handling and fallback values
    """

    def __init__(self, config: ModelAssessmentConfig):
        """Initialize the ModelAssessment with the given configuration.

        Sets up metric functions, creates output directories, and initializes
        the report generation system if enabled.

        Args:
            config (ModelAssessmentConfig): Configuration object containing
                all assessment parameters and settings.
        """
        self.config = config
        self.results: dict[str, Any] = {}
        self.report_doc: Document | None = None

        # Create output directory for results and reports
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Import ReportGeneration lazily to avoid circular imports
        if self.config.generate_report:
            try:
                from ..reports.report_generation import ReportGeneration

                self._report_generation_class = ReportGeneration
            except ImportError:
                print(
                    "Warning: ReportGeneration class not available. Report generation disabled."
                )
                self.config.generate_report = False

        self.metric_functions: dict[str, Callable] | None = None

    def pre_process(self, data: Any) -> dict[str, Any]:
        """Standardizes the input of the step.

        Validates and standardizes the input data format for model assessment.

        Args:
            data: Input data that should contain trained model and test data.
                  Can be a dict or any structure containing the required assessment data.

        Returns:
            dict[str, Any]: Standardized data dictionary with required keys.

        Raises:
            ValueError: If required assessment data is missing.
        """
        if isinstance(data, dict):
            processed_data = data.copy()
        else:
            # If data is not a dict, assume it's a tuple/list with (model, x_test, y_test, ...)
            if hasattr(data, "__iter__") and len(data) >= 3:
                processed_data = {
                    "model": data[0],
                    "x_test": data[1],
                    "y_test": data[2],
                }
                if len(data) >= 4:
                    processed_data["kwargs"] = data[3]
            else:
                raise ValueError(
                    "Input data must be a dict or iterable with at least (model, x_test, y_test)"
                )

        # Validate required keys
        required_keys = ["model", "x_test", "y_test"]
        missing_keys = [key for key in required_keys if key not in processed_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in input data: {missing_keys}")

        # Ensure optional keys exist with defaults
        if "kwargs" not in processed_data:
            processed_data["kwargs"] = {}

        # Validate that model exists and is not None
        if processed_data["model"] is None:
            raise ValueError("Model cannot be None for assessment")

        return processed_data

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Main logic of the step.

        Performs the actual model assessment using the provided data.

        Args:
            data (dict[str, Any]): Preprocessed data containing model and test data.

        Returns:
            dict[str, Any]: Data with assessment results added.
        """
        # Extract assessment parameters
        model = data["model"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        kwargs = data.get("kwargs", {})

        self._setup_metrics()
        # Perform evaluation
        assessment_results = self.evaluate(model, x_test, y_test, **kwargs)

        # Add assessment results to data
        data["assessment_results"] = assessment_results
        data["metrics"] = assessment_results["metrics"]
        data["predictions"] = assessment_results["predictions"]
        data["assessor"] = self

        return data

    def post_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Standardizes the output of the step.

        Performs any final processing and ensures output format consistency.

        Args:
            data (dict[str, Any]): Data with assessment results.

        Returns:
            dict[str, Any]: Final processed data ready for next pipeline step.
        """
        # Ensure all expected outputs are present
        expected_outputs = ["assessment_results", "metrics", "predictions", "assessor"]
        for key in expected_outputs:
            if key not in data:
                raise RuntimeError(
                    f"Assessment step failed to produce expected output: {key}"
                )

        # Add metadata about the assessment step
        data["assessment_completed"] = True
        data["task_type"] = self.config.task_type
        data["assessment_timestamp"] = pd.Timestamp.now().isoformat()

        # Add summary for easy access
        data["assessment_summary"] = self.summary()

        return data

    def _setup_metrics(self):
        """Configure metric functions based on the task type.

        Creates a mapping between metric names and their corresponding
        calculation functions, with appropriate parameters for each task type.

        For classification tasks, metrics use weighted averaging to handle
        class imbalance. For regression tasks, standard regression metrics
        are configured.

        Note:
            - Classification metrics use zero_division=0 to handle edge cases
            - Average precision requires at least 2 classes to be meaningful
            - All functions are wrapped with appropriate error handling
        """
        if self.config.task_type == TaskType.CLASSIFICATION:
            self.metric_functions = {
                "accuracy": accuracy_score,
                "balanced_accuracy": balanced_accuracy_score,
                "precision": lambda y_true, y_pred: precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "recall": lambda y_true, y_pred: recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "f1": lambda y_true, y_pred: f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                ),
                "average_precision": lambda y_true, y_pred: (
                    average_precision_score(y_true, y_pred, average="weighted")
                    if len(np.unique(y_true)) > 1
                    else 0.0
                ),
            }
        else:  # TaskType.REGRESSION
            self.metric_functions = {
                "explained_variance": explained_variance_score,
            }

    def evaluate(
        self,
        model: Union[MLP, SklearnModels, Any],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.DataFrame, pd.Series, np.ndarray],
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate model performance on test data with comprehensive metrics.

        This method performs a complete model evaluation including prediction
        generation, metric calculation, result storage, and optional report
        generation and export.

        Args:
            model (Union[MLP, SklearnModels, Any]): Trained model instance.
                Can be a PyTorch MLP, SklearnModels wrapper, or any sklearn-
                compatible model with a predict method.
            X_test (Union[pd.DataFrame, np.ndarray]): Test input features.
                Will be automatically converted to numpy array for processing.
            y_test (Union[pd.DataFrame, pd.Series, np.ndarray]): Test target
                values. Will be flattened and converted to numpy array.
            **kwargs: Additional keyword arguments passed to the model's
                predict method.

        Returns:
            dict[str, Any]: Comprehensive evaluation results containing:
                - model_name (str): Name of the evaluated model
                - task_type (TaskType): Classification or regression
                - predictions (np.ndarray): Model predictions on test data
                - true_values (np.ndarray): Actual target values
                - X_test (np.ndarray): Test features (for report generation)
                - metrics (dict[str, float]): Calculated metric values
                - config (dict): Assessment configuration as dictionary
                - timestamp (str): ISO format timestamp of evaluation

        Example:
            >>> results = assessor.evaluate(trained_model, X_test, y_test)
            >>> print(f"Model: {results['model_name']}")
            >>> print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            >>>
            >>> # Access predictions and true values
            >>> predictions = results['predictions']
            >>> true_values = results['true_values']

        Note:
            - Results are stored in self.results for later access
            - Report generation and result export occur automatically if enabled
            - Metric calculation is robust with error handling for edge cases
        """
        # Store model reference for report generation
        self._current_model = model

        # Convert inputs to consistent numpy array format
        X_test_array = self._to_numpy(X_test)
        y_test_array = self._to_numpy(y_test).flatten()

        # Generate predictions using appropriate method for model type
        predictions = self._get_predictions(model, X_test_array, **kwargs)

        # Calculate all requested metrics
        if self.metric_functions is None:
            self._setup_metrics()
        metrics_results = self._calculate_metrics(y_test_array, predictions)

        # Store comprehensive results
        self.results = {
            "model_name": self._get_model_name(model),
            "task_type": self.config.task_type,
            "predictions": predictions,
            "true_values": y_test_array,
            "X_test": X_test_array,  # Store for report generation
            "metrics": metrics_results,
            "config": self.config.model_dump(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Generate LaTeX report if enabled
        if self.config.generate_report:
            self._generate_report(X_test_array, y_test_array)

        # Export results to CSV files if enabled
        if self.config.export_results:
            self._export_results()

        print(self.summary())

        return self.results

    def _get_predictions(
        self, model: Union[MLP, SklearnModels, Any], X_test: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Generate predictions from model with proper handling for different types.

        Dispatches prediction generation to the appropriate method based on
        the model type, handling the different interfaces transparently.

        Args:
            model (Union[MLP, SklearnModels, Any]): Trained model instance.
            X_test (np.ndarray): Test features as numpy array.
            **kwargs: Additional arguments passed to the prediction method.

        Returns:
            np.ndarray: Model predictions as a numpy array.

        Note:
            - PyTorch MLP models require special DataLoader handling
            - SklearnModels wrapper and sklearn models use direct predict method
            - All predictions are returned in consistent numpy array format
        """
        if isinstance(model, MLP):
            return self._get_mlp_predictions(model, X_test, **kwargs)
        elif isinstance(model, SklearnModels):
            return model.predict(X_test, **kwargs)
        else:
            # Assume it's a sklearn model or has predict method
            return model.predict(X_test, **kwargs)

    def _get_mlp_predictions(
        self, model: MLP, X_test: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Generate predictions from PyTorch MLP model using DataLoader.

        Creates a DataLoader from test data and uses the model's predict method
        to generate predictions on the specified device.

        Args:
            model (MLP): PyTorch MLP model instance.
            X_test (np.ndarray): Test features as numpy array.
            **kwargs: Additional arguments passed to model.predict().

        Returns:
            np.ndarray: Model predictions as numpy array.

        Note:
            - Creates dummy labels for DataLoader compatibility
            - Uses configured batch_size and device from assessment config
            - Maintains gradient tracking disabled for prediction efficiency
        """
        # Create PyTorch tensor from numpy array
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        # Create dummy labels for DataLoader compatibility
        y_dummy = torch.zeros(X_tensor.shape[0])
        dataset = TensorDataset(X_tensor, y_dummy)

        # Create DataLoader with configured batch size
        test_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # Generate predictions using model's predict method
        return model.predict(test_loader, device=self.config.device, **kwargs)

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate evaluation metrics based on configuration.

        Computes all requested metrics using the configured metric functions,
        with robust error handling for edge cases and invalid configurations.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Model predictions.

        Returns:
            dict[str, float]: Dictionary mapping metric names to their values.
                Metrics that fail to compute are set to NaN with a warning.

        Note:
            - Handles edge cases like single-class predictions gracefully
            - Warns about unavailable metrics for the current task type
            - All metric values are converted to Python float for JSON compatibility
        """
        metrics_results = {}

        for metric_name in self.config.metrics:
            if self.metric_functions and metric_name in self.metric_functions.keys():
                try:
                    metric_value = self.metric_functions[metric_name](y_true, y_pred)
                    metrics_results[metric_name] = float(metric_value)
                except Exception as e:
                    print(f"Warning: Could not calculate {metric_name}: {e}")
                    metrics_results[metric_name] = np.nan
            else:
                print(
                    f"Warning: Metric '{metric_name}' not available for task type '{self.config.task_type}'"
                )
                metrics_results[metric_name] = np.nan

        return metrics_results

    def _get_model_name(self, model: Any) -> str:
        """Extract a human-readable name from the model object.

        Args:
            model (Any): Model instance to extract name from.

        Returns:
            str: Model class name or "Unknown_Model" if name cannot be determined.
        """
        if hasattr(model, "__class__"):
            return model.__class__.__name__
        return "Unknown_Model"

    def _to_numpy(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Convert various data types to numpy array format.

        Handles pandas DataFrames/Series and numpy arrays uniformly,
        ensuring consistent data format for all downstream processing.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.ndarray]): Input data
                in various supported formats.

        Returns:
            np.ndarray: Data converted to numpy array format.

        Note:
            - Preserves data structure and dtype when possible
            - Handles both pandas and numpy input gracefully
            - Falls back to np.array() for other array-like objects
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def _export_results(self):
        """Export evaluation results to CSV files.

        Creates two CSV files in the output directory:
        1. predictions.csv: Contains predictions, true values, and metrics
        2. metrics_summary.csv: Contains aggregated metrics and metadata

        The export includes model metadata and timestamps for result tracking.

        Note:
            - Creates output directory if it doesn't exist
            - Handles export errors gracefully with warning messages
            - Files are timestamped and include model identification
        """
        try:
            # Create DataFrame with predictions and true values
            predictions_df = pd.DataFrame(
                {
                    "true_values": self.results["true_values"],
                    "predictions": self.results["predictions"],
                }
            )

            # Add metrics as additional columns for easy analysis
            for metric_name, metric_value in self.results["metrics"].items():
                predictions_df[f"metric_{metric_name}"] = metric_value

            # Add metadata columns
            predictions_df["model_name"] = self.results["model_name"]
            predictions_df["task_type"] = self.results["task_type"].value

            # Save predictions with metadata
            predictions_path = self.config.output_dir / "predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)

            # Create metrics summary DataFrame
            metrics_df = pd.DataFrame([self.results["metrics"]])
            metrics_df["model_name"] = self.results["model_name"]
            metrics_df["task_type"] = self.results["task_type"].value
            metrics_df["timestamp"] = self.results["timestamp"]

            # Save metrics summary
            metrics_path = self.config.output_dir / "metrics_summary.csv"
            metrics_df.to_csv(metrics_path, index=False)

            print(f"Results exported to {self.config.output_dir}")

        except Exception as e:
            print(f"Warning: Results export failed: {e}")

    def _generate_report(self, X_test: np.ndarray, y_test: np.ndarray):
        """Generate LaTeX report using the ReportGeneration class.

        Creates a comprehensive report including model performance metrics,
        visualizations, and analysis. Integrates with the legacy ReportGeneration
        system while adapting to its expected data format.

        Args:
            X_test (np.ndarray): Test features for report generation.
            y_test (np.ndarray): Test targets for report generation.

        Note:
            - Requires ReportGeneration class to be available
            - Converts data to pandas Series format for legacy compatibility
            - Uses pre-calculated metrics and predictions for efficiency
            - Handles missing ReportGeneration gracefully with warnings
        """
        if not hasattr(self, "_report_generation_class"):
            print("Warning: ReportGeneration not available")
            return

        # Convert numpy arrays to pandas Series for legacy compatibility
        X_test_series = pd.Series(X_test.flatten() if len(X_test.shape) > 1 else X_test)
        y_test_series = pd.Series(y_test)

        # Create empty training data (required by legacy interface)
        X_train_series = pd.Series([])
        y_train_series = pd.Series([])

        # Determine report title from configuration or use default
        report_title = (
            self.config.report_title
            or f"Model Assessment Report - {self.results['model_name']}"
        )

        # Map metrics to legacy format
        report_metrics = self._map_metrics_for_report()

        # Create ReportGeneration instance with legacy constructor
        report_generator = self._report_generation_class(
            model=self._current_model,
            X_train=X_train_series,
            y_train=y_train_series,
            X_test=X_test_series,
            y_test=y_test_series,
            metrics=report_metrics,
            title=report_title,
            author=self.config.report_author,
            reports_dir=self.config.output_dir,
            export_report_after_generate=True,
            # Pass pre-calculated values to avoid recomputation
            predictions=pd.Series(self.results["predictions"]),
            calculated_metrics=self.results["metrics"],
        )

        # Generate the comprehensive report
        self.report_doc = report_generator.generate_summary_report()

    def _map_metrics_for_report(self) -> list[str]:
        """Map assessment metrics to ReportGeneration format.

        Adapts the current metric configuration to the format expected
        by the legacy ReportGeneration system.

        Returns:
            list[str]: List of metric names in ReportGeneration format.

        Note:
            - Currently returns metrics as-is, but can be extended for
              more complex mapping if needed
            - Provides abstraction layer for future ReportGeneration updates
        """
        return self.config.metrics

    def get_metric(self, metric_name: str) -> float:
        """Retrieve a specific metric value from the last evaluation.

        Provides convenient access to individual metric values without
        accessing the full results dictionary.

        Args:
            metric_name (str): Name of the metric to retrieve.

        Returns:
            float: The metric value.

        Raises:
            ValueError: If no evaluation has been run or if the specified
                metric was not calculated.

        Example:
            >>> assessor.evaluate(model, X_test, y_test)
            >>> accuracy = assessor.get_metric("accuracy")
            >>> f1_score = assessor.get_metric("f1")
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluate() first.")

        if metric_name not in self.results["metrics"]:
            available_metrics = list(self.results["metrics"].keys())
            raise ValueError(
                f"Metric '{metric_name}' not found. Available metrics: {available_metrics}"
            )

        return self.results["metrics"][metric_name]

    def summary(self) -> str:
        """Generate a formatted text summary of evaluation results.

        Creates a human-readable summary including model information,
        evaluation metadata, and all calculated metrics with appropriate
        formatting.

        Returns:
            str: Formatted summary string suitable for printing or logging.

        Example:
            >>> assessor.evaluate(model, X_test, y_test)
            >>> print(assessor.summary())
            Model Assessment Summary
            ========================
            Model: RandomForestClassifier
            Task Type: TaskType.CLASSIFICATION
            Timestamp: 2024-01-15T10:30:45.123456

            Metrics:
              accuracy: 0.8750
              f1: 0.8542
              precision: 0.8634
              recall: 0.8750

        Note:
            - NaN values are displayed as "N/A"
            - Numeric values are formatted to 4 decimal places
            - Returns informative message if no evaluation has been performed
        """
        if not self.results:
            return "No evaluation results available. Run evaluate() first."

        summary_lines = [
            "Model Assessment Summary",
            "========================",
            f"Model: {self.results['model_name']}",
            f"Task Type: {self.results['task_type']}",
            f"Timestamp: {self.results['timestamp']}",
            "",
            "Metrics:",
        ]

        # Add formatted metric values
        for metric_name, metric_value in self.results["metrics"].items():
            if not np.isnan(metric_value):
                summary_lines.append(f"  {metric_name}: {metric_value:.4f}")
            else:
                summary_lines.append(f"  {metric_name}: N/A")

        return "\n".join(summary_lines)
