from pydantic import ValidationError
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil

from pathlib import Path
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader

from ThreeWToolkit.assessment.model_assess import ModelAssessment
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.enums import TaskType
from ThreeWToolkit.models.mlp import MLP
from ThreeWToolkit.models.sklearn_models import SklearnModels


class TestModelAssessmentConfig:
    """Test suite for ModelAssessmentConfig validation and initialization."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = ModelAssessmentConfig()

        assert config.metrics == ["accuracy", "f1"]
        assert config.export_results is True
        assert config.generate_report is False
        assert config.task_type == TaskType.CLASSIFICATION
        assert config.batch_size == 64
        assert config.device in ["cpu", "cuda"]
        assert config.report_author == "3W Toolkit Report"

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = ModelAssessmentConfig(
            metrics=["precision", "recall"],
            output_dir=Path("./custom_output"),
            export_results=False,
            generate_report=True,
            task_type=TaskType.REGRESSION,
            batch_size=32,
            device="cpu",
            report_title="Custom Report",
            report_author="Test Author",
        )

        assert config.metrics == ["precision", "recall"]
        assert config.output_dir == Path("./custom_output")
        assert config.export_results is False
        assert config.generate_report is True
        assert config.task_type == TaskType.REGRESSION
        assert config.batch_size == 32
        assert config.device == "cpu"
        assert config.report_title == "Custom Report"
        assert config.report_author == "Test Author"

    def test_invalid_task_type(self):
        """Test validation fails for invalid task type."""
        with pytest.raises(ValidationError, match="task_type"):
            ModelAssessmentConfig(task_type="invalid_type")

    def test_invalid_metrics(self):
        """Test validation fails for invalid metrics."""
        with pytest.raises(ValidationError, match="Invalid metrics"):
            ModelAssessmentConfig(metrics=["invalid_metric", "accuracy"])

    def test_invalid_device(self):
        """Test validation fails for invalid device."""
        with pytest.raises(ValidationError, match="device must be one of"):
            ModelAssessmentConfig(device="tpu")

    def test_invalid_batch_size(self):
        """Test validation fails for non-positive batch size."""
        with pytest.raises(ValidationError):
            ModelAssessmentConfig(batch_size=0)

        with pytest.raises(ValidationError):
            ModelAssessmentConfig(batch_size=-1)


class TestModelAssessmentInitialization:
    """Test suite for ModelAssessment initialization."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_basic_initialization(self, temp_dir):
        """Test basic initialization without report generation."""
        config = ModelAssessmentConfig(output_dir=temp_dir, generate_report=False)
        assessor = ModelAssessment(config)

        assert assessor.config == config
        assert assessor.results == {}
        assert assessor.report_doc is None
        assert temp_dir.exists()

    def test_initialization_with_report_generation_success(self, temp_dir):
        """Test initialization with successful report generation import."""
        config = ModelAssessmentConfig(output_dir=temp_dir, generate_report=True)

        with patch(
            "ThreeWToolkit.reports.report_generation.ReportGeneration"
        ) as mock_report:
            assessor = ModelAssessment(config)
            assert hasattr(assessor, "_report_generation_class")
            assert assessor._report_generation_class == mock_report

    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = temp_dir / "nested" / "output"
        config = ModelAssessmentConfig(output_dir=nested_dir)

        _ = ModelAssessment(config)
        assert nested_dir.exists()


class TestModelAssessmentPreProcess:
    """Test suite for pre_process method."""

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir)
        return ModelAssessment(config)

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_preprocess_with_dict_input(self, assessor):
        """Test preprocessing with dictionary input."""
        model = Mock()
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])

        data = {
            "model": model,
            "x_test": X_test,
            "y_test": y_test,
            "kwargs": {"verbose": True},
        }

        result = assessor.pre_process(data)

        assert result["model"] == model
        assert np.array_equal(result["x_test"], X_test)
        assert np.array_equal(result["y_test"], y_test)
        assert result["kwargs"] == {"verbose": True}

    def test_preprocess_with_tuple_input(self, assessor):
        """Test preprocessing with tuple/list input."""
        model = Mock()
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])
        kwargs = {"verbose": True}

        data = (model, X_test, y_test, kwargs)
        result = assessor.pre_process(data)

        assert result["model"] == model
        assert np.array_equal(result["x_test"], X_test)
        assert np.array_equal(result["y_test"], y_test)
        assert result["kwargs"] == kwargs

    def test_preprocess_with_tuple_without_kwargs(self, assessor):
        """Test preprocessing with tuple input without kwargs."""
        model = Mock()
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])

        data = (model, X_test, y_test)
        result = assessor.pre_process(data)

        assert result["model"] == model
        assert result["kwargs"] == {}

    def test_preprocess_missing_required_keys(self, assessor):
        """Test preprocessing fails with missing required keys."""
        with pytest.raises(ValueError, match="Missing required keys"):
            assessor.pre_process({"model": Mock()})

    def test_preprocess_with_none_model(self, assessor):
        """Test preprocessing fails with None model."""
        data = {"model": None, "x_test": np.array([[1, 2]]), "y_test": np.array([0])}

        with pytest.raises(ValueError, match="Model cannot be None"):
            assessor.pre_process(data)

    def test_preprocess_with_invalid_input_type(self, assessor):
        """Test preprocessing fails with invalid input type."""
        with pytest.raises(ValueError, match="must be a dict or iterable"):
            assessor.pre_process(10)

    def test_preprocess_dict_without_kwargs(self, assessor):
        """Test preprocessing adds empty kwargs if missing in dict."""
        model = Mock()
        data = {"model": model, "x_test": np.array([[1, 2]]), "y_test": np.array([0])}

        result = assessor.pre_process(data)
        assert result["kwargs"] == {}


class TestModelAssessmentRun:
    """Test suite for run method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy"],
            export_results=False,
            generate_report=False,
        )
        return ModelAssessment(config)

    def test_run_successful_evaluation(self, assessor):
        """Test run method executes evaluation successfully."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0, 1]))

        data = {
            "model": model,
            "x_test": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "y_test": np.array([0, 1, 0, 1]),
            "kwargs": {},
        }

        result = assessor.run(data)

        assert "assessment_results" in result
        assert "metrics" in result
        assert "predictions" in result
        assert "assessor" in result
        assert result["assessor"] == assessor

    def test_run_with_custom_kwargs(self, assessor):
        """Test run method passes kwargs to evaluate."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1]))

        data = {
            "model": model,
            "x_test": np.array([[1, 2], [3, 4]]),
            "y_test": np.array([0, 1]),
            "kwargs": {"custom_param": "value"},
        }

        result = assessor.run(data)
        assert "assessment_results" in result


class TestModelAssessmentPostProcess:
    """Test suite for post_process method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(
            output_dir=temp_dir, task_type=TaskType.CLASSIFICATION
        )
        assessor = ModelAssessment(config)
        assessor.results = {
            "metrics": {"accuracy": 0.85},
            "predictions": np.array([0, 1]),
            "model_name": "TestModel",
        }
        return assessor

    def test_postprocess_missing_expected_output(self, assessor):
        """Test post_process fails when expected outputs are missing."""
        data = {"assessment_results": {}}

        with pytest.raises(RuntimeError, match="failed to produce expected output"):
            assessor.post_process(data)


class TestModelAssessmentMetricSetup:
    """Test suite for _setup_metrics method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_setup_classification_metrics(self, temp_dir):
        """Test metric setup for classification tasks."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir, task_type=TaskType.CLASSIFICATION
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        expected_metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
        ]
        for metric in expected_metrics:
            assert metric in assessor.metric_functions

    def test_setup_regression_metrics(self, temp_dir):
        """Test metric setup for regression tasks."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            task_type=TaskType.REGRESSION,
            metrics=["explained_variance"],
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        assert "explained_variance" in assessor.metric_functions

    def test_classification_metrics_callable(self, temp_dir):
        """Test that classification metrics are callable."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir, task_type=TaskType.CLASSIFICATION
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        for metric_func in assessor.metric_functions.values():
            result = metric_func(y_true, y_pred)
            assert isinstance(result, (int, float, np.number))


class TestModelAssessmentEvaluate:
    """Test suite for evaluate method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def sklearn_model(self):
        """Create a mock sklearn model."""
        model = Mock()
        model.__class__.__name__ = "MockSklearnModel"
        model.predict = Mock(return_value=np.array([0, 1, 0, 1]))
        return model

    def test_evaluate_with_pandas_dataframe(self, temp_dir, sklearn_model):
        """Test evaluation with pandas DataFrame input."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy"],
            export_results=False,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        X_test = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = pd.Series([0, 1, 0, 1])

        results = assessor.evaluate(sklearn_model, X_test, y_test)

        assert "accuracy" in results["metrics"]
        assert isinstance(results["X_test"], np.ndarray)
        assert isinstance(results["true_values"], np.ndarray)

    def test_evaluate_with_export_results(self, temp_dir, sklearn_model):
        """Test evaluation with result export enabled."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy"],
            export_results=True,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 1, 0, 1])

        assessor.evaluate(sklearn_model, X_test, y_test)

        assert (temp_dir / "predictions.csv").exists()
        assert (temp_dir / "metrics_summary.csv").exists()

    def test_evaluate_with_report_generation(self, temp_dir, sklearn_model):
        """Test evaluation with report generation enabled."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy"],
            export_results=False,
            generate_report=True,
            report_title="Test Report",
        )

        mock_report_class = Mock()
        mock_report_instance = Mock()
        mock_report_class.return_value = mock_report_instance

        with patch(
            "ThreeWToolkit.reports.report_generation.ReportGeneration",
            mock_report_class,
        ):
            assessor = ModelAssessment(config)
            assessor._setup_metrics()

            X_test = np.array([[1, 2], [3, 4]])
            y_test = np.array([0, 1])

            assessor.evaluate(sklearn_model, X_test, y_test)

            mock_report_class.assert_called_once()
            mock_report_instance.generate_summary_report.assert_called_once()

    def test_evaluate_regression_model(self, temp_dir):
        """Test evaluation for regression tasks."""
        model = Mock()
        model.__class__.__name__ = "RegressionModel"
        model.predict = Mock(return_value=np.array([1.5, 2.5, 3.5]))

        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["explained_variance"],
            task_type=TaskType.REGRESSION,
            export_results=False,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        X_test = np.array([[1], [2], [3]])
        y_test = np.array([1.2, 2.3, 3.4])

        results = assessor.evaluate(model, X_test, y_test)

        assert "explained_variance" in results["metrics"]
        assert results["task_type"] == TaskType.REGRESSION


class TestModelAssessmentGetPredictions:
    """Test suite for _get_predictions method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir)
        return ModelAssessment(config)

    def test_get_predictions_sklearn_model(self, assessor):
        """Test getting predictions from sklearn-like model."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0]))
        X_test = np.array([[1, 2], [3, 4], [5, 6]])

        predictions = assessor._get_predictions(model, X_test)

        model.predict.assert_called_once_with(X_test)
        assert np.array_equal(predictions, np.array([0, 1, 0]))

    def test_get_predictions_sklearn_models_wrapper(self, assessor):
        """Test getting predictions from SklearnModels wrapper."""
        model = Mock(spec=SklearnModels)
        model.predict = Mock(return_value=np.array([1, 0, 1]))
        X_test = np.array([[1, 2], [3, 4], [5, 6]])

        predictions = assessor._get_predictions(model, X_test)

        model.predict.assert_called_once_with(X_test)
        assert np.array_equal(predictions, np.array([1, 0, 1]))

    def test_get_predictions_mlp_model(self, assessor):
        """Test getting predictions from PyTorch MLP model."""
        model = Mock(spec=MLP)
        expected_predictions = np.array([0, 1, 0, 1])

        with patch.object(
            assessor, "_get_mlp_predictions", return_value=expected_predictions
        ):
            X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            predictions = assessor._get_predictions(model, X_test)

            assert np.array_equal(predictions, expected_predictions)

    def test_get_predictions_with_kwargs(self, assessor):
        """Test getting predictions with additional kwargs."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1]))
        X_test = np.array([[1, 2], [3, 4]])

        _ = assessor._get_predictions(model, X_test, verbose=True)

        model.predict.assert_called_once_with(X_test, verbose=True)


class TestModelAssessmentGetMLPPredictions:
    """Test suite for _get_mlp_predictions method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir, batch_size=2, device="cpu")
        return ModelAssessment(config)

    def test_get_mlp_predictions(self, assessor):
        """Test getting predictions from MLP model."""
        model = Mock(spec=MLP)
        expected_predictions = np.array([0, 1, 0, 1])
        model.predict = Mock(return_value=expected_predictions)

        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        predictions = assessor._get_mlp_predictions(model, X_test)

        # Verify DataLoader was passed to predict
        call_args = model.predict.call_args
        assert isinstance(call_args[0][0], DataLoader)
        assert call_args[1]["device"] == "cpu"
        assert np.array_equal(predictions, expected_predictions)

    def test_get_mlp_predictions_with_kwargs(self, assessor):
        """Test MLP predictions with additional kwargs."""
        model = Mock(spec=MLP)
        model.predict = Mock(return_value=np.array([0, 1]))

        X_test = np.array([[1, 2], [3, 4]])

        _ = assessor._get_mlp_predictions(model, X_test, custom_arg="value")

        call_args = model.predict.call_args
        assert call_args[1]["custom_arg"] == "value"


class TestModelAssessmentCalculateMetrics:
    """Test suite for _calculate_metrics method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(
            output_dir=temp_dir, metrics=["accuracy", "f1", "precision"]
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()
        return assessor

    def test_calculate_metrics_success(self, assessor, capsys):
        """Test successful metric calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = assessor._calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert metrics["accuracy"] == 1.0
        assert isinstance(metrics["f1"], float)

    def test_calculate_metrics_with_errors(self, assessor, capsys):
        """Test metric calculation handles errors gracefully."""
        assessor.metric_functions["error_metric"] = Mock(
            side_effect=Exception("Test error")
        )
        assessor.config.metrics.append("error_metric")

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = assessor._calculate_metrics(y_true, y_pred)

        captured = capsys.readouterr()
        assert "Warning: Could not calculate error_metric" in captured.out
        assert np.isnan(metrics["error_metric"])

    def test_calculate_metrics_unavailable_metric(self, assessor, capsys):
        """Test handling of unavailable metrics."""
        assessor.config.metrics.append("unavailable_metric")

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = assessor._calculate_metrics(y_true, y_pred)

        captured = capsys.readouterr()
        assert "Warning: Metric 'unavailable_metric' not available" in captured.out
        assert np.isnan(metrics["unavailable_metric"])

    def test_calculate_metrics_returns_float(self, assessor):
        """Test that metrics are returned as Python floats."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = assessor._calculate_metrics(y_true, y_pred)

        for value in metrics.values():
            assert isinstance(value, (float, type(np.nan)))


class TestModelAssessmentHelperMethods:
    """Test suite for helper methods."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir)
        return ModelAssessment(config)

    def test_get_model_name_with_class(self, assessor):
        """Test getting model name from object with __class__."""
        model = Mock()
        model.__class__.__name__ = "TestModel"

        name = assessor._get_model_name(model)
        assert name == "TestModel"

    def test_to_numpy_from_dataframe(self, assessor):
        """Test converting pandas DataFrame to numpy."""
        df = pd.DataFrame([[1, 2], [3, 4]])
        result = assessor._to_numpy(df)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_to_numpy_from_series(self, assessor):
        """Test converting pandas Series to numpy."""
        series = pd.Series([1, 2, 3, 4])
        result = assessor._to_numpy(series)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4

    def test_to_numpy_from_ndarray(self, assessor):
        """Test that numpy arrays pass through unchanged."""
        arr = np.array([[1, 2], [3, 4]])
        result = assessor._to_numpy(arr)

        assert result is arr

    def test_to_numpy_from_list(self, assessor):
        """Test converting list to numpy array."""
        lst = [[1, 2], [3, 4]]
        result = assessor._to_numpy(lst)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)


class TestModelAssessmentExportResults:
    """Test suite for _export_results method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor_with_results(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir)
        assessor = ModelAssessment(config)
        assessor.results = {
            "model_name": "TestModel",
            "task_type": TaskType.CLASSIFICATION,
            "predictions": np.array([0, 1, 0, 1]),
            "true_values": np.array([0, 1, 0, 1]),
            "metrics": {"accuracy": 1.0, "f1": 1.0},
            "timestamp": "2024-01-01T00:00:00",
        }
        return assessor

    def test_export_results_creates_files(self, assessor_with_results, capsys):
        """Test that export creates both CSV files."""
        assessor_with_results._export_results()

        output_dir = assessor_with_results.config.output_dir
        assert (output_dir / "predictions.csv").exists()
        assert (output_dir / "metrics_summary.csv").exists()

        captured = capsys.readouterr()
        assert "Results exported to" in captured.out

    def test_export_results_predictions_content(self, assessor_with_results):
        """Test predictions CSV contains correct data."""
        assessor_with_results._export_results()

        predictions_path = assessor_with_results.config.output_dir / "predictions.csv"
        df = pd.read_csv(predictions_path)

        assert "true_values" in df.columns
        assert "predictions" in df.columns
        assert "metric_accuracy" in df.columns
        assert "metric_f1" in df.columns
        assert "model_name" in df.columns
        assert "task_type" in df.columns
        assert len(df) == 4

    def test_export_results_metrics_content(self, assessor_with_results):
        """Test metrics summary CSV contains correct data."""
        assessor_with_results._export_results()

        metrics_path = assessor_with_results.config.output_dir / "metrics_summary.csv"
        df = pd.read_csv(metrics_path)

        assert "accuracy" in df.columns
        assert "f1" in df.columns
        assert "model_name" in df.columns
        assert "task_type" in df.columns
        assert "timestamp" in df.columns
        assert len(df) == 1
        assert df["accuracy"].iloc[0] == 1.0


class TestModelAssessmentGetMetric:
    """Test suite for get_metric method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor_with_results(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir)
        assessor = ModelAssessment(config)
        assessor.results = {
            "metrics": {"accuracy": 0.85, "f1": 0.82, "precision": 0.88}
        }
        return assessor

    def test_get_metric_success(self, assessor_with_results):
        """Test successfully retrieving a metric value."""
        accuracy = assessor_with_results.get_metric("accuracy")
        assert accuracy == 0.85

        f1 = assessor_with_results.get_metric("f1")
        assert f1 == 0.82

    def test_get_metric_no_results(self, temp_dir):
        """Test get_metric fails when no evaluation has been run."""
        config = ModelAssessmentConfig(output_dir=temp_dir)
        assessor = ModelAssessment(config)

        with pytest.raises(ValueError, match="No evaluation results found"):
            assessor.get_metric("accuracy")

    def test_get_metric_not_found(self, assessor_with_results):
        """Test get_metric fails when metric doesn't exist."""
        with pytest.raises(ValueError, match="Metric 'nonexistent' not found"):
            assessor_with_results.get_metric("nonexistent")

    def test_get_metric_shows_available_metrics(self, assessor_with_results):
        """Test error message shows available metrics."""
        try:
            assessor_with_results.get_metric("invalid")
        except ValueError as e:
            assert "Available metrics:" in str(e)
            assert "accuracy" in str(e)
            assert "f1" in str(e)


class TestModelAssessmentSummary:
    """Test suite for summary method."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def assessor_with_results(self, temp_dir):
        config = ModelAssessmentConfig(output_dir=temp_dir)
        assessor = ModelAssessment(config)
        assessor.results = {
            "model_name": "RandomForestClassifier",
            "task_type": TaskType.CLASSIFICATION.value,
            "timestamp": "2024-01-15T10:30:45.123456",
            "metrics": {
                "accuracy": 0.8750,
                "f1": 0.8542,
                "precision": 0.8634,
                "invalid_metric": np.nan,
            },
        }
        return assessor

    def test_summary_no_results(self, temp_dir):
        """Test summary when no evaluation has been run."""
        config = ModelAssessmentConfig(output_dir=temp_dir)
        assessor = ModelAssessment(config)

        summary = assessor.summary()
        assert "No evaluation results available" in summary

    def test_summary_with_results(self, assessor_with_results):
        """Test summary generates correct format with results."""
        summary = assessor_with_results.summary()

        assert "Model Assessment Summary" in summary
        assert "========================" in summary
        assert "Model: RandomForestClassifier" in summary
        assert "Task Type: classification" in summary
        assert "Timestamp: 2024-01-15T10:30:45.123456" in summary
        assert "Metrics:" in summary
        assert "accuracy: 0.8750" in summary
        assert "f1: 0.8542" in summary
        assert "precision: 0.8634" in summary

    def test_summary_handles_nan_values(self, assessor_with_results):
        """Test summary displays N/A for NaN metric values."""
        summary = assessor_with_results.summary()
        assert "invalid_metric: N/A" in summary

    def test_summary_formatting(self, assessor_with_results):
        """Test summary has proper indentation and line breaks."""
        summary = assessor_with_results.summary()
        lines = summary.split("\n")

        # Check that metrics are indented
        metric_lines = [
            line
            for line in lines
            if line.strip().startswith(("accuracy", "f1", "precision"))
        ]
        for line in metric_lines:
            assert line.startswith("  ")

    def test_summary_with_regression(self, temp_dir):
        """Test summary for regression task."""
        config = ModelAssessmentConfig(output_dir=temp_dir)
        assessor = ModelAssessment(config)
        assessor.results = {
            "model_name": "LinearRegression",
            "task_type": TaskType.REGRESSION.value,
            "timestamp": "2024-01-15T10:30:45",
            "metrics": {"explained_variance": 0.9234},
        }

        summary = assessor.summary()
        assert "Task Type: regression" in summary
        assert "explained_variance: 0.9234" in summary


class TestModelAssessmentIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_complete_classification_workflow(self, temp_dir, capsys):
        """Test complete workflow from initialization to export."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy", "f1", "precision", "recall"],
            task_type=TaskType.CLASSIFICATION,
            export_results=True,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        # Create mock model
        model = Mock()
        model.__class__.__name__ = "RandomForest"
        model.predict = Mock(return_value=np.array([0, 1, 0, 1, 1, 0]))

        # Create test data
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y_test = np.array([0, 1, 0, 1, 1, 0])

        # Evaluate
        results = assessor.evaluate(model, X_test, y_test)

        # Verify results
        assert results["model_name"] == "RandomForest"
        assert "accuracy" in results["metrics"]
        assert results["metrics"]["accuracy"] == 1.0

        # Verify exports
        assert (temp_dir / "predictions.csv").exists()
        assert (temp_dir / "metrics_summary.csv").exists()

        # Verify metric retrieval
        accuracy = assessor.get_metric("accuracy")
        assert accuracy == 1.0

        # Verify summary
        summary = assessor.summary()
        assert "RandomForest" in summary

    def test_complete_regression_workflow(self, temp_dir):
        """Test complete workflow for regression task."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["explained_variance"],
            task_type=TaskType.REGRESSION,
            export_results=True,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        model = Mock()
        model.__class__.__name__ = "LinearRegression"
        model.predict = Mock(return_value=np.array([1.1, 2.0, 3.2, 4.1]))

        X_test = np.array([[1], [2], [3], [4]])
        y_test = np.array([1.0, 2.0, 3.0, 4.0])

        results = assessor.evaluate(model, X_test, y_test)

        assert results["task_type"] == TaskType.REGRESSION
        assert "explained_variance" in results["metrics"]

    def test_pipeline_workflow(self, temp_dir):
        """Test using ModelAssessment in a pipeline."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy"],
            export_results=False,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        # Simulate pipeline data
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0, 1]))

        pipeline_data = {
            "model": model,
            "x_test": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "y_test": np.array([0, 1, 0, 1]),
        }

        # Pre-process
        processed_data = assessor.pre_process(pipeline_data)
        assert "kwargs" in processed_data

        # Run
        run_data = assessor.run(processed_data)
        assert "assessment_results" in run_data
        assert "metrics" in run_data

        # Post-process
        final_data = assessor.post_process(run_data)
        assert final_data["assessment_completed"] is True
        assert "assessment_timestamp" in final_data
        assert "assessment_summary" in final_data

    def test_mlp_model_workflow(self, temp_dir):
        """Test workflow with PyTorch MLP model."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy"],
            batch_size=2,
            device="cpu",
            export_results=False,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        # Create mock MLP model
        model = Mock(spec=MLP)
        model.predict = Mock(return_value=np.array([0, 1, 0, 1]))

        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 1, 0, 1])

        _ = assessor.evaluate(model, X_test, y_test)

        # Verify DataLoader was created and used
        model.predict.assert_called_once()
        call_args = model.predict.call_args
        assert isinstance(call_args[0][0], DataLoader)
        assert call_args[1]["device"] == "cpu"

    def test_workflow_with_pandas_data(self, temp_dir):
        """Test workflow with pandas DataFrame and Series."""
        config = ModelAssessmentConfig(
            output_dir=temp_dir,
            metrics=["accuracy", "f1"],
            export_results=True,
            generate_report=False,
        )
        assessor = ModelAssessment(config)
        assessor._setup_metrics()

        model = Mock()
        model.__class__.__name__ = "DecisionTree"
        model.predict = Mock(return_value=np.array([0, 1, 0, 1, 1, 0]))

        X_test = pd.DataFrame(
            {"feature1": [1, 3, 5, 7, 9, 11], "feature2": [2, 4, 6, 8, 10, 12]}
        )
        y_test = pd.Series([0, 1, 0, 1, 1, 0])

        results = assessor.evaluate(model, X_test, y_test)

        assert isinstance(results["X_test"], np.ndarray)
        assert isinstance(results["true_values"], np.ndarray)
        assert results["metrics"]["accuracy"] == 1.0
