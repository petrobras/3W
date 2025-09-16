import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression

# Assuming these are the imports from your project
from ThreeWToolkit.assessment.model_assess import ModelAssessment, ModelAssessmentConfig
from ThreeWToolkit.core.enums import TaskType
from ThreeWToolkit.models.mlp import MLP


class TestModelAssessmentConfig:
    """Test cases for ModelAssessmentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelAssessmentConfig()
        
        assert config.metrics == ["accuracy", "f1"]
        assert config.export_results is True
        assert config.generate_report is False
        assert config.task_type == TaskType.CLASSIFICATION
        assert config.batch_size == 64
        assert config.device in ["cpu", "cuda"]
        assert config.report_author == "3W Toolkit Report"

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_path = Path("./custom_results")
        config = ModelAssessmentConfig(
            metrics=["accuracy", "precision", "recall", "f1"],
            output_dir=custom_path,
            export_results=False,
            generate_report=True,
            task_type=TaskType.REGRESSION,
            batch_size=128,
            device="cpu",
            report_title="Custom Report",
            report_author="Test Author"
        )
        
        assert config.metrics == ["accuracy", "precision", "recall", "f1"]
        assert config.output_dir == custom_path
        assert config.export_results is False
        assert config.generate_report is True
        assert config.task_type == TaskType.REGRESSION
        assert config.batch_size == 128
        assert config.device == "cpu"
        assert config.report_title == "Custom Report"
        assert config.report_author == "Test Author"

    def test_invalid_task_type(self):
        """Test validation of invalid task type."""
        with pytest.raises(ValueError, match="Input should be 'classification' or 'regression'"):
            ModelAssessmentConfig(task_type="invalid_task")

    def test_invalid_batch_size(self):
        """Test validation of invalid batch size."""
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            ModelAssessmentConfig(batch_size=0)

    def test_invalid_device(self):
        """Test validation of invalid device."""
        with pytest.raises(ValueError, match="device must be one of"):
            ModelAssessmentConfig(device="invalid_device")

    def test_invalid_metrics(self):
        """Test validation of invalid metrics."""
        with pytest.raises(ValueError, match="Invalid metrics"):
            ModelAssessmentConfig(metrics=["invalid_metric", "accuracy"])

    def test_regression_metrics(self):
        """Test valid regression metrics."""
        config = ModelAssessmentConfig(
            metrics=["explained_variance"],
            task_type=TaskType.REGRESSION
        )
        assert config.metrics == ["explained_variance"]

    def test_classification_metrics(self):
        """Test valid classification metrics."""
        config = ModelAssessmentConfig(
            metrics=["accuracy", "precision", "recall", "f1", "balanced_accuracy"],
            task_type=TaskType.CLASSIFICATION
        )
        assert config.metrics == ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]


class TestModelAssessment:
    """Test cases for ModelAssessment class."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, 
            random_state=42, n_redundant=0
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=100, n_features=10, noise=0.1, random_state=42
        )
        return X, y

    @pytest.fixture
    def mock_mlp_model(self):
        """Create a mock MLP model."""
        mock_model = Mock(spec=MLP)
        mock_model.__class__.__name__ = "MLP"
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        return mock_model

    @pytest.fixture
    def mock_sklearn_model(self):
        """Create a mock sklearn model."""
        mock_model = Mock()
        mock_model.__class__.__name__ = "LogisticRegression"
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        return mock_model

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return tmp_path / "test_assessment"

    def test_initialization_default_config(self):
        """Test ModelAssessment initialization with default config."""
        config = ModelAssessmentConfig()
        assessor = ModelAssessment(config)
        
        assert assessor.config == config
        assert assessor.results == {}
        assert hasattr(assessor, 'metric_functions')
        assert assessor.config.output_dir.exists()

    def test_initialization_classification_metrics(self):
        """Test initialization with classification metrics setup."""
        config = ModelAssessmentConfig(task_type=TaskType.CLASSIFICATION)
        assessor = ModelAssessment(config)
        
        expected_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "average_precision"]
        for metric in expected_metrics:
            assert metric in assessor.metric_functions

    def test_initialization_regression_metrics(self):
        """Test initialization with regression metrics setup."""
        config = ModelAssessmentConfig(task_type=TaskType.REGRESSION)
        assessor = ModelAssessment(config)
        
        expected_metrics = ["explained_variance"]
        for metric in expected_metrics:
            assert metric in assessor.metric_functions

    def test_to_numpy_conversion(self):
        """Test data conversion to numpy arrays."""
        config = ModelAssessmentConfig()
        assessor = ModelAssessment(config)
        
        # Test pandas DataFrame
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        result = assessor._to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
        
        # Test pandas Series
        series = pd.Series([1, 2, 3])
        result = assessor._to_numpy(series)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        
        # Test numpy array
        arr = np.array([1, 2, 3])
        result = assessor._to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_get_model_name(self):
        """Test model name extraction."""
        config = ModelAssessmentConfig()
        assessor = ModelAssessment(config)
        
        # Test with real sklearn model
        model = LogisticRegression()
        name = assessor._get_model_name(model)
        assert name == "LogisticRegression"
        
        # Test with object without __class__
        mock_model = Mock()
        del mock_model.__class__
        name = assessor._get_model_name(mock_model)
        assert name == "Mock"

    def test_evaluate_sklearn_classification(self, classification_data, temp_output_dir):
        """Test evaluation with sklearn classification model."""
        X, y = classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Configure assessment
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1", "precision", "recall"],
            task_type=TaskType.CLASSIFICATION,
            output_dir=temp_output_dir,
            export_results=False
        )
        
        assessor = ModelAssessment(config)
        results = assessor.evaluate(model, X_test, y_test)
        
        # Verify results structure
        assert "model_name" in results
        assert "predictions" in results
        assert "true_values" in results
        assert "metrics" in results
        assert results["model_name"] == "LogisticRegression"
        assert len(results["predictions"]) == len(y_test)
        
        # Verify metrics
        for metric in config.metrics:
            assert metric in results["metrics"]
            assert isinstance(results["metrics"][metric], (int, float))
            assert not np.isnan(results["metrics"][metric])

    def test_evaluate_sklearn_regression(self, regression_data, temp_output_dir):
        """Test evaluation with sklearn regression model."""
        X, y = regression_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Configure assessment
        config = ModelAssessmentConfig(
            metrics=["explained_variance"],
            task_type=TaskType.REGRESSION,
            output_dir=temp_output_dir,
            export_results=False
        )
        
        assessor = ModelAssessment(config)
        results = assessor.evaluate(model, X_test, y_test)
        
        # Verify results
        assert results["model_name"] == "LinearRegression"
        assert len(results["predictions"]) == len(y_test)
        
        # Verify regression metrics
        for metric in config.metrics:
            assert metric in results["metrics"]
            assert isinstance(results["metrics"][metric], (int, float))

    @patch('torch.utils.data.DataLoader')
    def test_evaluate_mlp_model(self, mock_dataloader, mock_mlp_model, classification_data, temp_output_dir):
        """Test evaluation with MLP model."""
        X, y = classification_data
        X_test, y_test = X[80:], y[80:]
        
        # Configure mock
        mock_mlp_model.predict.return_value = np.random.randint(0, 2, len(y_test))
        
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1"],
            task_type=TaskType.CLASSIFICATION,
            output_dir=temp_output_dir,
            export_results=False
        )
        
        assessor = ModelAssessment(config)
        results = assessor.evaluate(mock_mlp_model, X_test, y_test)
        
        # Verify MLP-specific behavior
        assert results["model_name"] == "MLP"
        mock_mlp_model.predict.assert_called_once()

    def test_calculate_metrics_classification(self, temp_output_dir):
        """Test metrics calculation for classification."""
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1", "precision", "recall"],
            task_type=TaskType.CLASSIFICATION,
            output_dir=temp_output_dir
        )
        
        assessor = ModelAssessment(config)
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = assessor._calculate_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        
        # Verify metric values are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_calculate_metrics_regression(self, temp_output_dir):
        """Test metrics calculation for regression."""
        config = ModelAssessmentConfig(
            metrics=["explained_variance"],
            task_type=TaskType.REGRESSION,
            output_dir=temp_output_dir
        )
        
        assessor = ModelAssessment(config)
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        metrics = assessor._calculate_metrics(y_true, y_pred)
        
        assert "explained_variance" in metrics

    def test_export_results(self, mock_sklearn_model, classification_data, temp_output_dir):
        """Test results export functionality."""
        X, y = classification_data
        X_test, y_test = X[80:], y[80:]
        
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1"],
            output_dir=temp_output_dir,
            export_results=True
        )
        
        assessor = ModelAssessment(config)
        
        # Mock the model prediction
        mock_sklearn_model.predict.return_value = np.random.randint(0, 2, len(y_test))
        
        results = assessor.evaluate(mock_sklearn_model, X_test, y_test)
        
        # Check if files were created
        predictions_file = temp_output_dir / "predictions.csv"
        metrics_file = temp_output_dir / "metrics_summary.csv"
        
        assert predictions_file.exists()
        assert metrics_file.exists()
        
        # Verify file contents
        predictions_df = pd.read_csv(predictions_file)
        assert "true_values" in predictions_df.columns
        assert "predictions" in predictions_df.columns
        assert len(predictions_df) == len(y_test)
        
        metrics_df = pd.read_csv(metrics_file)
        assert "model_name" in metrics_df.columns
        assert "task_type" in metrics_df.columns

    def test_get_metric_method(self, mock_sklearn_model, classification_data, temp_output_dir):
        """Test get_metric convenience method."""
        X, y = classification_data
        X_test, y_test = X[80:], y[80:]
        
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1"],
            output_dir=temp_output_dir,
            export_results=False
        )
        
        assessor = ModelAssessment(config)
        
        # Before evaluation, should raise error
        with pytest.raises(ValueError, match="No evaluation results found"):
            assessor.get_metric("accuracy")
        
        # After evaluation
        mock_sklearn_model.predict.return_value = np.random.randint(0, 2, len(y_test))
        assessor.evaluate(mock_sklearn_model, X_test, y_test)
        
        # Should return metric value
        accuracy = assessor.get_metric("accuracy")
        assert isinstance(accuracy, (int, float))
        assert 0 <= accuracy <= 1
        
        # Non-existent metric should raise error
        with pytest.raises(ValueError, match="Metric 'invalid' not found"):
            assessor.get_metric("invalid")

    def test_summary_method(self, mock_sklearn_model, classification_data, temp_output_dir):
        """Test summary method."""
        X, y = classification_data
        X_test, y_test = X[80:], y[80:]
        
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1"],
            output_dir=temp_output_dir,
            export_results=False
        )
        
        assessor = ModelAssessment(config)
        
        # Before evaluation
        summary = assessor.summary()
        assert "No evaluation results available" in summary
        
        # After evaluation
        mock_sklearn_model.predict.return_value = np.random.randint(0, 2, len(y_test))
        assessor.evaluate(mock_sklearn_model, X_test, y_test)
        
        summary = assessor.summary()
        assert "Model Assessment Summary" in summary
        assert "accuracy:" in summary
        assert "f1:" in summary

    def test_pandas_input_handling(self, temp_output_dir):
        """Test handling of pandas DataFrame and Series inputs."""
        # Create pandas data
        X_df = pd.DataFrame(np.random.rand(20, 5), columns=[f'feature_{i}' for i in range(5)])
        y_series = pd.Series(np.random.randint(0, 2, 20))
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_df.values, y_series.values)
        
        config = ModelAssessmentConfig(
            metrics=["accuracy"],
            task_type=TaskType.CLASSIFICATION,
            output_dir=temp_output_dir,
            export_results=False
        )
        
        assessor = ModelAssessment(config)
        results = assessor.evaluate(model, X_df, y_series)
        
        # Should handle pandas inputs correctly
        assert "metrics" in results
        assert "accuracy" in results["metrics"]

    def test_map_metrics_for_report(self, temp_output_dir):
        """Test metric mapping for report generation."""
        config = ModelAssessmentConfig(
            metrics=["accuracy", "f1", "explained_variance"],
            output_dir=temp_output_dir
        )
        
        assessor = ModelAssessment(config)
        mapped_metrics = assessor._map_metrics_for_report()
        
        expected_mapping = ["accuracy", "f1", "explained_variance"]
        assert mapped_metrics == expected_mapping


class TestModelAssessmentIntegration:
    """Integration tests for ModelAssessment with real models."""

    def test_full_classification_pipeline(self, tmp_path):
        """Test complete classification assessment pipeline."""
        # Generate data
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Configure assessment
        config = ModelAssessmentConfig(
            metrics=["accuracy", "precision", "recall", "f1", "balanced_accuracy"],
            task_type=TaskType.CLASSIFICATION,
            output_dir=tmp_path / "classification_results",
            export_results=True
        )
        
        # Run assessment
        assessor = ModelAssessment(config)
        results = assessor.evaluate(model, X_test, y_test)
        
        # Verify complete results
        assert all(metric in results["metrics"] for metric in config.metrics)
        assert all(not np.isnan(results["metrics"][metric]) for metric in config.metrics)
        assert results["task_type"] == TaskType.CLASSIFICATION
        
        # Verify files were created
        assert (tmp_path / "classification_results" / "predictions.csv").exists()
        assert (tmp_path / "classification_results" / "metrics_summary.csv").exists()
        
        # Verify summary works
        summary = assessor.summary()
        assert "RandomForestClassifier" in summary

    def test_full_regression_pipeline(self, tmp_path):
        """Test complete regression assessment pipeline."""
        # Generate data
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Configure assessment
        config = ModelAssessmentConfig(
            metrics=["explained_variance"],
            task_type=TaskType.REGRESSION,
            output_dir=tmp_path / "regression_results",
            export_results=True
        )
        
        # Run assessment
        assessor = ModelAssessment(config)
        results = assessor.evaluate(model, X_test, y_test)
        
        # Verify complete results
        assert all(metric in results["metrics"] for metric in config.metrics)
        assert all(not np.isnan(results["metrics"][metric]) for metric in config.metrics)
        assert results["task_type"] == TaskType.REGRESSION
        
        # Verify files were created
        assert (tmp_path / "regression_results" / "predictions.csv").exists()
        assert (tmp_path / "regression_results" / "metrics_summary.csv").exists()