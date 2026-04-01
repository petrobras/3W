"""Tests for BaseAssessment and BaseAssessmentVisualization."""

import pytest
from pydantic import ValidationError

from ThreeWToolkit.core import (
    BaseAssessmentVisualization,
    TaskTypeEnum,
    DataSplitEnum,
)
from ThreeWToolkit.assessment import (
    ModelAssessmentConfig,
    AssessmentVisualizationConfig,
)


class TestModelAssessmentConfig:
    """Test ModelAssessmentConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelAssessmentConfig()

        assert config.metrics == ["accuracy", "f1"]
        assert config.export_results is True
        assert config.generate_report is False
        assert config.task_type == TaskTypeEnum.CLASSIFICATION
        assert config.batch_size == 64
        assert config.dataset_split == DataSplitEnum.TEST

    def test_custom_metrics(self):
        """Test custom metrics list."""
        config = ModelAssessmentConfig(
            metrics=["accuracy", "precision", "recall", "f1"]
        )

        assert len(config.metrics) == 4
        assert "precision" in config.metrics
        assert "recall" in config.metrics

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric name raises error."""
        with pytest.raises(ValidationError):
            ModelAssessmentConfig(metrics=["accuracy", "invalid_metric"])

    def test_valid_classification_metrics(self):
        """Test all valid classification metrics."""
        valid_metrics = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
        ]
        config = ModelAssessmentConfig(metrics=valid_metrics)
        assert config.metrics == valid_metrics

    def test_valid_regression_metric(self):
        """Test valid regression metric."""
        config = ModelAssessmentConfig(
            metrics=["explained_variance"],
            task_type=TaskTypeEnum.REGRESSION,
        )
        assert "explained_variance" in config.metrics

    def test_classification_task_type(self):
        """Test classification task type."""
        config = ModelAssessmentConfig(task_type=TaskTypeEnum.CLASSIFICATION)
        assert config.task_type == TaskTypeEnum.CLASSIFICATION

    def test_regression_task_type(self):
        """Test regression task type."""
        config = ModelAssessmentConfig(task_type=TaskTypeEnum.REGRESSION)
        assert config.task_type == TaskTypeEnum.REGRESSION

    def test_custom_output_dir(self, tmp_path):
        """Test custom output directory."""
        config = ModelAssessmentConfig(output_dir=tmp_path)
        assert config.output_dir == tmp_path

    def test_batch_size_positive(self):
        """Test that batch size must be positive."""
        with pytest.raises(ValidationError):
            ModelAssessmentConfig(batch_size=0)

        with pytest.raises(ValidationError):
            ModelAssessmentConfig(batch_size=-1)

    def test_custom_batch_size(self):
        """Test custom batch size."""
        config = ModelAssessmentConfig(batch_size=128)
        assert config.batch_size == 128

    def test_device_cpu(self):
        """Test CPU device."""
        config = ModelAssessmentConfig(device="cpu")
        assert config.device == "cpu"

    def test_device_cuda(self):
        """Test CUDA device."""
        config = ModelAssessmentConfig(device="cuda")
        assert config.device == "cuda"

    def test_invalid_device_raises_error(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValidationError):
            ModelAssessmentConfig(device="tpu")

    def test_report_title_optional(self):
        """Test optional report title."""
        config = ModelAssessmentConfig(report_title=None)
        assert config.report_title is None

        config = ModelAssessmentConfig(report_title="My Report")
        assert config.report_title == "My Report"

    def test_report_author_default(self):
        """Test default report author."""
        config = ModelAssessmentConfig()
        assert config.report_author == "3W Toolkit Report"

    def test_custom_report_author(self):
        """Test custom report author."""
        config = ModelAssessmentConfig(report_author="Test Author")
        assert config.report_author == "Test Author"

    def test_dataset_split_values(self):
        """Test all dataset split values."""
        for split in [
            DataSplitEnum.TRAIN,
            DataSplitEnum.VALIDATION,
            DataSplitEnum.TEST,
            DataSplitEnum.CUSTOM,
        ]:
            config = ModelAssessmentConfig(dataset_split=split)
            assert config.dataset_split == split


class TestAssessmentVisualizationConfig:
    """Test AssessmentVisualizationConfig validation."""

    def test_default_class_names_none(self):
        """Test default class names is None."""
        config = AssessmentVisualizationConfig()
        assert config.class_names is None

    def test_valid_class_names(self):
        """Test valid class names list."""
        config = AssessmentVisualizationConfig(class_names=["Normal", "Anomaly"])
        assert config.class_names == ["Normal", "Anomaly"]

    def test_empty_class_names_raises_error(self):
        """Test that empty class names raises error."""
        with pytest.raises(ValidationError):
            AssessmentVisualizationConfig(class_names=[])

    def test_non_string_class_names_raises_error(self):
        """Test that non-string class names raises error."""
        with pytest.raises(ValidationError):
            AssessmentVisualizationConfig(class_names=["Valid", 123])

    def test_empty_string_class_name_raises_error(self):
        """Test that empty string class name raises error."""
        with pytest.raises(ValidationError):
            AssessmentVisualizationConfig(class_names=["Valid", ""])

    def test_whitespace_only_class_name_raises_error(self):
        """Test that whitespace-only class name raises error."""
        with pytest.raises(ValidationError):
            AssessmentVisualizationConfig(class_names=["Valid", "   "])

    def test_multiple_class_names(self):
        """Test multiple class names."""
        names = ["Class A", "Class B", "Class C", "Class D"]
        config = AssessmentVisualizationConfig(class_names=names)
        assert config.class_names == names
        assert len(config.class_names) == 4


class TestBaseAssessmentVisualization:
    """Test BaseAssessmentVisualization base class."""

    def test_visualization_stores_config(self):
        """Test that visualization stores config."""

        class ConcreteVisualization(BaseAssessmentVisualization):
            pass

        config = AssessmentVisualizationConfig(class_names=["A", "B"])
        viz = ConcreteVisualization(config)

        assert viz.config.class_names == ["A", "B"]

    def test_visualization_with_none_class_names(self):
        """Test visualization with None class names."""

        class ConcreteVisualization(BaseAssessmentVisualization):
            pass

        config = AssessmentVisualizationConfig()
        viz = ConcreteVisualization(config)

        assert viz.config.class_names is None
