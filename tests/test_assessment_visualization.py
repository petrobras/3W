import pytest
import pydantic
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend for tests
from matplotlib.figure import Figure

from ThreeWToolkit.core.base_assessment_visualization import (
    AssessmentVisualizationConfig,
)
from ThreeWToolkit.assessment.assessment_visualizations import AssessmentVisualization


class TestAssessmentVisualization:
    @pytest.fixture
    def default_config(self):
        return AssessmentVisualizationConfig(class_names=["A", "B", "C"])

    @pytest.fixture
    def viz(self, default_config):
        return AssessmentVisualization(default_config)

    def test_plot_confusion_matrix_list(self, viz):
        y_true = [0, 1, 2, 2, 1]
        y_pred = [0, 2, 2, 2, 1]
        fig = viz.plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_plot_confusion_matrix_with_pandas(self, viz):
        y_true = pd.Series([0, 1, 1, 2])
        y_pred = pd.Series([0, 1, 2, 2])
        fig = viz.plot_confusion_matrix(y_true, y_pred, normalize=False)
        assert isinstance(fig, Figure)

    def test_plot_confusion_matrix_with_numpy(self, viz):
        y_true = np.array([0, 1, 1, 2])
        y_pred = np.array([0, 1, 2, 2])
        fig = viz.plot_confusion_matrix(y_true, y_pred, normalize=True)
        assert isinstance(fig, Figure)

    def test_plot_confusion_matrix_length_mismatch(self, viz):
        y_true = [0, 1]
        y_pred = [0]
        with pytest.raises(
            ValueError, match="length of y_true and y_pred must be the same"
        ):
            viz.plot_confusion_matrix(y_true, y_pred)

    def test_plot_confusion_matrix_type_error(self, viz):
        y_true = "not a valid type"
        y_pred = [0]
        with pytest.raises(
            TypeError, match="y_true must be a pandas Series, numpy array, or list"
        ):
            viz.plot_confusion_matrix(y_true, y_pred)

    def test_plot_confusion_matrix_class_names_mismatch(self):
        config = AssessmentVisualizationConfig(class_names=["A", "B", "C"])
        viz = AssessmentVisualization(config)
        y_true = [0, 1, 1, 1]
        y_pred = [0, 1, 1, 1]
        with pytest.raises(ValueError, match="Number of class names"):
            viz.plot_confusion_matrix(y_true, y_pred)

    def test_plot_confusion_matrix_no_class_names(self):
        config = AssessmentVisualizationConfig(class_names=None)
        viz = AssessmentVisualization(config)
        y_true = [0, 1, 1, 1]
        y_pred = [0, 1, 1, 1]
        fig = viz.plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_plot_confusion_matrix_with_ax(self, viz):
        fig, ax = plt.subplots()
        y_true = [0, 1, 2, 2, 1]
        y_pred = [0, 2, 2, 2, 1]
        returned_fig = viz.plot_confusion_matrix(y_true, y_pred, ax=ax)
        # Should return the same figure as the one from ax
        assert returned_fig is fig

    def test_plot_confusion_matrix_with_ax_and_custom_figsize(self, viz):
        fig, ax = plt.subplots(figsize=(8, 4))
        y_true = [0, 1, 2, 2, 1]
        y_pred = [0, 2, 2, 2, 1]
        returned_fig = viz.plot_confusion_matrix(y_true, y_pred, ax=ax, figsize=(8, 4))
        assert returned_fig is fig
        # The figure size should remain as specified
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 4

    def test_feature_visualization_basic(self, viz):
        feature_importances = [0.1, 0.4, 0.2, 0.3]
        feature_names = ["f1", "f2", "f3", "f4"]
        fig = viz.feature_visualization(feature_importances, feature_names)
        assert isinstance(fig, Figure)

    def test_feature_visualization_top_n(self, viz):
        feature_importances = [0.1, 0.4, 0.2, 0.3]
        feature_names = ["f1", "f2", "f3", "f4"]
        fig = viz.feature_visualization(feature_importances, feature_names, top_n=2)
        assert isinstance(fig, Figure)

    def test_feature_visualization_with_numpy(self, viz):
        feature_importances = np.array([0.1, 0.4, 0.2, 0.3])
        feature_names = np.array(["f1", "f2", "f3", "f4"])
        fig = viz.feature_visualization(feature_importances, feature_names, color="red")
        assert isinstance(fig, Figure)

    def test_feature_visualization_with_ax(self, viz):
        fig, ax = plt.subplots()
        feature_importances = [0.1, 0.4, 0.2, 0.3]
        feature_names = ["f1", "f2", "f3", "f4"]
        returned_fig = viz.feature_visualization(
            feature_importances, feature_names, ax=ax
        )
        assert returned_fig is fig

    def test_feature_visualization_length_mismatch(self, viz):
        feature_importances = [0.1, 0.4, 0.2]
        feature_names = ["f1", "f2", "f3", "f4"]
        with pytest.raises(
            ValueError,
            match="Length of feature_importances and feature_names must match.",
        ):
            viz.feature_visualization(feature_importances, feature_names)


class TestAssessmentVisualizationConfig:
    def test_accepts_none(self):
        cfg = AssessmentVisualizationConfig(class_names=None)
        assert cfg.class_names is None

    def test_accepts_valid_list(self):
        cfg = AssessmentVisualizationConfig(class_names=["A", "B", "C"])
        assert cfg.class_names == ["A", "B", "C"]

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="non-empty list"):
            AssessmentVisualizationConfig(class_names=[])

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            AssessmentVisualizationConfig(class_names=["A", ""])

    def test_rejects_non_string_element(self):
        with pytest.raises(
            pydantic.ValidationError, match="Input should be a valid string"
        ):
            AssessmentVisualizationConfig(class_names=["A", 123])


class TestBaseAssessmentVisualization:
    def test_config_is_set(self):
        config = AssessmentVisualizationConfig(class_names=["A", "B"])
        obj = AssessmentVisualization(config)
        assert obj.config == config
