import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from unittest.mock import patch

from ThreeWToolkit.data_visualization.clustering_plots import (
    DataQualityHeatmap,
    DendrogramPlot,
    ClusterSizeCurvePlot,
    SelectionHeatmapPlot,
    ClusteringOverlayPlot,
    RankedDistancePlot,
)
from ThreeWToolkit.clustering._utils import compute_dba_centroid


class TestDataQualityHeatmap:
    """Test suite for DataQualityHeatmap visualizer."""

    @pytest.fixture
    def quality_df(self):
        return pd.DataFrame(
            {
                "P-MON-CKP": [0.01, 0.05, 0.80],
                "T-TPT": [0.02, 0.90, 0.03],
            },
            index=["inst_0", "inst_1", "inst_2"],
        )

    def test_returns_figure_and_axes(self, quality_df):
        viz = DataQualityHeatmap(quality_df)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_provided_axes(self, quality_df):
        fig, ax = plt.subplots()
        viz = DataQualityHeatmap(quality_df)
        returned_fig, returned_ax = viz.plot(ax=ax)

        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_custom_title(self, quality_df):
        viz = DataQualityHeatmap(quality_df, title="Custom Title")
        fig, ax = viz.plot()

        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_instance_labels_shown(self, quality_df):
        labels = ["Well A", "Well B", "Well C"]
        viz = DataQualityHeatmap(quality_df, instance_labels=labels)
        fig, ax = viz.plot()

        tick_texts = [t.get_text() for t in ax.get_yticklabels()]
        assert tick_texts == labels
        plt.close(fig)

    def test_no_instance_labels_by_default(self, quality_df):
        viz = DataQualityHeatmap(quality_df)
        fig, ax = viz.plot()

        tick_texts = [t.get_text() for t in ax.get_yticklabels()]
        assert tick_texts == []
        plt.close(fig)

    def test_from_data_map_returns_instance(self):
        data_map = {
            "P-MON-CKP": [np.array([1.0, 2.0, 3.0]), np.array([np.nan, 2.0, 3.0])],
            "T-TPT": [np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])],
        }
        viz = DataQualityHeatmap.from_data_map(data_map)
        assert isinstance(viz, DataQualityHeatmap)
        assert viz.quality_df.shape == (2, 2)
        assert list(viz.quality_df.columns) == ["P-MON-CKP", "T-TPT"]

    def test_from_data_map_filters_variables(self):
        data_map = {
            "P-MON-CKP": [np.array([1.0, 2.0])],
            "T-TPT": [np.array([1.0, 2.0])],
        }
        viz = DataQualityHeatmap.from_data_map(data_map, variables=["T-TPT"])
        assert list(viz.quality_df.columns) == ["T-TPT"]

    def test_from_data_map_plots_without_error(self):
        data_map = {"P-MON-CKP": [np.array([1.0, np.nan, 3.0])]}
        viz = DataQualityHeatmap.from_data_map(data_map)
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_compute_quality_score_nan_only(self):
        series = np.array([np.nan, np.nan, 1.0, 2.0])
        score = DataQualityHeatmap._compute_quality_score(series, frozen_threshold=0.0)
        assert score == 0.5

    def test_compute_quality_score_frozen_only(self):
        series = np.array([1.0, 1.0, 1.0, 1.0])  # all diffs == 0
        score = DataQualityHeatmap._compute_quality_score(series, frozen_threshold=0.0)
        assert score == 1.0

    def test_compute_quality_score_empty_series(self):
        score = DataQualityHeatmap._compute_quality_score(
            np.array([]), frozen_threshold=0.0
        )
        assert score == 1.0

    def test_compute_quality_score_capped_at_one(self):
        series = np.array([np.nan, np.nan, 1.0, 1.0])  # 50% NaN + 33% frozen
        score = DataQualityHeatmap._compute_quality_score(series, frozen_threshold=0.0)
        assert score <= 1.0


class TestDendrogramPlot:
    """Test suite for DendrogramPlot visualizer."""

    @pytest.fixture
    def linkage_matrix(self):
        dm = np.array(
            [
                [0.0, 1.0, 5.0, 6.0],
                [1.0, 0.0, 4.0, 5.0],
                [5.0, 4.0, 0.0, 2.0],
                [6.0, 5.0, 2.0, 0.0],
            ]
        )
        condensed = squareform(dm)
        return linkage(condensed, method="average")

    def test_returns_figure_and_axes(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_with_threshold_line(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix, threshold=3.0)
        fig, ax = viz.plot()

        # Check that a horizontal line was drawn
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        assert len(lines) >= 1
        plt.close(fig)

    def test_without_threshold(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix, threshold=None)
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_uses_provided_axes(self, linkage_matrix):
        fig, ax = plt.subplots()
        viz = DendrogramPlot(linkage_matrix)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_show_instance_indices(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix, show_instance_indices=True)
        fig, ax = viz.plot()
        tick_texts = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_texts == ["0", "1", "2", "3"]
        plt.close(fig)

    def test_show_original_instance_indices(self, linkage_matrix):
        """Original pre-filter indices replace sequential labels."""
        original_indices = [0, 3, 7, 11]
        viz = DendrogramPlot(
            linkage_matrix,
            show_instance_indices=True,
            instance_indices=original_indices,
        )
        fig, ax = viz.plot()
        tick_texts = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_texts == ["0", "3", "7", "11"]
        plt.close(fig)

    def test_no_instance_indices_by_default(self, linkage_matrix):
        viz = DendrogramPlot(linkage_matrix)
        fig, ax = viz.plot()
        tick_texts = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_texts == []
        plt.close(fig)


class TestClusterSizeCurvePlot:
    """Test suite for ClusterSizeCurvePlot visualizer."""

    @pytest.fixture
    def common_counts(self):
        return {0.1: 2, 0.2: 3, 0.3: 4, 0.5: 5, 1.0: 5}

    def test_returns_figure_and_axes(self, common_counts):
        viz = ClusterSizeCurvePlot(common_counts)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_has_data(self, common_counts):
        viz = ClusterSizeCurvePlot(common_counts)
        fig, ax = viz.plot()
        assert ax.has_data()
        plt.close(fig)

    def test_uses_provided_axes(self, common_counts):
        fig, ax = plt.subplots()
        viz = ClusterSizeCurvePlot(common_counts)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)


class TestSelectionHeatmapPlot:
    """Test suite for SelectionHeatmapPlot visualizer."""

    @pytest.fixture
    def selection_data(self):
        mask = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
            ]
        )
        thresholds = [0.3, 0.5, 0.8]
        return mask, thresholds

    def test_returns_figure_and_axes(self, selection_data):
        mask, thresholds = selection_data
        viz = SelectionHeatmapPlot(mask, thresholds)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_provided_axes(self, selection_data):
        mask, thresholds = selection_data
        fig, ax = plt.subplots()
        viz = SelectionHeatmapPlot(mask, thresholds)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_instance_indices_on_y_axis(self, selection_data):
        mask, thresholds = selection_data
        original_indices = [0, 3, 7, 11, 15]
        viz = SelectionHeatmapPlot(mask, thresholds, instance_indices=original_indices)
        fig, ax = viz.plot()
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert y_labels == ["0", "3", "7", "11", "15"]
        plt.close(fig)

    def test_threshold_labels_on_x_axis(self, selection_data):
        mask, thresholds = selection_data
        viz = SelectionHeatmapPlot(mask, thresholds)
        fig, ax = viz.plot()
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert x_labels == ["0.30", "0.50", "0.80"]
        plt.close(fig)


class TestClusteringOverlayPlot:
    """Test suite for ClusteringOverlayPlot visualizer."""

    @pytest.fixture
    def series_data(self):
        np.random.seed(42)
        series = [np.random.randn(50) for _ in range(5)]
        selected = [0, 1, 2]
        return series, selected

    def test_returns_figure_and_axes(self, series_data):
        series, selected = series_data
        viz = ClusteringOverlayPlot(series, selected)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_with_centroid(self, series_data):
        series, selected = series_data
        centroid = np.zeros(50)
        viz = ClusteringOverlayPlot(series, selected, centroid=centroid)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_without_centroid(self, series_data):
        series, selected = series_data
        viz = ClusteringOverlayPlot(series, selected, centroid=None)
        fig, ax = viz.plot()

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_uses_provided_axes(self, series_data):
        series, selected = series_data
        fig, ax = plt.subplots()
        viz = ClusteringOverlayPlot(series, selected)
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_variable_length_series(self):
        series = [np.ones(30), np.ones(50), np.ones(100)]
        viz = ClusteringOverlayPlot(series, selected_indices=[0])
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestRankedDistancePlot:
    """Test suite for RankedDistancePlot visualizer."""

    @pytest.fixture
    def distance_matrix(self):
        """4×4 symmetric normalized distance matrix."""
        return np.array(
            [
                [0.0, 0.2, 0.8, 0.9],
                [0.2, 0.0, 0.7, 0.8],
                [0.8, 0.7, 0.0, 0.3],
                [0.9, 0.8, 0.3, 0.0],
            ]
        )

    def test_returns_figure_and_axes(self, distance_matrix):
        viz = RankedDistancePlot(distance_matrix, selected_indices=[0, 1])
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_has_data(self, distance_matrix):
        viz = RankedDistancePlot(distance_matrix, selected_indices=[0, 1])
        fig, ax = viz.plot()
        assert ax.has_data()
        plt.close(fig)

    def test_uses_provided_axes(self, distance_matrix):
        fig, ax = plt.subplots()
        viz = RankedDistancePlot(distance_matrix, selected_indices=[0, 1])
        returned_fig, _ = viz.plot(ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_custom_title(self, distance_matrix):
        viz = RankedDistancePlot(
            distance_matrix, selected_indices=[0, 1], title="My Plot"
        )
        fig, ax = viz.plot()
        assert ax.get_title() == "My Plot"
        plt.close(fig)

    def test_empty_selection_shows_message(self, distance_matrix):
        viz = RankedDistancePlot(distance_matrix, selected_indices=[])
        fig, ax = viz.plot()
        texts = [t.get_text() for t in ax.texts]
        assert any("No instances selected" in t for t in texts)
        plt.close(fig)

    def test_with_univariate_indices(self, distance_matrix):
        """Vetoed instances (locally valid, globally rejected) are accepted without error."""
        viz = RankedDistancePlot(
            distance_matrix,
            selected_indices=[0, 1],
            univariate_indices=[0, 1, 2],
        )
        fig, ax = viz.plot()
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_instance_labels_on_x_axis(self, distance_matrix):
        viz = RankedDistancePlot(
            distance_matrix,
            selected_indices=[0, 1],
            instance_labels=[10, 11, 12, 13],
        )
        fig, ax = viz.plot()
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(x_labels) == 4
        assert all(label in ["10", "11", "12", "13"] for label in x_labels)
        plt.close(fig)

    def test_bars_sorted_by_avg_distance(self, distance_matrix):
        """Instances closer to selected group appear on the right."""
        viz = RankedDistancePlot(distance_matrix, selected_indices=[0, 1])
        fig, ax = viz.plot()
        bar_heights = [p.get_height() for p in ax.patches]
        assert bar_heights == sorted(bar_heights)
        plt.close(fig)


class TestComputeDBAcentroid:
    """Test suite for the compute_dba_centroid utility."""

    @pytest.fixture
    def series(self):
        np.random.seed(0)
        return [np.random.randn(30) for _ in range(4)]

    def test_returns_ndarray(self, series):
        with patch("dtaidistance.dtw_barycenter.dba_loop", return_value=np.zeros(30)):
            result = compute_dba_centroid(series)
        assert isinstance(result, np.ndarray)

    def test_uses_subset_when_indices_provided(self, series):
        captured = {}

        def mock_dba(subset, **kwargs):
            captured["n"] = len(subset)
            return np.zeros(30)

        with patch("dtaidistance.dtw_barycenter.dba_loop", side_effect=mock_dba):
            compute_dba_centroid(series, indices=[0, 2])

        assert captured["n"] == 2

    def test_uses_all_series_when_no_indices(self, series):
        captured = {}

        def mock_dba(subset, **kwargs):
            captured["n"] = len(subset)
            return np.zeros(30)

        with patch("dtaidistance.dtw_barycenter.dba_loop", side_effect=mock_dba):
            compute_dba_centroid(series)

        assert captured["n"] == len(series)

    def test_raises_value_error_on_empty_subset(self, series):
        with pytest.raises(ValueError, match="No series provided"):
            compute_dba_centroid(series, indices=[])

    def test_raises_import_error_when_dtaidistance_missing(self, series):
        with patch.dict(
            "sys.modules", {"dtaidistance": None, "dtaidistance.dtw_barycenter": None}
        ):
            with pytest.raises(ImportError, match="dtaidistance"):
                compute_dba_centroid(series)
