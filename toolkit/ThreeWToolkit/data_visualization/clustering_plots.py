from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from .base_visualizer import BaseVisualizer


class DataQualityHeatmap(BaseVisualizer):
    """Heatmap of quality metrics (NaN and frozen ratios) per instance × variable.

    Expects a DataFrame where rows are instances and columns are sensor variables,
    with cell values being a quality score in [0, 1] (e.g., NaN ratio, frozen ratio,
    or a combined metric).

    Use the ``from_data_map`` factory to build directly from the raw
    ``{variable: [array, ...]}`` structure returned by
    ``ParquetDataset.load_instances_by_variable()``.
    """

    def __init__(
        self,
        quality_df: pd.DataFrame,
        title: str = "Data Quality Heatmap",
        figsize: tuple[int, int] = (12, 8),
        instance_labels: list[str] | None = None,
    ) -> None:
        self.quality_df = quality_df
        self.title = title
        self.figsize = figsize
        self.instance_labels = instance_labels

    @classmethod
    def from_data_map(
        cls,
        data_map: dict[str, list[np.ndarray]],
        variables: list[str] | None = None,
        frozen_threshold: float = 0.0,
        title: str = "NaN + Frozen Ratio per Instance x Variable",
        figsize: tuple[int, int] = (12, 8),
        instance_labels: list[str] | None = None,
    ) -> "DataQualityHeatmap":
        """Build a DataQualityHeatmap directly from a raw data map.

        Computes a combined quality score per instance per variable:
        ``min(nan_ratio + frozen_ratio, 1.0)``.

        Args:
            data_map: Mapping of variable name to list of time series arrays,
                as returned by ``ParquetDataset.load_instances_by_variable()``.
            variables: Subset of variables to include. Defaults to all keys in
                ``data_map``.
            frozen_threshold: Consecutive differences with absolute value at or
                below this value are counted as frozen. Defaults to ``0.0``.
            title: Plot title.
            figsize: Figure size in inches.
            instance_labels: Optional y-axis labels, one per instance row.

        Returns:
            DataQualityHeatmap: Ready-to-plot instance.
        """
        target_vars = variables if variables is not None else list(data_map.keys())
        quality_data = {
            var: [
                cls._compute_quality_score(series, frozen_threshold)
                for series in data_map[var]
            ]
            for var in target_vars
            if var in data_map
        }
        return cls(
            pd.DataFrame(quality_data),
            title=title,
            figsize=figsize,
            instance_labels=instance_labels,
        )

    @staticmethod
    def _compute_quality_score(series: np.ndarray, frozen_threshold: float) -> float:
        """Combined NaN + frozen defect ratio for a single series, capped at 1.0."""
        if len(series) == 0:
            return 1.0
        nan_ratio = float(np.isnan(series).mean())
        diffs = np.diff(series)
        frozen_ratio = (
            float((np.abs(diffs) <= frozen_threshold).mean()) if len(diffs) > 0 else 0.0
        )
        return min(nan_ratio + frozen_ratio, 1.0)

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        df = self.quality_df.copy()
        if self.instance_labels is not None:
            df.index = self.instance_labels

        show_labels = self.instance_labels is not None
        label_fontsize = max(4, min(10, 200 // len(df))) if show_labels else 10

        sns.heatmap(
            df,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Quality Score (0 = clean, 1 = bad)"},
            yticklabels=show_labels,
        )
        if show_labels:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=label_fontsize)
        ax.set_title(self.title)
        ax.set_xlabel("Sensor Variable")
        ax.set_ylabel("Instance")
        fig.tight_layout()
        return fig, ax


class DendrogramPlot(BaseVisualizer):
    """Dendrogram of the hierarchical clustering tree with an optional threshold cut.

    The threshold cut is drawn as a horizontal dashed red line and corresponds to the
    normalized distance at which the main cluster is extracted.
    """

    def __init__(
        self,
        linkage_matrix: np.ndarray,
        threshold: float | None = None,
        title: str = "Hierarchical Clustering Dendrogram",
        figsize: tuple[int, int] = (14, 6),
        show_instance_indices: bool = False,
        instance_indices: list[int] | None = None,
    ) -> None:
        self.linkage_matrix = linkage_matrix
        self.threshold = threshold
        self.title = title
        self.figsize = figsize
        self.show_instance_indices = show_instance_indices
        self.instance_indices = instance_indices

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        n_instances = len(self.linkage_matrix) + 1

        if self.show_instance_indices:
            labels = (
                [str(i) for i in self.instance_indices]
                if self.instance_indices is not None
                else [str(i) for i in range(n_instances)]
            )
        else:
            labels = None

        color_threshold = self.threshold if self.threshold is not None else 0.0
        dendrogram(
            self.linkage_matrix,
            ax=ax,
            color_threshold=color_threshold,
            above_threshold_color="gray",
            labels=labels,
            no_labels=not self.show_instance_indices,
        )

        if self.show_instance_indices:
            label_fontsize = max(4, min(9, 200 // n_instances))
            ax.set_xticklabels(
                ax.get_xticklabels(), fontsize=label_fontsize, rotation=90
            )

        if self.threshold is not None:
            ax.axhline(
                y=self.threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold = {self.threshold:.2f}",
            )
            ax.legend()

        ax.set_title(self.title)
        ax.set_xlabel("Instances")
        ax.set_ylabel("Normalized Distance")
        fig.tight_layout()
        return fig, ax


class ClusterSizeCurvePlot(BaseVisualizer):
    """Line plot of main cluster size versus distance threshold.

    Accepts ``common_counts`` directly from ``MultivariateConsensus.common_counts_``
    (a dict mapping each threshold to the number of surviving instances).
    """

    def __init__(
        self,
        common_counts: dict[float, int],
        title: str = "Main Cluster Size vs. Threshold",
        figsize: tuple[int, int] = (10, 5),
    ) -> None:
        self.common_counts = common_counts
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        thresholds = list(self.common_counts.keys())
        counts = list(self.common_counts.values())

        markerline, stemlines, baseline = ax.stem(thresholds, counts)
        plt.setp(markerline, color="steelblue", markersize=5)
        plt.setp(stemlines, color="steelblue", linewidth=1.2)
        plt.setp(baseline, color="gray", linewidth=0.8)
        ax.set_title(self.title)
        ax.set_xlabel("Normalized Distance Threshold")
        ax.set_ylabel("Cluster Size (# Instances)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return fig, ax


class SelectionHeatmapPlot(BaseVisualizer):
    """Binary heatmap showing which instances are selected at each distance threshold.

    Layout: instances on the y-axis, distance thresholds (0.0 → 1.0) on the x-axis.
    A white cell means the instance was selected at that threshold; black means rejected.

    Accepts ``selection_mask`` and ``thresholds_analyzed_`` directly from a fitted
    ``MultivariateConsensus`` instance. Optionally accepts ``instance_indices`` to
    label rows with original pre-filter dataset indices instead of local ones.
    """

    def __init__(
        self,
        selection_mask: np.ndarray,
        thresholds: list[float],
        title: str = "Instance Selection Heatmap",
        figsize: tuple[int, int] = (14, 6),
        instance_indices: list[int] | None = None,
    ) -> None:
        self.selection_mask = selection_mask
        self.thresholds = thresholds
        self.title = title
        self.figsize = figsize
        self.instance_indices = instance_indices

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        # Transpose: rows = instances, columns = thresholds
        matrix = self.selection_mask.T
        n_instances = matrix.shape[0]

        y_labels = (
            [str(i) for i in self.instance_indices]
            if self.instance_indices is not None
            else [str(i) for i in range(n_instances)]
        )
        x_labels = [f"{t:.2f}" for t in self.thresholds]
        label_fontsize = max(4, min(9, 200 // n_instances))

        df = pd.DataFrame(matrix, index=y_labels, columns=x_labels)
        sns.heatmap(
            df,
            ax=ax,
            cmap="gray",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=label_fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha="right")
        ax.set_title(self.title)
        ax.set_xlabel("Distance Threshold")
        ax.set_ylabel("Instance Index")
        fig.tight_layout()
        return fig, ax


class ClusteringOverlayPlot(BaseVisualizer):
    """Overlays all time series, distinguishing selected from rejected instances.

    Selected instances are drawn in blue; rejected instances are drawn in gray.
    An optional pre-computed centroid array (e.g. DBA) is overlaid in red on top.

    All series are plotted on a normalized [0, 1] time axis to accommodate
    variable-length inputs.
    """

    def __init__(
        self,
        series: list[np.ndarray],
        selected_indices: list[int],
        centroid: np.ndarray | None = None,
        title: str = "Clustering Overlay",
        figsize: tuple[int, int] = (14, 6),
    ) -> None:
        self.series = series
        self.selected_indices = selected_indices
        self.centroid = centroid
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        selected_set = set(self.selected_indices)

        for i, ts in enumerate(self.series):
            x = np.linspace(0, 1, len(ts))
            if i in selected_set:
                ax.plot(x, ts, color="steelblue", alpha=0.9, linewidth=1.0, zorder=2)
            else:
                ax.plot(x, ts, color="lightgray", alpha=0.5, linewidth=0.6, zorder=1)

        if self.centroid is not None:
            x_centroid = np.linspace(0, 1, len(self.centroid))
            ax.plot(
                x_centroid,
                self.centroid,
                color="red",
                linewidth=2.5,
                zorder=3,
                label="Centroid",
            )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="steelblue",
                linewidth=1.5,
                label=f"Selected ({len(self.selected_indices)})",
            ),
            Line2D(
                [0],
                [0],
                color="lightgray",
                linewidth=1.5,
                label=f"Rejected ({len(self.series) - len(self.selected_indices)})",
            ),
        ]
        if self.centroid is not None:
            legend_elements.append(
                Line2D([0], [0], color="red", linewidth=2.5, label="Centroid")
            )

        ax.legend(handles=legend_elements, loc="upper right")
        ax.set_title(self.title)
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("Value")
        fig.tight_layout()
        return fig, ax


class RankedDistancePlot(BaseVisualizer):
    """Bar chart ranking instances by their average distance to the selected group.

    Instances are sorted from most distant (left, outlier) to closest (right, centroid).
    Three categories are distinguished by color:

    - **Blue** — consensus selected (survives across all variables at this threshold).
    - **Blue hatched** — vetoed: valid locally (in this variable's cluster) but rejected
      by the multivariate consensus.
    - **Salmon** — local outlier: rejected even within this variable's univariate cluster.

    Args:
        distance_matrix: Square normalized distance matrix ``(n, n)``.
        selected_indices: Local indices of consensus-selected instances.
        univariate_indices: Local indices of the univariate main cluster at the same
            threshold. Used to identify vetoed instances. If ``None``, all non-selected
            instances are treated as local outliers.
        instance_labels: Original dataset indices to show on the x-axis. If ``None``,
            local indices are used.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        selected_indices: list[int],
        univariate_indices: list[int] | None = None,
        instance_labels: list[int] | None = None,
        title: str = "Ranked Distance Plot",
        figsize: tuple[int, int] = (14, 5),
    ) -> None:
        self.distance_matrix = distance_matrix
        self.selected_indices = selected_indices
        self.univariate_indices = univariate_indices
        self.instance_labels = instance_labels
        self.title = title
        self.figsize = figsize

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = cast(Figure, ax.get_figure())

        if not self.selected_indices:
            ax.text(
                0.5,
                0.5,
                "No instances selected",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(self.title)
            return fig, ax

        # Average distance from each instance to the selected group
        avg_distances = np.mean(self.distance_matrix[:, self.selected_indices], axis=1)
        sorted_indices = np.argsort(avg_distances)
        sorted_distances = avg_distances[sorted_indices]

        selected_set = set(self.selected_indices)
        univariate_set = (
            set(self.univariate_indices)
            if self.univariate_indices is not None
            else set()
        )

        colors: list[str] = []
        hatches: list[str | None] = []
        for idx in sorted_indices:
            if idx in selected_set:
                colors.append("steelblue")
                hatches.append(None)
            elif idx in univariate_set:
                colors.append("steelblue")
                hatches.append("///")
            else:
                colors.append("salmon")
                hatches.append(None)

        bars = ax.bar(
            range(len(sorted_indices)),
            sorted_distances,
            color=colors,
            alpha=0.8,
            edgecolor="white",
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)

        # X-axis labels
        labels = (
            [str(self.instance_labels[i]) for i in sorted_indices]
            if self.instance_labels is not None
            else [str(i) for i in sorted_indices]
        )
        if len(sorted_indices) < 50:
            ax.set_xticks(range(len(sorted_indices)))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)

        legend_elements = [
            Patch(
                facecolor="steelblue", label=f"Consensus selected ({len(selected_set)})"
            ),
            Patch(facecolor="salmon", label="Local outlier"),
        ]
        if self.univariate_indices is not None:
            legend_elements.insert(
                1,
                Patch(
                    facecolor="steelblue", hatch="///", label="Vetoed (valid locally)"
                ),
            )
        ax.legend(handles=legend_elements)
        ax.set_title(self.title)
        ax.set_xlabel("Instance (sorted by distance to selected group)")
        ax.set_ylabel("Avg Distance to Selected Group")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig, ax
