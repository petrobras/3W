import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from typing import cast
from ..core.base_assessment_visualization import (
    AssessmentVisualizationConfig,
    BaseAssessmentVisualization,
)


class AssessmentVisualization(BaseAssessmentVisualization):
    """Class for visualizing assessment results"""

    def __init__(self, config: AssessmentVisualizationConfig):
        self.config = config
        self.class_names = config.class_names

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray | pd.Series | list,
        y_pred: np.ndarray | pd.Series | list,
        title: str = "Confusion Matrix",
        ax: Axes | None = None,
        normalize: bool = True,
        figsize: tuple[int, int] = (12, 8),
        fontsize: int = 10,
    ) -> Figure:
        """
        Plots a confusion matrix for the given true and predicted labels.

        Args:
            y_true (np.ndarray | pd.Series | list): Array-like of true labels
            y_pred (np.ndarray | pd.Series | list): Array-like of predicted labels
            title (str): Title for the plot
            ax (Axes): Matplotlib Axes to plot into. Creates new if None.
            normalize (bool): Whether to normalize the confusion matrix
            figsize (tuple[int, int]): Size of the figure
            fontsize (int): Base font size for labels

        Returns:
            fig (plt.Figure): confusion matrix plot as a matplotlib Figure

        Usage:
            confusion_matrix = plotter.plot_confusion_matrix(y_true=y_true_list, y_pred=y_pred_list)
        """

        if not isinstance(y_true, (pd.Series, np.ndarray, list)):
            raise TypeError("y_true must be a pandas Series, numpy array, or list.")

        if len(y_true) != len(y_pred):
            raise ValueError("length of y_true and y_pred must be the same.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.figure)

        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            data=cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            square=True,
        )

        # Set labels
        ax.set_xlabel("Predicted", fontsize=fontsize)
        ax.set_ylabel("True", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize + 2)

        # Set class names if provided in config
        if self.class_names is not None:
            if len(self.class_names) != cm.shape[0]:
                raise ValueError(
                    f"Number of class names ({len(self.class_names)}) "
                    f"does not match confusion matrix size ({cm.shape[0]})"
                )

            tick_marks = np.arange(len(self.class_names)) + 0.5
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(self.class_names, rotation=45, ha="right")
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(self.class_names, rotation=0)

        return fig

    def feature_visualization(
        self,
        feature_importances: np.ndarray | list,
        feature_names: np.ndarray | list,
        title: str = "Feature Importances",
        ax: Axes | None = None,
        figsize: tuple[int, int] = (12, 8),
        top_n: int | None = None,
        color: str = "skyblue",
    ) -> Figure:
        """
        Plots feature importances from a tree-based model.

        Args:
            feature_importances (np.ndarray | list): Array-like of feature importance values
            feature_names (np.ndarray | list): Array-like of feature names
            title (str): Title for the plot
            ax (Axes): Matplotlib Axes to plot into. Creates new if None.
            figsize (tuple[int, int]): Size of the figure
            top_n (int | None): Number of top features to display. If None, shows all.
            color (str): Color for the bars

        Returns:
            fig (plt.Figure): feature importance plot as a matplotlib Figure

        Usage:
            feature_importance_plot = plotter.feature_visualization(
                feature_importances=model.feature_importances_,
                feature_names=feature_names,
                top_n=20
            )
        """
        if len(feature_importances) != len(feature_names):
            raise ValueError(
                "Length of feature_importances and feature_names must match."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = cast(Figure, ax.figure)

        feature_importances = np.asarray(feature_importances)
        feature_names = np.asarray(feature_names)

        # Sort features by importance
        indices = np.argsort(feature_importances)
        if top_n is not None:
            indices = indices[-top_n:]

        ax.barh(
            range(len(indices)),
            feature_importances[indices],
            color=color,
            align="center",
        )

        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(feature_names[indices])
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()

        return fig
