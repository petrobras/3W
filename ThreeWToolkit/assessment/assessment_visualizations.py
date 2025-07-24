from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
from typing import cast
from ..core.base_assessment_visualization import (
    BaseAssessmentVisualization,
    BaseAssessmentVisualizationConfig,
)


class ConfigAssessmentVisualization(BaseAssessmentVisualizationConfig):
    pass


class AssessmentVisualization(BaseAssessmentVisualization):
    """Class for visualizing assessment results"""

    def __init__(self, config: ConfigAssessmentVisualization):
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
