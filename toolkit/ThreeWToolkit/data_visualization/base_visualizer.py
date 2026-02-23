from abc import ABC, abstractmethod
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class BaseVisualizer(ABC):
    """
    Base class for all visualization objects.

    The contract is:
        vis = SomePlot(...)
        fig, ax = vis.plot(ax=None)

    If `ax` is None, the implementation must create a new Figure/Axes.
    If `ax` is provided, it must draw on that Axes and return (ax.get_figure(), ax).
    """

    @abstractmethod
    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes | None]:
        """
        Render the visualization.

        Args:
            ax: Optional matplotlib Axes to draw on. If None, the implementation
                must create a new Figure and Axes. If provided, the visualization
                must be drawn on this Axes.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes where the visualization is rendered.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
