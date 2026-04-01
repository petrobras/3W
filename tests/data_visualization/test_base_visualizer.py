import pytest
from abc import ABC
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ThreeWToolkit.data_visualization.base_visualizer import BaseVisualizer


class TestBaseVisualizer:
    def test_is_abstract_base_class(self):
        """Test that BaseVisualizer is an abstract base class."""
        assert issubclass(BaseVisualizer, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that BaseVisualizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVisualizer()

    def test_plot_method_is_abstract(self):
        """Test that the plot method is abstract."""
        # Check that plot is defined as abstract
        assert hasattr(BaseVisualizer, "plot")
        assert hasattr(BaseVisualizer.plot, "__isabstractmethod__")
        assert BaseVisualizer.plot.__isabstractmethod__ is True

    def test_concrete_implementation_must_implement_plot(self):
        """Test that concrete implementations must implement plot method."""
        
        # Define a class that doesn't implement plot
        class IncompleteVisualizer(BaseVisualizer):
            pass
        
        # Should not be able to instantiate
        with pytest.raises(TypeError):
            IncompleteVisualizer()

    def test_concrete_implementation_with_plot(self):
        """Test that a concrete implementation with plot method can be instantiated."""
        
        class ConcreteVisualizer(BaseVisualizer):
            def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes | None]:
                import matplotlib.pyplot as plt
                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = ax.get_figure()
                return fig, ax
        
        # Should be able to instantiate
        visualizer = ConcreteVisualizer()
        assert isinstance(visualizer, BaseVisualizer)

    def test_plot_signature(self):
        """Test that plot method has correct signature."""
        import inspect
        
        sig = inspect.signature(BaseVisualizer.plot)
        params = list(sig.parameters.keys())
        
        # Should have 'self' and 'ax' parameters
        assert "self" in params
        assert "ax" in params
        
        # ax should have default value of None
        assert sig.parameters["ax"].default is None

    def test_plot_return_annotation(self):
        """Test that plot method has correct return type annotation."""
        import inspect
        
        sig = inspect.signature(BaseVisualizer.plot)
        return_annotation = sig.return_annotation
        
        # Should return tuple[Figure, Axes | None]
        assert return_annotation != inspect.Signature.empty

    def test_concrete_implementation_works_correctly(self):
        """Test that a proper concrete implementation works as expected."""
        import matplotlib.pyplot as plt
        
        class WorkingVisualizer(BaseVisualizer):
            def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes | None]:
                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = ax.get_figure()
                ax.plot([1, 2, 3], [1, 2, 3])
                return fig, ax
        
        visualizer = WorkingVisualizer()
        
        # Test without providing axes
        fig1, ax1 = visualizer.plot()
        assert isinstance(fig1, Figure)
        assert isinstance(ax1, Axes)
        plt.close(fig1)
        
        # Test with providing axes
        fig2, ax2 = plt.subplots()
        returned_fig, returned_ax = visualizer.plot(ax=ax2)
        assert returned_fig is fig2
        assert returned_ax is ax2
        plt.close(fig2)

    def test_docstring_exists(self):
        """Test that the plot method has a docstring."""
        assert BaseVisualizer.plot.__doc__ is not None
        assert len(BaseVisualizer.plot.__doc__) > 0

    def test_class_docstring_exists(self):
        """Test that BaseVisualizer has a class docstring."""
        assert BaseVisualizer.__doc__ is not None
        assert "visualization" in BaseVisualizer.__doc__.lower()
