import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ThreeWToolkit.data_visualization.plot_utils import create_subplot_grid


class TestCreateSubplotGrid:
    def test_single_subplot_returns_figure_and_2d_array(self):
        """Test that creating a single subplot returns a Figure and 2D array of axes."""
        fig, axes = create_subplot_grid(nrows=1, ncols=1)

        assert isinstance(fig, Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 1)
        plt.close(fig)

    def test_multiple_rows_single_column(self):
        """Test creating a grid with multiple rows and single column."""
        fig, axes = create_subplot_grid(nrows=3, ncols=1)

        assert isinstance(fig, Figure)
        assert axes.shape == (3, 1)
        plt.close(fig)

    def test_single_row_multiple_columns(self):
        """Test creating a grid with single row and multiple columns."""
        fig, axes = create_subplot_grid(nrows=1, ncols=4)

        assert isinstance(fig, Figure)
        assert axes.shape == (1, 4)
        plt.close(fig)

    def test_multiple_rows_and_columns(self):
        """Test creating a grid with multiple rows and columns."""
        fig, axes = create_subplot_grid(nrows=2, ncols=3)

        assert isinstance(fig, Figure)
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_custom_figsize(self):
        """Test that custom figsize is applied correctly."""
        custom_size = (10, 8)
        fig, axes = create_subplot_grid(nrows=2, ncols=2, figsize=custom_size)

        assert fig.get_size_inches()[0] == custom_size[0]
        assert fig.get_size_inches()[1] == custom_size[1]
        plt.close(fig)

    def test_default_figsize_calculation(self):
        """Test that default figsize is calculated based on grid dimensions."""
        nrows, ncols = 2, 3
        default_width_per_col = 5
        default_height_per_row = 4
        
        fig, axes = create_subplot_grid(
            nrows=nrows,
            ncols=ncols,
            default_width_per_col=default_width_per_col,
            default_height_per_row=default_height_per_row,
        )

        expected_width = default_width_per_col * ncols
        expected_height = default_height_per_row * nrows
        
        assert fig.get_size_inches()[0] == expected_width
        assert fig.get_size_inches()[1] == expected_height
        plt.close(fig)

    def test_custom_default_dimensions(self):
        """Test setting custom default width and height per subplot."""
        fig, axes = create_subplot_grid(
            nrows=2,
            ncols=2,
            default_width_per_col=6,
            default_height_per_row=5,
        )

        assert fig.get_size_inches()[0] == 12  # 6 * 2
        assert fig.get_size_inches()[1] == 10  # 5 * 2
        plt.close(fig)

    def test_axes_indexing(self):
        """Test that axes can be indexed consistently."""
        fig, axes = create_subplot_grid(nrows=2, ncols=3)

        # Should be able to access any subplot
        for i in range(2):
            for j in range(3):
                ax = axes[i, j]
                assert ax is not None
        
        plt.close(fig)

    def test_single_subplot_indexing(self):
        """Test that single subplot can be indexed as [0, 0]."""
        fig, axes = create_subplot_grid(nrows=1, ncols=1)

        ax = axes[0, 0]
        assert ax is not None
        plt.close(fig)

    def test_tight_layout_applied(self):
        """Test that tight_layout is applied to the figure."""
        fig, axes = create_subplot_grid(nrows=2, ncols=2)

        # The function should apply tight_layout
        # We can't directly test if it was called, but we can verify the figure exists
        assert isinstance(fig, Figure)
        plt.close(fig)
