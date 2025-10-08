import pytest
import pandas as pd
import numpy as np
from ThreeWToolkit.reports.report_generation import ReportGeneration


# A helper class to create an object from a dictionary
class MockConfig:
    def __init__(self, config_dict: dict):
        """
        Populates the instance's __dict__ from the passed dictionary.
        This makes attributes like self.parameter_alpha directly accessible.
        """
        self.__dict__.update(config_dict)

    def __iter__(self):
        """
        Makes the object iterable for loops like `for key, val in obj:`.
        It yields key-value pairs, mimicking the behavior of `dict.items()`.
        """
        yield from self.__dict__.items()


class MockModel:
    def __init__(self):
        config_dict = {
            "parameter_alpha": 0.1,
            "some_list": [1, 2, 3],
            "learning_rate": "auto",
        }
        # The 'config' attribute isan instance of our special MockConfig class
        self.config = MockConfig(config_dict)


@pytest.fixture
def mock_model():
    """Provides a mock model instance for tests."""
    return MockModel()


@pytest.fixture
def sample_data():
    """Provides sample pandas Series for testing."""
    return {
        "X_train": pd.Series(np.random.rand(50), name="feature_1"),
        "y_train": pd.Series(np.random.randint(0, 2, 50), name="target"),
        "X_test": pd.Series(np.random.rand(20), name="feature_1"),
        "y_test": pd.Series(np.random.randint(0, 2, 20), name="target"),
        "predictions": pd.Series(np.random.randint(0, 2, 20), name="preds"),
    }


@pytest.fixture
def report_generator_instance(tmp_path, mock_model, sample_data):
    """Creates a ReportGeneration instance using a temporary directory."""
    plot_config = {
        "PlotSeries": {
            "series": sample_data["y_test"],
            "title": "Test Series Plot",
            "xlabel": "Index",
            "ylabel": "Values",
        },
        "PlotMultipleSeries": {
            "series_list": [sample_data["y_test"], sample_data["predictions"]],
            "title": "Multiple Series Plot",
            "xlabel": "Index",
            "ylabel": "Values",
            "labels": ["True Values", "Predictions"],
        },
    }

    instance = ReportGeneration(
        model=mock_model,
        **sample_data,
        calculated_metrics={"Accuracy": 0.95, "F1 Score": 0.92},
        plot_config=plot_config,
        title="Test_Report",
        latex_dir=tmp_path / "latex",
        reports_dir=tmp_path / "reports",
        export_report_after_generate=False,  # Keep this False for most tests
    )
    return instance


@pytest.fixture
def sample_results_dict():
    """Provides a valid sample 'results' dictionary for CSV export."""
    return {
        "X_test": pd.DataFrame(
            {
                "feature_A": [1, 2, 3, 4, 5],
                "feature_B": [10, 20, 30, 40, 50],
            }
        ),
        "true_values": [1, 0, 1, 0, 1],
        "predictions": [1, 1, 1, 0, 0],
        "model_name": "MyAwesomeModel",
        "metrics": {"Accuracy": 0.6, "F1 Score": 0.667},
    }
