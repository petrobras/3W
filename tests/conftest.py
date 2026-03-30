import pytest
import matplotlib
import pandas as pd
import numpy as np
from ThreeWToolkit.reports.report_generation import ReportGeneration
from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs


@pytest.fixture(autouse=True, scope="session")
def set_matplotlib_backend():
    matplotlib.use("Agg")


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


# Mock Dataset for preprocessing and feature extraction tests
class MockDataset(BaseDataset):
    """
    Mock dataset implementation for testing preprocessing and feature extraction.

    Stores a list of events (DatasetOutputs) and provides indexed access.
    """

    def __init__(self, events: list[DatasetOutputs]):
        """
        Initialize mock dataset with a list of events.

        Args:
            events: List of DatasetOutputs representing dataset events/samples
        """
        self.events = events

    def __len__(self) -> int:
        """Return the number of events in the dataset."""
        return len(self.events)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        """Return the event at the given index."""
        return self.events[idx]


def create_mock_dataset(
    num_events: int = 120,
    num_timesteps: int = 1024,
    num_sensors: int = 12,
    global_mean: float = 50.0,
    global_std: float = 15.0,
    nan_rate: float = 0.0,
    seed: int | None = 42,
) -> MockDataset:
    """
    Factory function to create parametrizable mock datasets.

    Args:
        num_events: Number of events/samples in the dataset
        num_timesteps: Number of time steps per event
        num_sensors: Number of sensor columns
        global_mean: Global mean for signal generation (across all events)
        global_std: Global standard deviation for signal generation
        nan_rate: Probability of NaN values (0.0 to 1.0)
        seed: Random seed for reproducibility (None for non-deterministic)

    Returns:
        MockDataset with generated events

    Example:
        >>> # Create dataset with 5 events, 20% NaN rate
        >>> dataset = create_mock_dataset(num_events=5, nan_rate=0.2)
        >>> len(dataset)
        5
        >>> # Create dataset with custom mean/std
        >>> dataset = create_mock_dataset(global_mean=100, global_std=20)
    """
    if seed is not None:
        np.random.seed(seed)

    events = []
    for event_id in range(num_events):
        # Generate signal data from normal distribution with specified mean/std
        signal_data = {}
        for sensor_idx in range(num_sensors):
            values = np.random.normal(
                loc=global_mean + event_id * 10,  # Slight shift per event
                scale=global_std,
                size=num_timesteps,
            )

            # Add NaNs at specified rate
            if nan_rate > 0:
                nan_mask = np.random.random(num_timesteps) < nan_rate
                values[nan_mask] = np.nan

            signal_data[f"sensor{sensor_idx + 1}"] = values

        # Create label series
        label_transition_idx = np.random.randint(num_timesteps // 3, 2 * num_timesteps // 3)
        label_value = np.zeros(num_timesteps, dtype=np.int64)
        label_value[:label_transition_idx] = 0  # Class 0 for first part
        label_value[label_transition_idx:] = event_id % 3 + 1  # Class 1 or 2 for second part, cycling through events
        labels = pd.Series(label_value, name="label", dtype="Int64")

        events.append(
            DatasetOutputs(
                signal=pd.DataFrame(signal_data),
                label=labels,
                metadata={"event_id": event_id, "generated": True},
            )
        )

    return MockDataset(events)


@pytest.fixture
def mock_dataset_factory():
    """
    Fixture that provides the factory function for creating mock datasets.

    Use this when you need to create custom datasets in tests.

    Example:
        def test_something(mock_dataset_factory):
            dataset = mock_dataset_factory(num_events=10, nan_rate=0.3)
            # ... test code
    """
    return create_mock_dataset
