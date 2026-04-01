"""Tests for BasePreprocessing class."""

import pytest
import pandas as pd

from ThreeWToolkit.core import (
    BasePreprocessing,
    BasePreprocessingConfig,
    DatasetOutputs,
)


class TestBasePreprocessingImplementation:
    """Test BasePreprocessing implementation requirements."""

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        class IncompletePreprocessing(BasePreprocessing):
            pass

        class DummyConfig(BasePreprocessingConfig):
            target_: type = IncompletePreprocessing

        config = DummyConfig()

        with pytest.raises(TypeError):
            config.build()

    def test_complete_implementation(self):
        """Test a complete implementation of BasePreprocessing."""

        class SimplePreprocessing(BasePreprocessing):
            def transform(self, data: DatasetOutputs) -> DatasetOutputs:
                # Simple transform: multiply signal by 2
                new_signal = data.signal * 2
                return DatasetOutputs(
                    signal=new_signal,
                    label=data.label,
                    metadata=data.metadata,
                )

        class SimpleConfig(BasePreprocessingConfig):
            target_: type = SimplePreprocessing

        config = SimpleConfig()
        preprocessing = config.build()

        signal = pd.DataFrame({"sensor_0": [1.0, 2.0]})
        label = pd.Series([0, 1])
        data = DatasetOutputs(signal=signal, label=label)

        result = preprocessing.transform(data)

        assert result.signal["sensor_0"].iloc[0] == 2.0
        assert result.signal["sensor_0"].iloc[1] == 4.0


class TestBasePreprocessingFit:
    """Test BasePreprocessing fit method."""

    def test_default_fit_does_nothing(self, mock_dataset_factory):
        """Test that default fit method does nothing."""

        class SimplePreprocessing(BasePreprocessing):
            def __init__(self, config):
                super().__init__(config)
                self.fit_called = False

            def transform(self, data: DatasetOutputs) -> DatasetOutputs:
                return data

        class SimpleConfig(BasePreprocessingConfig):
            target_: type = SimplePreprocessing

        preprocessing = SimpleConfig().build()
        dataset = mock_dataset_factory(num_events=5)

        # Should not raise
        preprocessing.fit(dataset)

    def test_custom_fit_implementation(self, mock_dataset_factory):
        """Test custom fit implementation."""

        class StatefulPreprocessing(BasePreprocessing):
            def __init__(self, config):
                super().__init__(config)
                self.mean = None

            def fit(self, dataset):
                # Compute global mean from dataset
                all_values = []
                for event in dataset:
                    all_values.extend(event.signal.values.flatten())
                self.mean = sum(all_values) / len(all_values)

            def transform(self, data: DatasetOutputs) -> DatasetOutputs:
                if self.mean is None:
                    raise RuntimeError("Must fit before transform")
                new_signal = data.signal - self.mean
                return DatasetOutputs(
                    signal=new_signal,
                    label=data.label,
                    metadata=data.metadata,
                )

        class StatefulConfig(BasePreprocessingConfig):
            target_: type = StatefulPreprocessing

        preprocessing = StatefulConfig().build()
        dataset = mock_dataset_factory(num_events=5, global_mean=50.0)

        preprocessing.fit(dataset)

        assert preprocessing.mean is not None
        # Mean should be close to 50.0
        assert 40.0 < preprocessing.mean < 60.0


class TestBasePreprocessingConfig:
    """Test BasePreprocessingConfig."""

    def test_config_stores_target(self):
        """Test that config stores target class."""

        class MyPreprocessing(BasePreprocessing):
            def transform(self, data):
                return data

        class MyConfig(BasePreprocessingConfig):
            target_: type = MyPreprocessing

        config = MyConfig()
        assert config.target_ == MyPreprocessing

    def test_config_accessible_from_preprocessing(self):
        """Test that config is accessible from preprocessing instance."""

        class MyPreprocessing(BasePreprocessing):
            def transform(self, data):
                return data

        class MyConfig(BasePreprocessingConfig):
            param_a: int = 5
            param_b: str = "test"
            target_: type = MyPreprocessing

        config = MyConfig(param_a=10, param_b="custom")
        preprocessing = config.build()

        assert preprocessing.config.param_a == 10
        assert preprocessing.config.param_b == "custom"
