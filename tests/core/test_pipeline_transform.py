"""Tests for BasePipeline, BaseTransform, and PredictionStrategy."""

import pytest
import pandas as pd
import numpy as np

from ThreeWToolkit.core import (
    BasePipeline,
    BasePipelineConfig,
    PredictionStrategy,
    DatasetOutputs,
    TaskTypeEnum,
    BaseTransform,
    BaseTransformConfig,
)


class TestBasePipeline:
    """Test BasePipeline class."""

    def test_pipeline_abstract(self):
        """Test that BasePipeline is abstract."""

        class ConcretePipeline(BasePipeline):
            pass

        pipeline = ConcretePipeline()

        with pytest.raises(NotImplementedError):
            pipeline.run()

    def test_pipeline_implementation(self):
        """Test a complete pipeline implementation."""

        class MyPipeline(BasePipeline):
            def __init__(self, config):
                self.config = config
                self.executed = False

            def run(self):
                self.executed = True
                return "Pipeline completed"

        class MyPipelineConfig(BasePipelineConfig):
            name: str = "test"
            _target: type = MyPipeline

        config = MyPipelineConfig(name="my_pipeline")
        pipeline = config.build()
        result = pipeline.run()

        assert pipeline.executed is True
        assert result == "Pipeline completed"


class TestBasePipelineConfig:
    """Test BasePipelineConfig."""

    def test_config_stores_target(self):
        """Test that config stores target class."""

        class MyPipeline(BasePipeline):
            def run(self):
                pass

        class MyConfig(BasePipelineConfig):
            _target: type = MyPipeline

        config = MyConfig()
        assert config._target == MyPipeline

    def test_config_build_returns_pipeline(self):
        """Test that build returns pipeline instance."""

        class MyPipeline(BasePipeline):
            def __init__(self, config):
                self.config = config

            def run(self):
                return self.config.steps

        class MyConfig(BasePipelineConfig):
            steps: int = 5
            _target: type = MyPipeline

        config = MyConfig(steps=10)
        pipeline = config.build()

        assert isinstance(pipeline, MyPipeline)
        assert pipeline.run() == 10


class TestBaseTransform:
    """Test BaseTransform class."""

    def test_transform_abstract_transform_event(self):
        """Test that transform_event must be implemented."""

        class IncompleteTransform(BaseTransform):
            def fit(self, dataset):
                pass

        class IncompleteConfig(BaseTransformConfig):
            _target: type = IncompleteTransform

        transform = IncompleteConfig().build()

        signal = pd.DataFrame({"s": [1.0]})
        data = DatasetOutputs(signal=signal, label=None)

        with pytest.raises(NotImplementedError):
            transform.transform_event(data)

    def test_transform_complete_implementation(self, mock_dataset_factory):
        """Test complete transform implementation."""

        class ScaleTransform(BaseTransform):
            def __init__(self, config):
                super().__init__(config)
                self.scale = 1.0

            def fit(self, dataset):
                # Compute max value across dataset
                max_val = 0.0
                for event in dataset:
                    event_max = event.signal.max().max()
                    if event_max > max_val:
                        max_val = event_max
                self.scale = max_val if max_val > 0 else 1.0

            def transform_event(self, data: DatasetOutputs) -> DatasetOutputs:
                new_signal = data.signal / self.scale
                return DatasetOutputs(
                    signal=new_signal,
                    label=data.label,
                    metadata=data.metadata,
                )

        class ScaleConfig(BaseTransformConfig):
            _target: type = ScaleTransform

        transform = ScaleConfig().build()
        dataset = mock_dataset_factory(num_events=5, global_mean=50.0)

        transform.fit(dataset)

        # Transform a single event
        event = dataset[0]
        result = transform.transform_event(event)

        # Values should be scaled down
        assert result.signal.max().max() <= 1.0 or np.isclose(
            result.signal.max().max(), 1.0, atol=0.1
        )


class TestPredictionStrategy:
    """Test PredictionStrategy abstract class."""

    def test_prediction_strategy_abstract(self):
        """Test that PredictionStrategy is abstract."""

        class IncompleteStrategy(PredictionStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()

    def test_prediction_strategy_implementation(self):
        """Test a complete prediction strategy implementation."""

        class SimpleStrategy(PredictionStrategy):
            def predict(self, model, task=None, **kwargs):
                X = kwargs.get("X")
                if X is not None:
                    return np.ones(len(X))
                return np.array([0])

        strategy = SimpleStrategy()

        # Default requires_dataloader is False
        assert strategy.requires_dataloader is False

        # Test predict
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = strategy.predict(None, X=X)

        assert len(result) == 3
        assert all(r == 1 for r in result)

    def test_prediction_strategy_with_dataloader_requirement(self):
        """Test strategy that requires dataloader."""

        class DataloaderStrategy(PredictionStrategy):
            def predict(self, model, task=None, **kwargs):
                dataloader = kwargs.get("dataloader")
                return np.array([1, 2, 3])

            @property
            def requires_dataloader(self):
                return True

        strategy = DataloaderStrategy()

        assert strategy.requires_dataloader is True

    def test_prediction_strategy_with_task_type(self):
        """Test strategy using task type."""

        class TaskAwareStrategy(PredictionStrategy):
            def predict(self, model, task=None, **kwargs):
                if task == TaskTypeEnum.CLASSIFICATION:
                    return np.array([0, 1, 0, 1])
                elif task == TaskTypeEnum.REGRESSION:
                    return np.array([0.1, 0.5, 0.3, 0.8])
                return np.array([])

        strategy = TaskAwareStrategy()

        class_result = strategy.predict(None, task=TaskTypeEnum.CLASSIFICATION)
        reg_result = strategy.predict(None, task=TaskTypeEnum.REGRESSION)

        assert all(isinstance(v, (int, np.integer)) for v in class_result)
        assert all(isinstance(v, (float, np.floating)) for v in reg_result)
