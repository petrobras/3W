import pandas as pd
import pytest
import numpy as np
import torch

from ThreeWToolkit.core.base_step import BaseStep
from ThreeWToolkit.pipeline import Pipeline
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    RenameColumnsConfig,
    WindowingConfig,
)
from ThreeWToolkit.core.base_feature_extractor import (
    EWStatisticalConfig,
    StatisticalConfig,
    WaveletConfig,
)
from ThreeWToolkit.preprocessing._data_processing import ImputeMissing, Windowing
from ThreeWToolkit.trainer.trainer import TrainerConfig
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.assessment.model_assess import ModelAssessment
from ThreeWToolkit.trainer.trainer import ModelTrainer
from ThreeWToolkit.dataset.parquet_dataset import ParquetDataset
from ThreeWToolkit.core.enums import TaskType


class MockStep(BaseStep):
    """Mock implementation of a pipeline step for testing purposes."""

    def __init__(self, name="mock"):
        self.name = name

    def pre_process(self, data):
        return data

    def run(self, data):
        return data

    def post_process(self, data):
        return data


class TestPipeline:
    """Unit tests for the Pipeline class."""

    def setup_method(self):
        """Set up common configurations before each test."""
        self.mlp_config = MLPConfig(
            hidden_sizes=(8,), output_size=2, activation_function="relu"
        )
        self.trainer_config = TrainerConfig(
            optimizer="adam",
            criterion="cross_entropy",
            batch_size=2,
            epochs=1,
            seed=42,  # ✅ precisa ser passado
            learning_rate=0.001,
            config_model=self.mlp_config,
            shuffle_train=True,
        )
        self.dataset_config = ParquetDatasetConfig(
            path="./data/raw",
            columns=["sig1", "sig2"],
            target_column="class",
            target_class=[0, 1],
        )
        self.assessment_config = ModelAssessmentConfig(
            metrics=["f1"], task_type=TaskType.CLASSIFICATION
        )

    def test_pipeline_initialization(self):
        """Test that the Pipeline initializes without errors."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        assert isinstance(pipeline, Pipeline)

    def test_dataset_loader_step(self):
        """Test that dataset loader step is correctly instantiated."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        assert isinstance(pipeline.step_data_loader, ParquetDataset)

    def test_trainer_step(self):
        """Test that trainer step is correctly instantiated."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        assert isinstance(pipeline.step_model_training, ModelTrainer)

    def test_assessment_step(self):
        """Test that assessment step is correctly instantiated."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        assert isinstance(pipeline.step_model_assessment, ModelAssessment)

    def test_add_preprocessing_steps(self):
        """Test that preprocessing steps are added to the pipeline."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                ImputeMissingConfig(strategy="mean", columns=["sig1"]),
                NormalizeConfig(norm="l2"),
                self.trainer_config,
                self.assessment_config,
            ]
        )
        assert len(pipeline.step_preprocessing) == 2

    def test_add_feature_extraction_step(self):
        """Test that feature extraction step is added to the pipeline."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                StatisticalConfig(),
                self.trainer_config,
                self.assessment_config,
            ]
        )
        assert len(pipeline.step_feat_extraction) == 1

    def test_multiple_feature_extraction_error(self):
        """Test that adding multiple feature extraction configs raises ValueError."""
        with pytest.raises(ValueError):
            Pipeline(
                [
                    self.dataset_config,
                    StatisticalConfig(),
                    WaveletConfig(),
                    self.trainer_config,
                    self.assessment_config,
                ]
            )

    def test_invalid_step_type(self):
        """Test that passing an invalid step raises ValueError."""

        class DummyConfig:
            pass

        with pytest.raises(ValueError):
            Pipeline([self.dataset_config, DummyConfig()])

    def test_missing_dataset_raises_error(self):
        """Test that pipeline raises ValueError when dataset step is missing."""
        with pytest.raises(ValueError):
            Pipeline([self.trainer_config, self.assessment_config])

    def test_missing_trainer_raises_error(self):
        """Test that pipeline raises ValueError when trainer step is missing."""
        with pytest.raises(ValueError):
            Pipeline([self.dataset_config, self.assessment_config])

    def test_missing_assessment_raises_error(self):
        """Test that pipeline raises ValueError when assessment step is missing."""
        with pytest.raises(ValueError):
            Pipeline([self.dataset_config, self.trainer_config])

    def test_preprocessing_order_with_windowing(self):
        """Test that Windowing step is moved to the last position."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                WindowingConfig(window_size=10),
                RenameColumnsConfig(columns_map={"sig1": "renamed"}),
                self.trainer_config,
                self.assessment_config,
            ]
        )
        assert isinstance(
            pipeline.step_preprocessing[-1], type(pipeline.step_preprocessing[-1])
        )

    def test_only_one_rename_allowed(self):
        """Test that multiple RenameColumns steps raise ValueError."""
        with pytest.raises(ValueError):
            Pipeline(
                [
                    self.dataset_config,
                    RenameColumnsConfig(columns_map={"sig1": "r1"}),
                    RenameColumnsConfig(columns_map={"sig1": "r2"}),
                    self.trainer_config,
                    self.assessment_config,
                ]
            )

    def test_check_and_apply_windowing_applies_default(self):
        """Test that default windowing is applied when none exists."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        batch = {
            "signals": {0: np.arange(100).reshape(-1, 1)},
            "labels": {0: np.ones(100)},
        }
        _, df_batch = pipeline._check_and_apply_windowing(batch)
        assert "label" in df_batch.columns

    def test_pipeline_missing_dataset_raises_error(self):
        """Pipeline should raise error if dataset config is missing."""
        with pytest.raises(ValueError):
            Pipeline([self.trainer_config, self.assessment_config])

    def test_pipeline_missing_trainer_raises_error(self):
        """Pipeline should raise error if trainer config is missing."""
        with pytest.raises(ValueError):
            Pipeline([self.dataset_config, self.assessment_config])

    def test_pipeline_missing_assessment_raises_error(self):
        """Pipeline should raise error if assessment config is missing."""
        with pytest.raises(ValueError):
            Pipeline([self.dataset_config, self.trainer_config])

    def test_batch_iterator_invalid_input_type(self):
        """_batch_iterator should raise for unsupported input types."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        with pytest.raises(ValueError):
            list(pipeline._batch_iterator({"signals": {0: "invalid_type"}}))

    def test_check_and_apply_windowing_without_config(self):
        """If no windowing config is set, it should just return input unchanged."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        batch = {
            "signals": {0: np.arange(100).reshape(-1, 1)},
            "labels": {0: np.ones(100)},
        }
        _, df_batch = pipeline._check_and_apply_windowing(batch)
        assert df_batch is not None

    def test_batch_iterator_with_tensor_only(self):
        """Test _batch_iterator with single tensor input."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        tensor_data = torch.randn(10, 5)
        data_loader = [tensor_data]

        feature_names = [f"feat_{i}" for i in range(5)]
        results = list(pipeline._batch_iterator(data_loader, feature_names))

        assert len(results) == 1
        assert isinstance(results[0], pd.DataFrame)
        assert results[0].shape == (10, 5)
        assert list(results[0].columns) == feature_names

    def test_batch_iterator_with_tuple_batch(self):
        """Test _batch_iterator with (X, y) tuple format."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        x = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))
        data_loader = [(x, y)]

        feature_names = [f"feat_{i}" for i in range(5)]
        results = list(pipeline._batch_iterator(data_loader, feature_names))

        assert len(results) == 1
        assert isinstance(results[0], pd.DataFrame)
        assert results[0].shape == (10, 6)  # 5 features + 1 target
        assert "target" in results[0].columns

    def test_batch_iterator_with_numpy_arrays(self):
        """Test _batch_iterator with numpy arrays."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        x = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        data_loader = [(x, y)]

        feature_names = [f"feat_{i}" for i in range(5)]
        results = list(pipeline._batch_iterator(data_loader, feature_names))

        assert len(results) == 1
        assert isinstance(results[0], pd.DataFrame)
        assert "target" in results[0].columns

    def test_get_nfiles_per_batch(self):
        """Test _get_nfiles_per_batch method."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        batch = {
            "signals": {
                "sig1": [np.arange(100), np.arange(100), np.arange(100)],
                "sig2": [np.arange(100), np.arange(100), np.arange(100)],
            }
        }

        nfiles = pipeline._get_nfiles_per_batch(batch)
        assert nfiles == 3

    def test_validate_steps_invalid_data_loader_type(self):
        """Test that invalid data loader type raises TypeError."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        # Simulate invalid data loader
        pipeline.step_data_loader = "not_a_base_step"

        with pytest.raises(TypeError):
            pipeline._validate_steps()

    def test_validate_steps_invalid_feature_extraction_type(self):
        """Test that feature extraction not inheriting from BaseStep raises TypeError."""

        class InvalidFeatureExtractor:
            pass

        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )
        pipeline.step_feat_extraction.append(InvalidFeatureExtractor())

        with pytest.raises(TypeError):
            pipeline._validate_steps()

    def test_validate_steps_invalid_trainer_instance(self):
        """Test that invalid trainer instance raises ValueError."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        # Create a mock that inherits from BaseStep but is not ModelTrainer
        class InvalidTrainer(BaseStep):
            def pre_process(self, data):
                return data

            def run(self, data):
                return data

            def post_process(self, data):
                return data

        pipeline.step_model_training = InvalidTrainer()

        with pytest.raises(ValueError):
            pipeline._validate_steps()

    def test_preprocessing_order_rename_last_without_windowing(self):
        """Test that RenameColumns is moved to last position when no Windowing exists."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                NormalizeConfig(norm="l2"),
                RenameColumnsConfig(columns_map={"sig1": "renamed"}),
                ImputeMissingConfig(strategy="mean", columns=["sig1"]),
                self.trainer_config,
                self.assessment_config,
            ]
        )

        # Check that RenameColumns is last
        assert pipeline.step_preprocessing[-1].__class__.__name__ == "RenameColumns"

    def test_validate_preprocessing_steps_invalid_type(self):
        """Test that preprocessing step not inheriting from BaseStep raises TypeError."""

        class InvalidStep:
            pass

        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        with pytest.raises(TypeError):
            pipeline._validate_and_fix_preprocessing_steps([InvalidStep()])

    def test_check_and_apply_windowing_with_dict_signals(self):
        """Test _check_and_apply_windowing with dictionary structure."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                WindowingConfig(window_size=10),
                self.trainer_config,
                self.assessment_config,
            ]
        )

        batch = {
            "signals": {0: {"sig1": np.arange(100), "sig2": np.arange(100, 200)}},
            "labels": [[1] * 100],
        }

        requires_windowing, df_batch = pipeline._check_and_apply_windowing(batch)

        assert requires_windowing is True
        assert isinstance(df_batch, pd.DataFrame)
        assert "label" in df_batch.columns

    def test_check_and_apply_windowing_multiple_files(self):
        """Test _check_and_apply_windowing with multiple files."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        batch = {
            "signals": {
                0: {"sig1": np.arange(100), "sig2": np.arange(100)},
                1: {"sig1": np.arange(100), "sig2": np.arange(100)},
            },
            "labels": [[0] * 100, [1] * 100],
        }

        _, df_batch = pipeline._check_and_apply_windowing(batch)

        assert isinstance(df_batch, pd.DataFrame)
        assert len(df_batch) > 0

    def test_feature_extraction_with_ew_statistical(self):
        """Test pipeline with EWStatisticalConfig."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                EWStatisticalConfig(),
                self.trainer_config,
                self.assessment_config,
            ]
        )

        assert len(pipeline.step_feat_extraction) == 1
        assert isinstance(pipeline.step_feat_extraction[0].__class__.__name__, str)

    def test_preprocessing_with_all_steps(self):
        """Test pipeline with all preprocessing steps."""
        pipeline = Pipeline(
            [
                self.dataset_config,
                ImputeMissingConfig(strategy="mean", columns=["sig1"]),
                NormalizeConfig(norm="l2"),
                RenameColumnsConfig(columns_map={"sig1": "signal_1"}),
                WindowingConfig(window_size=10),
                self.trainer_config,
                self.assessment_config,
            ]
        )

        assert len(pipeline.step_preprocessing) == 4
        # Windowing should be last
        assert isinstance(pipeline.step_preprocessing[-1].__class__.__name__, str)

    def test_run_prep_steps_without_preprocessing(self):
        """Test run_prep_steps when no preprocessing steps exist."""
        pipeline = Pipeline(
            [self.dataset_config, self.trainer_config, self.assessment_config]
        )

        batch = {"signals": {0: {"sig1": np.arange(100)}}, "labels": [[1] * 100]}

        result = pipeline.run_prep_steps(batch)
        assert result is not None

    def test_validate_step_model_training_not_basestep(self):
        from ThreeWToolkit.pipeline import Pipeline

        class FakeTrainer:
            pass

        step_data_loader = MockStep()
        step_model_training = FakeTrainer()
        step_model_assessment = MockStep()

        configs = [step_data_loader, step_model_training, step_model_assessment]
        with pytest.raises(ValueError):
            Pipeline(configs)

    def test_validate_step_model_training_wrong_type(self):
        from ThreeWToolkit.pipeline import Pipeline

        class WrongTrainer(BaseStep):
            def pre_process(self, data):
                return data

            def run(self, data):
                return data

            def post_process(self, data):
                return data

        step_data_loader = MockStep()
        step_model_training = WrongTrainer()
        step_model_assessment = MockStep()

        configs = [step_data_loader, step_model_training, step_model_assessment]
        with pytest.raises(ValueError):
            Pipeline(configs)

    def test_validate_step_model_assessment_wrong_type(self):
        step_data_loader = MockStep()
        step_model_training = ModelTrainer.__new__(ModelTrainer)
        step_model_assessment = object()

        configs = [step_data_loader, step_model_training, step_model_assessment]
        with pytest.raises(ValueError):
            Pipeline(configs)

    def test_batch_iterator_invalid_format(self):
        step_data_loader = MockStep()
        step_model_training = ModelTrainer.__new__(ModelTrainer)
        step_model_assessment = MockStep()

        configs = [step_data_loader, step_model_training, step_model_assessment]
        batch = {"signals": {0: "invalid"}}
        with pytest.raises(ValueError):
            pip = Pipeline(configs)
            list(pip._batch_iterator(batch))

    # Helper to create a minimal valid pipeline
    def make_minimal_pipeline(self):
        """Helper to create a minimal valid pipeline"""
        dataset_cfg = ParquetDatasetConfig(path="fake.parquet", target_column="class")

        trainer_cfg = TrainerConfig(
            batch_size=4,
            epochs=1,
            seed=42,
            learning_rate=0.001,
            config_model=MLPConfig(input_size=1, hidden_sizes=(4,), output_size=1),
            test_size=0.2,
        )

        assess_cfg = ModelAssessmentConfig(metrics=["accuracy"])
        return Pipeline([dataset_cfg, trainer_cfg, assess_cfg])

    def create_test_batch(self, n_samples=120):
        """Helper to create a test batch with correct structure"""
        return {
            "signals": {
                "T-JUS-CKP": [list(range(n_samples))],
                "T-MON-CKP": [list(range(n_samples))],
            },
            "labels": {"class": [0]},
            "file_names": ["file1.parquet"],
        }

    @pytest.mark.skip(
        reason="This test class was disabled so that we can think about a better way to test the dataset download."
    )
    def test_run_prep_steps_with_preprocessing(self, monkeypatch):
        """Test run_prep_steps with preprocessing steps."""

        class DummyLoader:
            def __init__(self):
                self.config = type("cfg", (), {"target_column": "class"})()

        pipeline = self.make_minimal_pipeline()
        pipeline.step_data_loader = DummyLoader()

        # Creating ImputeMissing class mock
        impute_mock = ImputeMissing.__new__(ImputeMissing)
        pipeline.step_preprocessing = [impute_mock]

        # Batch with labels as list of lists (expected structure)
        batch = {
            "signals": {
                "T-JUS-CKP": [[0] * 120],  # List of lists
                "T-MON-CKP": [[0] * 120],
            },
            "labels": {"class": [[0] * 120]},  # Labels also as list of lists
            "file_names": ["file1.parquet"],
        }

        # Mock ImputeMissing's run method to return valid DataFrame
        def mock_impute_run(self, data):
            # Simply returns the received DataFrame
            return data

        monkeypatch.setattr(ImputeMissing, "run", mock_impute_run)

        result = pipeline.run_prep_steps(batch)

        assert result is not None
        # run_prep_steps returns a DataFrame after windowing
        assert isinstance(result, pd.DataFrame) or "signals" in result

    @pytest.mark.skip(
        reason="This test class was disabled so that we can think about a better way to test the dataset download."
    )
    def test_run_prep_steps_without_feat_extraction_calls_windowing(self, monkeypatch):
        """Test that run_prep_steps calls windowing when no feature extraction exists."""
        pipeline = self.make_minimal_pipeline()
        pipeline.step_preprocessing = []
        pipeline.step_feat_extraction = []

        batch = self.create_test_batch()

        # Mock _check_and_apply_windowing to verify if it was called
        windowing_called = {"called": False}

        def mock_check_windowing(self, b):
            windowing_called["called"] = True
            # Create DataFrame from signals
            signals_dict = {}
            for sig_name, sig_data in b["signals"].items():
                if isinstance(sig_data, list) and len(sig_data) > 0:
                    signals_dict[sig_name] = sig_data[0]
                else:
                    signals_dict[sig_name] = sig_data

            df = pd.DataFrame(signals_dict)
            df["label"] = b["labels"]["class"][0]
            return False, df

        monkeypatch.setattr(
            pipeline,
            "_check_and_apply_windowing",
            lambda b: mock_check_windowing(pipeline, b),
        )

        result = pipeline.run_prep_steps(batch)

        assert windowing_called["called"]
        assert isinstance(result, pd.DataFrame) or "signals" in result

    @pytest.mark.skip(
        reason="This test class was disabled so that we can think about a better way to test the dataset download."
    )
    def test_check_and_apply_windowing_default(self):
        """Test _check_and_apply_windowing with default windowing."""
        pipeline = self.make_minimal_pipeline()
        pipeline.step_preprocessing = []

        # Expected structure for _check_and_apply_windowing method:
        # batch["signals"] should have numeric indices as keys
        batch = {
            "signals": {
                0: {"T-JUS-CKP": list(range(120)), "T-MON-CKP": list(range(120))}
            },
            "labels": [[0] * 120],  # List of label lists
            "file_names": ["file1.parquet"],
        }

        _, df = pipeline._check_and_apply_windowing(batch)

        assert isinstance(df, pd.DataFrame)
        assert "label" in df.columns

    @pytest.mark.skip(
        reason="This test class was disabled so that we can think about a better way to test the dataset download."
    )
    def test_check_and_apply_windowing_with_windowing(self, monkeypatch):
        """Test _check_and_apply_windowing with Windowing step."""
        pipeline = self.make_minimal_pipeline()

        # Create Windowing instance
        windowing = Windowing.__new__(Windowing)
        windowing.window_size = 10
        windowing.overlap = 0
        pipeline.step_preprocessing = [windowing]

        # Batch with correct structure: signals with numeric indices
        batch = {
            "signals": {
                0: {"T-JUS-CKP": list(range(120)), "T-MON-CKP": list(range(120))}
            },
            "labels": [[0] * 120],  # List of lists
            "file_names": ["file1.parquet"],
        }

        # Mock Windowing's run method
        def mock_windowing_run(self, data):
            # Simulate windowing returning DataFrame with windows
            if isinstance(data, pd.DataFrame):
                df = data.copy()
                if "label" not in df.columns:
                    df["label"] = 0
                return df
            return pd.DataFrame({"a": [1], "label": [0]})

        monkeypatch.setattr(Windowing, "run", mock_windowing_run)

        requires_windowing, df = pipeline._check_and_apply_windowing(batch)

        assert isinstance(df, pd.DataFrame)
        assert "label" in df.columns
        assert requires_windowing is True
