"""Tests for SklearnTrainer and SklearnTrainerConfig."""

import pytest
import numpy as np
import pandas as pd

from ThreeWToolkit.trainer.sklearn_trainer import SklearnTrainer, SklearnTrainerConfig
from ThreeWToolkit.models.sklearn_models import SklearnModelsConfig
from ThreeWToolkit.core.enums import ModelTypeEnum
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from ThreeWToolkit.core.base_dataset import BaseDataset


class MockDataset(BaseDataset):
    """Mock dataset for testing trainers."""

    def __init__(self, events: list[DatasetOutputs]):
        self.events = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> DatasetOutputs:
        return self.events[idx]


def create_mock_dataset(
    num_events: int = 20,
    num_features: int = 10,
    num_timesteps: int = 50,
    num_classes: int = 2,
    seed: int = 42,
) -> MockDataset:
    """Create a mock dataset for testing."""
    np.random.seed(seed)
    events = []

    for i in range(num_events):
        signal_data = np.random.randn(num_timesteps, num_features).astype(np.float32)
        signal_df = pd.DataFrame(
            signal_data, columns=[f"sensor_{j}" for j in range(num_features)]
        )
        label_data = np.full(num_timesteps, i % num_classes, dtype=np.int64)
        label_series = pd.Series(label_data, name="label")

        events.append(
            DatasetOutputs(
                signal=signal_df,
                label=label_series,
                metadata={"event_id": i},
            )
        )

    return MockDataset(events)


class TestSklearnTrainerConfig:
    """Tests for SklearnTrainerConfig validation."""

    @pytest.fixture
    def sklearn_config(self):
        """Provide basic sklearn model config."""
        return SklearnModelsConfig(model_type=ModelTypeEnum.LOGISTIC_REGRESSION)

    def test_valid_config(self, sklearn_config):
        """Valid config should be created."""
        config = SklearnTrainerConfig(
            config_model=sklearn_config,
            n_jobs=2,
            verbose=1,
        )
        assert config.n_jobs == 2
        assert config.verbose == 1

    def test_default_values(self, sklearn_config):
        """Config should have sensible defaults."""
        config = SklearnTrainerConfig(config_model=sklearn_config)
        assert config.n_jobs is None
        assert config.verbose == 0
        assert config.seed == 42

    def test_n_jobs_none(self, sklearn_config):
        """n_jobs=None should be valid."""
        config = SklearnTrainerConfig(config_model=sklearn_config, n_jobs=None)
        assert config.n_jobs is None

    def test_n_jobs_positive(self, sklearn_config):
        """Positive n_jobs should be valid."""
        config = SklearnTrainerConfig(config_model=sklearn_config, n_jobs=4)
        assert config.n_jobs == 4

    def test_n_jobs_negative_one(self, sklearn_config):
        """n_jobs=-1 (all cores) should be valid."""
        config = SklearnTrainerConfig(config_model=sklearn_config, n_jobs=-1)
        assert config.n_jobs == -1

    def test_n_jobs_zero_invalid(self, sklearn_config):
        """n_jobs=0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            SklearnTrainerConfig(config_model=sklearn_config, n_jobs=0)

    def test_verbose_non_negative(self, sklearn_config):
        """Negative verbose should raise ValidationError."""
        with pytest.raises(ValueError):
            SklearnTrainerConfig(config_model=sklearn_config, verbose=-1)

    def test_target_returns_trainer_class(self, sklearn_config):
        """target_ should return SklearnTrainer class."""
        config = SklearnTrainerConfig(config_model=sklearn_config)
        assert config.target_ == SklearnTrainer

    def test_class_weight_config(self, sklearn_config):
        """Class weight config should be accepted."""
        config = SklearnTrainerConfig(
            config_model=sklearn_config,
            use_class_weights=True,
            class_weight_strategy="balanced",
        )
        assert config.use_class_weights is True
        assert config.class_weight_strategy == "balanced"

    def test_manual_class_weights(self, sklearn_config):
        """Manual class weights should be configurable."""
        config = SklearnTrainerConfig(
            config_model=sklearn_config,
            use_class_weights=True,
            class_weight_strategy="manual",
            manual_class_weights={0: 1.0, 1: 2.0},
        )
        assert config.manual_class_weights == {0: 1.0, 1: 2.0}


class TestSklearnTrainer:
    """Tests for SklearnTrainer functionality."""

    @pytest.fixture
    def logistic_trainer_config(self):
        """Provide logistic regression trainer config."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION
        )
        return SklearnTrainerConfig(config_model=sklearn_config)

    @pytest.fixture
    def rf_trainer_config(self):
        """Provide random forest trainer config."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            model_params={"n_estimators": 10, "max_depth": 3},
        )
        return SklearnTrainerConfig(config_model=sklearn_config, n_jobs=1)

    @pytest.fixture
    def mock_dataset(self):
        """Provide mock dataset for testing."""
        return create_mock_dataset(num_events=20, num_features=10, num_timesteps=50)

    def test_trainer_initialization_logistic(self, logistic_trainer_config):
        """Logistic regression trainer should initialize correctly."""
        trainer = SklearnTrainer(logistic_trainer_config)
        assert trainer.config == logistic_trainer_config
        assert trainer.model is not None
        assert trainer.model.model_name == "LogisticRegression"

    def test_trainer_initialization_random_forest(self, rf_trainer_config):
        """Random forest trainer should initialize correctly."""
        trainer = SklearnTrainer(rf_trainer_config)
        assert trainer.model.model_name == "RandomForestClassifier"
        params = trainer.model.get_params()
        assert params["n_estimators"] == 10
        assert params["max_depth"] == 3

    def test_n_jobs_set(self, rf_trainer_config):
        """n_jobs should be set on model if supported."""
        trainer = SklearnTrainer(rf_trainer_config)
        params = trainer.model.get_params()
        assert params["n_jobs"] == 1

    def test_train_basic(self, logistic_trainer_config, mock_dataset):
        """Basic training should complete successfully."""
        trainer = SklearnTrainer(logistic_trainer_config)
        result = trainer.train(mock_dataset)

        assert result is not None
        assert result.model is not None
        assert result.train_dataset_size == len(mock_dataset)

    def test_train_with_validation(self, logistic_trainer_config, mock_dataset):
        """Training with validation should track val_score."""
        trainer = SklearnTrainer(logistic_trainer_config)
        val_dataset = create_mock_dataset(
            num_events=10, num_features=10, num_timesteps=50, seed=99
        )

        result = trainer.train(mock_dataset, val_dataset)

        assert result.val_dataset_size == len(val_dataset)
        assert "val_score" in result.history
        assert 0 <= result.history["val_score"] <= 1

    def test_train_with_class_weights_balanced(self, mock_dataset):
        """Training with balanced class weights should work."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION
        )
        config = SklearnTrainerConfig(
            config_model=sklearn_config,
            use_class_weights=True,
            class_weight_strategy="balanced",
        )
        trainer = SklearnTrainer(config)
        result = trainer.train(mock_dataset)

        assert result is not None
        # Check class_weight was set on model
        params = trainer.model.get_params()
        assert params.get("class_weight") is not None

    def test_train_with_manual_class_weights(self, mock_dataset):
        """Training with manual class weights should work."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION
        )
        config = SklearnTrainerConfig(
            config_model=sklearn_config,
            use_class_weights=True,
            class_weight_strategy="manual",
            manual_class_weights={0: 1.0, 1: 3.0},
        )
        trainer = SklearnTrainer(config)
        result = trainer.train(mock_dataset)

        assert result is not None
        params = trainer.model.get_params()
        assert params["class_weight"] == {0: 1.0, 1: 3.0}

    def test_manual_weights_without_dict_raises(self, mock_dataset):
        """Manual strategy without weights dict should raise ValueError."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION
        )
        config = SklearnTrainerConfig(
            config_model=sklearn_config,
            use_class_weights=True,
            class_weight_strategy="manual",
            manual_class_weights=None,  # Missing weights
        )
        trainer = SklearnTrainer(config)

        with pytest.raises(ValueError, match="manual_class_weights required"):
            trainer.train(mock_dataset)

    def test_model_predictions(self, logistic_trainer_config, mock_dataset):
        """Trained model should make predictions."""
        trainer = SklearnTrainer(logistic_trainer_config)
        trainer.train(mock_dataset)

        # Create test data with same number of features (not flattened timesteps)
        X_test = np.random.randn(10, 10)  # num_features only
        predictions = trainer.model.model_class.predict(X_test)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_model_probability_predictions(self, logistic_trainer_config, mock_dataset):
        """Trained model should predict probabilities."""
        trainer = SklearnTrainer(logistic_trainer_config)
        trainer.train(mock_dataset)

        X_test = np.random.randn(10, 10)  # num_features only
        probs = trainer.model.model_class.predict_proba(X_test)

        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_metadata_in_result(self, logistic_trainer_config, mock_dataset):
        """Training result should contain metadata."""
        trainer = SklearnTrainer(logistic_trainer_config)
        result = trainer.train(mock_dataset)

        assert "trainer_type" in result.metadata
        assert result.metadata["trainer_type"] == "SklearnTrainer"
        assert "seed" in result.metadata


class TestSklearnTrainerValidation:
    """Tests for SklearnTrainer dataset validation."""

    @pytest.fixture
    def trainer(self):
        """Provide trainer for validation tests."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION
        )
        config = SklearnTrainerConfig(config_model=sklearn_config)
        return SklearnTrainer(config)

    def test_empty_dataset_raises(self, trainer):
        """Empty dataset should raise ValueError."""
        empty_dataset = MockDataset([])

        with pytest.raises(ValueError, match="Training dataset is empty"):
            trainer.train(empty_dataset)

    def test_empty_validation_dataset_raises(self, trainer):
        """Empty validation dataset should raise ValueError."""
        train_dataset = create_mock_dataset(num_events=10)
        empty_val = MockDataset([])

        with pytest.raises(ValueError, match="Validation dataset is empty"):
            trainer.train(train_dataset, empty_val)

    def test_mismatched_columns_raises(self, trainer):
        """Mismatched signal columns should raise ValueError."""
        train_dataset = create_mock_dataset(num_events=10, num_features=10)

        # Create val dataset with different number of features
        val_events = []
        signal_df = pd.DataFrame(
            np.random.randn(50, 5),  # Different number of features
            columns=[f"sensor_{j}" for j in range(5)],
        )
        label_series = pd.Series(np.zeros(50, dtype=np.int64), name="label")
        val_events.append(
            DatasetOutputs(signal=signal_df, label=label_series, metadata={})
        )
        val_dataset = MockDataset(val_events)

        with pytest.raises(ValueError, match="different signal columns"):
            trainer.train(train_dataset, val_dataset)


class TestSklearnTrainerAllModels:
    """Tests for SklearnTrainer with all supported model types."""

    @pytest.fixture
    def mock_dataset(self):
        """Provide mock dataset for testing."""
        return create_mock_dataset(num_events=30, num_features=10, num_timesteps=20)

    @pytest.mark.parametrize(
        "model_type",
        [
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.KNN,
            ModelTypeEnum.GRADIENT_BOOSTING,
        ],
    )
    def test_train_all_model_types(self, model_type, mock_dataset):
        """All model types should train successfully."""
        sklearn_config = SklearnModelsConfig(model_type=model_type)
        config = SklearnTrainerConfig(config_model=sklearn_config)

        trainer = SklearnTrainer(config)
        result = trainer.train(mock_dataset)

        assert result is not None
        assert result.model is not None

    def test_train_svm(self, mock_dataset):
        """SVM should train (with probability for predict_proba)."""
        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.SVM,
            model_params={"probability": True},  # Required for predict_proba
        )
        config = SklearnTrainerConfig(config_model=sklearn_config)

        trainer = SklearnTrainer(config)
        result = trainer.train(mock_dataset)

        assert result is not None

    def test_train_naive_bayes(self, mock_dataset):
        """Naive Bayes should train (requires non-negative features)."""
        # Create dataset with non-negative features for Naive Bayes
        np.random.seed(42)
        events = []
        for i in range(30):
            signal_data = np.abs(np.random.randn(20, 10).astype(np.float32))
            signal_df = pd.DataFrame(
                signal_data, columns=[f"sensor_{j}" for j in range(10)]
            )
            label_data = np.full(20, i % 2, dtype=np.int64)
            label_series = pd.Series(label_data, name="label")
            events.append(
                DatasetOutputs(signal=signal_df, label=label_series, metadata={})
            )

        positive_dataset = MockDataset(events)

        sklearn_config = SklearnModelsConfig(model_type=ModelTypeEnum.NAIVE_BAYES)
        config = SklearnTrainerConfig(config_model=sklearn_config)

        trainer = SklearnTrainer(config)
        result = trainer.train(positive_dataset)

        assert result is not None


class TestSklearnTrainerIntegration:
    """Integration tests for SklearnTrainer with realistic scenarios."""

    def test_multiclass_classification(self):
        """Trainer should handle multi-class classification."""
        num_classes = 5
        dataset = create_mock_dataset(
            num_events=50, num_features=20, num_classes=num_classes
        )

        sklearn_config = SklearnModelsConfig(model_type=ModelTypeEnum.RANDOM_FOREST)
        config = SklearnTrainerConfig(config_model=sklearn_config)

        trainer = SklearnTrainer(config)
        result = trainer.train(dataset)

        # Model should predict all classes (with correct num features)
        X_test = np.random.randn(20, 20)  # num_features only
        predictions = trainer.model.model_class.predict(X_test)
        probs = trainer.model.model_class.predict_proba(X_test)

        assert probs.shape[1] == num_classes

    def test_reproducibility_with_seed(self):
        """Training with same seed should produce same results."""
        dataset = create_mock_dataset(num_events=30, seed=42)

        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            model_params={"n_estimators": 10},
        )
        config = SklearnTrainerConfig(config_model=sklearn_config, seed=42)

        # Train twice with same seed
        trainer1 = SklearnTrainer(config)
        result1 = trainer1.train(dataset)

        trainer2 = SklearnTrainer(config)
        result2 = trainer2.train(dataset)

        # Same predictions on same input (with correct num features)
        X_test = np.random.RandomState(42).randn(10, 10)  # num_features only
        pred1 = result1.model.model_class.predict(X_test)
        pred2 = result2.model.model_class.predict(X_test)

        assert np.array_equal(pred1, pred2)

    def test_large_dataset(self):
        """Trainer should handle larger datasets."""
        dataset = create_mock_dataset(
            num_events=100, num_features=50, num_timesteps=100
        )

        sklearn_config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION,
            model_params={"max_iter": 1000},
        )
        config = SklearnTrainerConfig(config_model=sklearn_config)

        trainer = SklearnTrainer(config)
        result = trainer.train(dataset)

        assert result.train_dataset_size == 100
