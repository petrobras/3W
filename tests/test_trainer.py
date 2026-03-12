import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pydantic
from unittest.mock import MagicMock, patch

from ThreeWToolkit.trainer.trainer import (
    ModelTrainer,
    TrainerConfig,
    TrainInput,
    TrainOutput,
)
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.core.enums import TaskTypeEnum


def mlp_config(input_size=8, output_size=1):
    """
    Creates a default MLPConfig used in trainer tests.
    """
    return MLPConfig(
        input_size=input_size,
        hidden_sizes=(16, 8),
        output_size=output_size,
        activation_function="relu",
        random_seed=42,
    )


def base_trainer_config(**overrides):
    """
    Creates a TrainerConfig instance with default test values.

    Args:
        **overrides (dict): Optional configuration values that override the defaults.

    Returns:
        A trainer configuration suitable for testing.
    """
    defaults = dict(
        batch_size=8,
        epochs=2,
        seed=42,
        learning_rate=1e-3,
        config_model=mlp_config(),
        optimizer="adam",
        criterion="mse",
        device="cpu",
        shuffle_train=True,
        deterministic=True,
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def make_data(n=40, input_size=8):
    """
    Generates synthetic tabular data for training tests.

    Args:
        n (int): Number of samples.
        input_size (int): Number of input features.

    Returns:
        A tuple (x, y) containing a DataFrame of features
        and a Series of target values.
    """
    x = pd.DataFrame(np.random.rand(n, input_size).astype(np.float32))
    y = pd.Series(np.random.rand(n).astype(np.float32))
    return x, y


class TestTrainerConfig:
    """
    Tests validation and behavior of TrainerConfig.
    """

    def test_valid_config(self):
        cfg = base_trainer_config()
        assert cfg.batch_size == 8
        assert cfg.optimizer == "adam"

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("batch_size", 0, "batch_size"),
            ("epochs", 0, "epochs"),
            ("learning_rate", 0.0, "learning_rate"),
        ],
    )
    def test_non_positive_fields_raise(self, field, value, match):
        """Ensures non-positive numeric fields raise validation errors."""
        with pytest.raises(pydantic.ValidationError, match=match):
            base_trainer_config(**{field: value})

    def test_invalid_optimizer_raises(self):
        """Ensures invalid optimizer names raise validation errors."""
        with pytest.raises(pydantic.ValidationError, match="optimizer must be one of"):
            base_trainer_config(optimizer="momentum")

    def test_invalid_criterion_raises(self):
        """Ensures invalid loss criteria raise validation errors."""
        with pytest.raises(pydantic.ValidationError, match="criterion must be one of"):
            base_trainer_config(criterion="huber")

    def test_invalid_device_raises(self):
        """Ensures unsupported device types raise validation errors."""
        with pytest.raises(pydantic.ValidationError, match="device must be"):
            base_trainer_config(device="tpu")

    def test_invalid_class_weight_strategy_raises(self):
        """Ensures invalid class weight strategies raise validation errors."""
        with pytest.raises(pydantic.ValidationError, match="class_weight_strategy"):
            base_trainer_config(class_weight_strategy="custom")

    def test_manual_class_weights_requires_dict(self):
        """Ensures manual class weights require a dictionary."""
        with pytest.raises(
            pydantic.ValidationError, match="manual_class_weights must be provided"
        ):
            base_trainer_config(
                use_class_weights=True,
                class_weight_strategy="manual",
                manual_class_weights=None,
            )

    def test_manual_class_weights_values_must_be_positive(self):
        """Ensures manual class weights must be positive."""
        with pytest.raises(pydantic.ValidationError, match="must be > 0"):
            base_trainer_config(
                use_class_weights=True,
                class_weight_strategy="manual",
                manual_class_weights={0: -1.0, 1: 1.0},
            )

    def test_task_type_defaults_to_classification(self):
        """Ensures default task type is classification."""
        cfg = base_trainer_config()
        assert cfg.task_type == TaskTypeEnum.CLASSIFICATION

    def test_batch_size_zero_raises(self):
        """Ensures batch_size equal to zero raises validation error."""
        with pytest.raises(pydantic.ValidationError, match="greater than 0"):
            base_trainer_config(batch_size=0)

    def test_epochs_zero_raises(self):
        """Ensures epochs equal to zero raises validation error."""
        with pytest.raises(pydantic.ValidationError, match="greater than 0"):
            base_trainer_config(epochs=0)

    def test_learning_rate_zero_raises(self):
        """Ensures learning_rate equal to zero raises validation error."""
        with pytest.raises(pydantic.ValidationError, match="greater than 0"):
            base_trainer_config(learning_rate=0)

    def test_n_splits_not_valid_when_cross_validation_raises(self):
        """Ensures invalid number of CV splits raises validation error."""
        with pytest.raises(pydantic.ValidationError, match="greater than 1"):
            base_trainer_config(cross_validation=True, n_splits=0)

    def test_device_cuda_without_gpu_raises(self):
        """Ensures selecting CUDA without GPU availability raises error."""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(
                pydantic.ValidationError, match="CUDA selected but no GPU is available"
            ):
                base_trainer_config(device="cuda")

    def test_manual_class_weights_key_negative_raises(self):
        """Ensures negative class weight keys raise validation errors."""
        with pytest.raises(pydantic.ValidationError):
            base_trainer_config(
                use_class_weights=True,
                class_weight_strategy="manual",
                manual_class_weights={0: -1.0},
            )


class TestModelTrainerPreProcess:
    """
    Tests validation and preprocessing of training inputs.
    """

    @pytest.fixture
    def trainer(self):
        """Creates a trainer instance for preprocessing tests."""
        return ModelTrainer(base_trainer_config())

    def test_valid_train_input_passes(self, trainer):
        """Ensures valid TrainInput passes preprocessing."""
        x, y = make_data()
        result = trainer.pre_process(TrainInput(x_train=x, y_train=y))
        assert result.x_train is x

    def test_raises_if_not_train_input(self, trainer):
        """Ensures non-TrainInput objects raise TypeError."""
        with pytest.raises(TypeError, match="TrainInput"):
            trainer.pre_process({"x_train": [], "y_train": []})

    def test_raises_if_x_train_is_none(self, trainer):
        """Ensures missing training data raises validation error."""
        with pytest.raises(ValueError, match="x_train and y_train must not be None"):
            trainer.pre_process(TrainInput(x_train=None, y_train=pd.Series([1])))

    def test_raises_if_only_x_val_provided(self, trainer):
        """Ensures validation data must be provided as a pair."""
        x, y = make_data()
        with pytest.raises(
            ValueError, match="x_val and y_val must be provided together"
        ):
            trainer.pre_process(TrainInput(x_train=x, y_train=y, x_val=x, y_val=None))


class TestModelTrainerTrainRouting:
    """
    Tests routing logic for different training scenarios.
    """

    @pytest.fixture
    def trainer(self):
        """Creates a regression trainer instance."""
        return ModelTrainer(
            base_trainer_config(criterion="mse", task_type=TaskTypeEnum.REGRESSION)
        )

    def test_train_with_auto_split(self, trainer):
        """Ensures training works with automatic validation split."""
        x, y = make_data()
        trainer.train(x, y)
        assert len(trainer.history["models"]) == 1

    def test_train_with_explicit_validation(self, trainer):
        """Ensures training works with explicitly provided validation data."""
        x_train, y_train = make_data(30)
        x_val, y_val = make_data(10)
        trainer.train(x_train, y_train, x_val, y_val)
        assert len(trainer.history["x_val"]) == 1

    def test_train_with_cross_validation_regression(self):
        """Ensures regression cross-validation trains multiple models."""
        cfg = base_trainer_config(
            cross_validation=True,
            n_splits=3,
            criterion="mse",
            task_type=TaskTypeEnum.REGRESSION,
        )
        trainer = ModelTrainer(cfg)
        x, y = make_data(60)
        trainer.train(x, y)
        assert len(trainer.history["models"]) == 3

    def test_train_with_cross_validation_classification(self):
        """Ensures classification cross-validation trains multiple models."""
        cfg = base_trainer_config(
            cross_validation=True,
            n_splits=3,
            criterion="cross_entropy",
            config_model=mlp_config(output_size=2),
            task_type=TaskTypeEnum.CLASSIFICATION,
        )
        trainer = ModelTrainer(cfg)
        x = pd.DataFrame(np.random.rand(60, 8).astype(np.float32))
        y = pd.Series(np.random.randint(0, 2, 60))
        trainer.train(x, y)
        assert len(trainer.history["models"]) == 3


class TestGetOptimizer:
    """Tests for optimizer creation inside ModelTrainer."""

    @pytest.fixture
    def trainer(self):
        """Returns a ModelTrainer instance with default configuration."""
        return ModelTrainer(base_trainer_config())

    def _model(self):
        """Creates a simple MLP model instance for optimizer tests."""
        from ThreeWToolkit.models.mlp import MLP

        return MLP(mlp_config())

    @pytest.mark.parametrize(
        "opt_name,opt_cls",
        [
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
            ("rmsprop", torch.optim.RMSprop),
        ],
    )
    def test_optimizer_types(self, trainer, opt_name, opt_cls):
        """Ensures that the correct optimizer class is returned."""
        model = self._model()
        trainer.config = base_trainer_config(optimizer=opt_name)
        opt = trainer._get_optimizer(model, opt_name)
        assert isinstance(opt, opt_cls)

    def test_optmizer_type_is_invalid(self, trainer):
        """Ensures an error is raised when an unknown optimizer is requested."""
        model = self._model()
        trainer.config = base_trainer_config(optimizer="adam")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            trainer._get_optimizer(model, "invalid")

    def test_raises_for_model_without_get_params(self, trainer):
        """Ensures an error is raised if the model does not expose parameters."""
        bad_model = MagicMock(spec=[])  # sem get_params
        with pytest.raises(TypeError, match="get_params"):
            trainer._get_optimizer(bad_model, "adam")


class TestGetFnCost:
    """Tests for loss function selection in ModelTrainer."""

    @pytest.fixture
    def trainer(self):
        """Returns a ModelTrainer instance with default configuration."""
        return ModelTrainer(base_trainer_config())

    @pytest.mark.parametrize(
        "criterion,expected_cls",
        [
            ("cross_entropy", nn.CrossEntropyLoss),
            ("binary_cross_entropy", nn.BCEWithLogitsLoss),
            ("mse", nn.MSELoss),
            ("mae", nn.L1Loss),
        ],
    )
    def test_criterion_types(self, trainer, criterion, expected_cls):
        """Ensures the correct loss function class is returned."""
        loss_fn = trainer._get_fn_cost(criterion)
        assert isinstance(loss_fn, expected_cls)

    def test_unknown_criterion_raises(self, trainer):
        """Ensures an error is raised for an unknown loss function."""
        with pytest.raises(ValueError, match="Unknown criterion"):
            trainer._get_fn_cost("huber")


class TestComputeClassWeights:
    """Tests for class weight computation strategies."""

    def test_balanced_strategy(self):
        """Ensures balanced strategy returns valid tensor weights."""
        cfg = base_trainer_config(
            use_class_weights=True, class_weight_strategy="balanced"
        )
        trainer = ModelTrainer(cfg)
        y = np.array([0] * 30 + [1] * 10)
        weights = trainer._compute_class_weights(y)
        assert isinstance(weights, torch.Tensor)
        assert weights.shape[0] == 2

    def test_manual_strategy(self):
        """Ensures manual class weights are correctly returned."""
        cfg = base_trainer_config(
            use_class_weights=True,
            class_weight_strategy="manual",
            manual_class_weights={0: 1.0, 1: 2.0},
        )
        trainer = ModelTrainer(cfg)
        weights = trainer._compute_class_weights(np.array([0, 1]))
        assert weights.tolist() == [1.0, 2.0]

    def test_manual_strategy_without_weights_raises(self):
        """Ensures an error is raised if manual weights are missing."""
        cfg = base_trainer_config(
            use_class_weights=True,
            class_weight_strategy="manual",
            manual_class_weights={0: 1.0, 1: 2.0},
        )
        trainer = ModelTrainer(cfg)
        trainer.config.manual_class_weights = None  # força o erro interno
        with pytest.raises(ValueError, match="manual_class_weights must be provided"):
            trainer._compute_class_weights(np.array([0, 1]))

    def test_get_fn_cost_injects_class_weights_when_enabled(self):
        """Ensures class weights are injected into the loss function."""
        cfg = base_trainer_config(
            use_class_weights=True,
            class_weight_strategy="balanced",
            criterion="cross_entropy",
            task_type=TaskTypeEnum.CLASSIFICATION,
            config_model=mlp_config(output_size=2),
        )
        trainer = ModelTrainer(cfg)
        y_train = np.array([0] * 30 + [1] * 10)

        loss_fn = trainer._get_fn_cost("cross_entropy", y_train=y_train)

        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert loss_fn.weight.shape[0] == 2


class TestSelectRows:
    """Tests for row selection utility used by the trainer."""

    @pytest.fixture
    def trainer(self):
        """Returns a ModelTrainer instance."""
        return ModelTrainer(base_trainer_config())

    def test_select_rows_dataframe(self, trainer):
        """Ensures rows are correctly selected from a DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = trainer._select_rows(df, [0, 2])
        assert list(result["a"]) == [1, 3]

    def test_select_rows_numpy(self, trainer):
        """Ensures rows are correctly selected from a NumPy array."""
        arr = np.array([10, 20, 30])
        result = trainer._select_rows(arr, [1, 2])
        assert list(result) == [20, 30]


class TestRunAndPostProcess:
    """Tests for the run and post_process pipeline of ModelTrainer."""

    def test_run_returns_train_output(self):
        """Ensures run returns a TrainOutput object."""
        cfg = base_trainer_config(criterion="mse", task_type=TaskTypeEnum.REGRESSION)
        trainer = ModelTrainer(cfg)
        x, y = make_data()
        result = trainer.run(TrainInput(x_train=x, y_train=y))
        assert isinstance(result, TrainOutput)
        assert len(result.models) == 1

    def test_post_process_populates_fields(self):
        """Ensures post_process populates metadata fields correctly."""
        cfg = base_trainer_config(criterion="mse", task_type=TaskTypeEnum.REGRESSION)
        trainer = ModelTrainer(cfg)
        x, y = make_data()
        train_output = trainer.run(TrainInput(x_train=x, y_train=y))
        result = trainer.post_process(train_output)
        assert result.model_name is not None
        assert result.trainer_config is cfg
        assert "train_losses" in result.history


class TestAssessMethod:
    """Tests for the assess method integration with ModelAssessment."""

    def _trained_trainer(self, task_type=TaskTypeEnum.REGRESSION, criterion="mse"):
        cfg = base_trainer_config(criterion=criterion, task_type=task_type)
        trainer = ModelTrainer(cfg)
        x, y = make_data()
        trainer.train(x, y)
        return trainer, *make_data(10)

    def test_assess_with_default_config_regression(self):
        """Ensures default regression metrics are used."""
        trainer, x_test, y_test = self._trained_trainer()
        with patch("ThreeWToolkit.trainer.trainer.ModelAssessment") as MockAssessor:
            MockAssessor.return_value.evaluate.return_value = MagicMock()
            _ = trainer.assess(x_test, y_test)
            call_config = MockAssessor.call_args[0][0]
            assert "explained_variance" in call_config.metrics
            assert call_config.task_type == TaskTypeEnum.REGRESSION

    def test_assess_with_default_config_classification(self):
        """Ensures default classification metrics are used."""
        cfg = base_trainer_config(
            criterion="cross_entropy",
            config_model=mlp_config(output_size=2),
            task_type=TaskTypeEnum.CLASSIFICATION,
        )
        trainer = ModelTrainer(cfg)
        x = pd.DataFrame(np.random.rand(40, 8).astype(np.float32))
        y = pd.Series(np.random.randint(0, 2, 40))
        trainer.train(x, y)
        x_test = pd.DataFrame(np.random.rand(10, 8).astype(np.float32))
        y_test = pd.Series(np.random.randint(0, 2, 10))

        with patch("ThreeWToolkit.trainer.trainer.ModelAssessment") as MockAssessor:
            MockAssessor.return_value.evaluate.return_value = MagicMock()
            _ = trainer.assess(x_test, y_test)
            call_config = MockAssessor.call_args[0][0]
            assert "accuracy" in call_config.metrics
            assert call_config.task_type == TaskTypeEnum.CLASSIFICATION

    def test_assess_with_explicit_config(self):
        """Ensures a custom assessment configuration is respected."""
        from ThreeWToolkit.assessment.model_assess import ModelAssessmentConfig

        trainer, x_test, y_test = self._trained_trainer()
        custom_config = ModelAssessmentConfig(
            metrics=["explained_variance"],
            task_type=TaskTypeEnum.REGRESSION,
        )
        with patch("ThreeWToolkit.trainer.trainer.ModelAssessment") as MockAssessor:
            MockAssessor.return_value.evaluate.return_value = MagicMock()
            trainer.assess(x_test, y_test, assessment_config=custom_config)
            MockAssessor.assert_called_once_with(custom_config)
