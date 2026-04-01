"""Tests for core enums."""

import pytest

from ThreeWToolkit.core import (
    ModelTypeEnum,
    EventPrefixEnum,
    TaskTypeEnum,
    DataSplitEnum,
    ActivationFunctionEnum,
    OptimizersEnum,
    CriterionEnum,
)


class TestModelTypeEnum:
    """Test ModelTypeEnum values."""

    def test_mlp_value(self):
        """Test MLP enum value."""
        assert ModelTypeEnum.MLP.value == "MLP"

    def test_logistic_regression_value(self):
        """Test LogisticRegression enum value."""
        assert ModelTypeEnum.LOGISTIC_REGRESSION.value == "LogisticRegression"

    def test_all_model_types_defined(self):
        """Test all expected model types are defined."""
        expected = {
            "MLP", "LogisticRegression", "RandomForest", "GradientBoosting",
            "SVM", "KNN", "DecisionTree", "NaiveBayes"
        }
        actual = {e.value for e in ModelTypeEnum}
        assert actual == expected

    def test_model_type_is_string_enum(self):
        """Test that ModelTypeEnum inherits from str."""
        assert isinstance(ModelTypeEnum.MLP, str)
        assert ModelTypeEnum.MLP == "MLP"


class TestEventPrefixEnum:
    """Test EventPrefixEnum values."""

    def test_real_prefix(self):
        """Test REAL event prefix."""
        assert EventPrefixEnum.REAL.value == "WELL"

    def test_simulated_prefix(self):
        """Test SIMULATED event prefix."""
        assert EventPrefixEnum.SIMULATED.value == "SIMULATED"

    def test_drawn_prefix(self):
        """Test DRAWN event prefix."""
        assert EventPrefixEnum.DRAWN.value == "DRAWN"

    def test_event_prefix_is_string_enum(self):
        """Test that EventPrefixEnum inherits from str."""
        assert isinstance(EventPrefixEnum.REAL, str)


class TestTaskTypeEnum:
    """Test TaskTypeEnum values."""

    def test_classification_value(self):
        """Test classification task type."""
        assert TaskTypeEnum.CLASSIFICATION.value == "classification"

    def test_regression_value(self):
        """Test regression task type."""
        assert TaskTypeEnum.REGRESSION.value == "regression"

    def test_task_type_is_string_enum(self):
        """Test that TaskTypeEnum inherits from str."""
        assert isinstance(TaskTypeEnum.CLASSIFICATION, str)


class TestDataSplitEnum:
    """Test DataSplitEnum values."""

    def test_train_split(self):
        """Test TRAIN split value."""
        assert DataSplitEnum.TRAIN.value == "train"

    def test_validation_split(self):
        """Test VALIDATION split value."""
        assert DataSplitEnum.VALIDATION.value == "validation"

    def test_test_split(self):
        """Test TEST split value."""
        assert DataSplitEnum.TEST.value == "test"

    def test_custom_split(self):
        """Test CUSTOM split value."""
        assert DataSplitEnum.CUSTOM.value == "custom"


class TestActivationFunctionEnum:
    """Test ActivationFunctionEnum values."""

    def test_relu_activation(self):
        """Test ReLU activation function."""
        assert ActivationFunctionEnum.RELU.value == "relu"

    def test_sigmoid_activation(self):
        """Test Sigmoid activation function."""
        assert ActivationFunctionEnum.SIGMOID.value == "sigmoid"

    def test_tanh_activation(self):
        """Test Tanh activation function."""
        assert ActivationFunctionEnum.TANH.value == "tanh"


class TestOptimizersEnum:
    """Test OptimizersEnum values."""

    def test_adam_optimizer(self):
        """Test Adam optimizer."""
        assert OptimizersEnum.ADAM.value == "adam"

    def test_adamw_optimizer(self):
        """Test AdamW optimizer."""
        assert OptimizersEnum.ADAMW.value == "adamw"

    def test_sgd_optimizer(self):
        """Test SGD optimizer."""
        assert OptimizersEnum.SGD.value == "sgd"

    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer."""
        assert OptimizersEnum.RMSPROP.value == "rmsprop"


class TestCriterionEnum:
    """Test CriterionEnum values."""

    def test_cross_entropy(self):
        """Test CrossEntropy criterion."""
        assert CriterionEnum.CROSS_ENTROPY.value == "cross_entropy"

    def test_binary_cross_entropy(self):
        """Test BinaryCrossEntropy criterion."""
        assert CriterionEnum.BINARY_CROSS_ENTROPY.value == "binary_cross_entropy"

    def test_mse_criterion(self):
        """Test MSE criterion."""
        assert CriterionEnum.MSE.value == "mse"

    def test_mae_criterion(self):
        """Test MAE criterion."""
        assert CriterionEnum.MAE.value == "mae"


class TestEnumUsagePatterns:
    """Test common enum usage patterns."""

    def test_enum_comparison(self):
        """Test enum comparison."""
        task = TaskTypeEnum.CLASSIFICATION
        assert task == TaskTypeEnum.CLASSIFICATION
        assert task != TaskTypeEnum.REGRESSION

    def test_enum_in_conditional(self):
        """Test using enum in conditional logic."""
        task = TaskTypeEnum.CLASSIFICATION

        if task == TaskTypeEnum.CLASSIFICATION:
            result = "class"
        else:
            result = "reg"

        assert result == "class"

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        splits = list(DataSplitEnum)
        assert len(splits) == 4
        assert DataSplitEnum.TRAIN in splits
