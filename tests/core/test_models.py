"""Tests for BaseModels and ModelsConfig."""

import pytest
import torch
from torch import nn
from ThreeWToolkit.core import (
    BaseModels,
    BaseSkLearnModels,
    BaseTorchModels,
    ModelsConfig,
    ModelTypeEnum,
)


class TestModelsConfig:
    """Test ModelsConfig validation."""

    def test_valid_mlp_model_type(self):
        """Test creating config with MLP model type."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        config = ModelsConfig(
            model_type=ModelTypeEnum.MLP,
            _target=DummyModel,
        )

        assert config.model_type == ModelTypeEnum.MLP

    def test_valid_random_forest_model_type(self):
        """Test creating config with RandomForest model type."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        config = ModelsConfig(
            model_type=ModelTypeEnum.RANDOM_FOREST,
            _target=DummyModel,
        )

        assert config.model_type == ModelTypeEnum.RANDOM_FOREST

    def test_all_valid_model_types(self):
        """Test all valid model types can be used."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        valid_types = [
            ModelTypeEnum.MLP,
            ModelTypeEnum.LOGISTIC_REGRESSION,
            ModelTypeEnum.RANDOM_FOREST,
            ModelTypeEnum.DECISION_TREE,
            ModelTypeEnum.GRADIENT_BOOSTING,
            ModelTypeEnum.KNN,
            ModelTypeEnum.NAIVE_BAYES,
            ModelTypeEnum.SVM,
        ]

        for model_type in valid_types:
            config = ModelsConfig(model_type=model_type, _target=DummyModel)
            assert config.model_type == model_type

    def test_default_random_seed(self):
        """Test default random seed is 42."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        config = ModelsConfig(
            model_type=ModelTypeEnum.MLP,
            _target=DummyModel,
        )

        assert config.random_seed == 42

    def test_custom_random_seed(self):
        """Test custom random seed."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        config = ModelsConfig(
            model_type=ModelTypeEnum.MLP,
            random_seed=123,
            _target=DummyModel,
        )

        assert config.random_seed == 123

    def test_none_random_seed(self):
        """Test None random seed (non-reproducible)."""

        class DummyModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        config = ModelsConfig(
            model_type=ModelTypeEnum.MLP,
            random_seed=None,
            _target=DummyModel,
        )

        assert config.random_seed is None


class TestBaseModels:
    """Test BaseModels abstract class."""

    def test_model_name_property(self):
        """Test model_name property returns class name."""

        class MyCustomModel(BaseModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        model = MyCustomModel()
        assert model.model_name == "MyCustomModel"

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        class IncompleteModel(BaseModels):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()


class TestBaseSkLearnModels:
    """Test BaseSkLearnModels abstract class."""

    def test_sklearn_model_implementation(self):
        """Test implementing a sklearn-style model."""

        class MySklearnModel(BaseSkLearnModels):
            def __init__(self):
                self._params = {"n_estimators": 100}

            def save(self, filename):
                pass

            def load(self, filename):
                return self

            def get_params(self):
                return self._params

        model = MySklearnModel()

        assert model.get_params() == {"n_estimators": 100}
        assert model.model_name == "MySklearnModel"

    def test_sklearn_model_requires_get_params(self):
        """Test that get_params must be implemented."""

        class IncompleteSklearnModel(BaseSkLearnModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

        with pytest.raises(TypeError):
            IncompleteSklearnModel()


class TestBaseTorchModels:
    """Test BaseTorchModels abstract class."""

    def test_torch_model_implementation(self):
        """Test implementing a torch-style model."""

        class MyTorchModel(BaseTorchModels):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                return self.linear(x)

            def get_params(self):
                return self.parameters()

            def save(self, filename):
                torch.save(self.state_dict(), filename)

            def load(self, filename):
                self.load_state_dict(torch.load(filename))
                return self

        model = MyTorchModel()

        assert model.model_name == "MyTorchModel"
        assert hasattr(model, "forward")
        assert hasattr(model, "parameters")

    def test_torch_model_requires_forward(self):
        """Test that forward must be implemented."""

        class IncompleteTorchModel(BaseTorchModels):
            def save(self, filename):
                pass

            def load(self, filename):
                pass

            def get_params(self):
                return []

        with pytest.raises(TypeError):
            IncompleteTorchModel()
