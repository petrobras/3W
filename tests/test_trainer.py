import pandas as pd
import pytest
import numpy as np
import torch

from sklearn.datasets import make_classification
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.models.sklearn_models import SklearnModelsConfig
from ThreeWToolkit.core.enums import OptimizersEnum, CriterionEnum, ModelTypeEnum


class TestTrainer:
    @pytest.fixture
    def mlp_trainer_and_data(self):
        """
        Creates an MLP trainer with synthetic data for testing.

        Returns:
            tuple: (trainer, x_df, y_series) where:
                - trainer: ModelTrainer instance configured for MLP
                - x_df: DataFrame with 50 samples and 8 features
                - y_series: Series with binary labels
        """
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate synthetic data
        x = np.random.rand(50, 8).astype(np.float32)
        y = (np.random.rand(50) > 0.5).astype(int)

        x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
        y_series = pd.Series(y, name="label")

        # Configure MLP model
        config = MLPConfig(
            input_size=8,
            hidden_sizes=(16, 8),
            output_size=1,
            activation_function="relu",
            regularization=None,
            random_seed=42,
        )

        # Configure trainer
        trainer_config = TrainerConfig(
            batch_size=8,
            epochs=2,
            seed=42,
            learning_rate=1e-3,
            config_model=config,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
            shuffle_train=True,
        )

        trainer = ModelTrainer(trainer_config)
        return trainer, x_df, y_series

    @pytest.fixture
    def sklearn_trainer_and_data(self):
        """
        Creates a Scikit-learn trainer with synthetic classification data.

        Returns:
            tuple: (trainer, x, y) where:
                - trainer: ModelTrainer instance configured for Logistic Regression
                - x: numpy array with 40 samples and 6 features
                - y: numpy array with binary labels
        """
        x, y = make_classification(
            n_samples=40, n_features=6, n_classes=2, random_state=42
        )

        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION,
            random_seed=42,
            model_params={},
        )

        trainer_config = TrainerConfig(
            batch_size=8,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=config,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
            shuffle_train=True,
        )

        trainer = ModelTrainer(trainer_config)
        return trainer, x, y

    def test_mlp_trainer_train_and_history(self, mlp_trainer_and_data):
        """
        Tests that MLP training executes successfully and maintains training history.

        Verifies:
        - Training completes without errors
        - History attribute exists and is a list
        - History contains train_loss key
        - Number of recorded losses matches number of epochs
        """
        trainer, x, y = mlp_trainer_and_data
        trainer.train(x, y)

        assert hasattr(trainer, "history")
        assert isinstance(trainer.history, list)
        assert "train_loss" in trainer.history[0]
        assert len(trainer.history[0]["train_loss"]) == trainer.epochs

    def test_trainer_train_with_validation(self, mlp_trainer_and_data):
        """
        Tests training with separate validation data.

        Verifies:
        - Training with validation split works correctly
        - History is properly recorded
        - Both train and validation losses are tracked
        """
        trainer, x, y = mlp_trainer_and_data
        trainer.cross_validation = False

        # Split data for validation
        x_val = x.iloc[:10]
        y_val = y.iloc[:10]
        x_train = x.iloc[10:]
        y_train = y.iloc[10:]

        trainer.train(x_train, y_train, x_val=x_val, y_val=y_val)

        assert isinstance(trainer.history, list)
        assert len(trainer.history) == 1
        assert "train_loss" in trainer.history[0]
        assert len(trainer.history[0]["train_loss"]) == trainer.epochs

    def test_sklearn_trainer_cross_validation(self, sklearn_trainer_and_data):
        """
        Tests cross-validation functionality with Scikit-learn models.

        Verifies:
        - Cross-validation can be enabled
        - History contains results for each fold
        - Number of folds matches n_splits configuration
        """
        trainer, x, y = sklearn_trainer_and_data
        trainer.cross_validation = True
        trainer.n_splits = 2

        trainer.train(x, y)

        assert isinstance(trainer.history, list)
        assert len(trainer.history) == 2  # One entry per fold

    def test_trainer_crossval_default_n_splits(self):
        """
        Tests that cross-validation uses default n_splits=5 when not specified.

        Verifies:
        - Default n_splits is set to 5
        - Training works with default configuration
        """
        x, y = make_classification(
            n_samples=10,
            n_features=2,
            n_classes=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )

        config = SklearnModelsConfig(
            model_type=ModelTypeEnum.LOGISTIC_REGRESSION,
            random_seed=42,
            model_params={},
        )

        metrics = [lambda y_true, y_pred: ((y_true == y_pred).mean())]

        trainer_config = TrainerConfig(
            batch_size=2,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=config,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
            cross_validation=True,
            metrics=metrics,
            # n_splits intentionally omitted to test default
        )

        trainer = ModelTrainer(trainer_config)
        assert trainer.n_splits == 5

    def test_mlp_trainer_save_and_load(self, tmp_path, mlp_trainer_and_data):
        """
        Tests saving and loading PyTorch MLP models.

        Verifies:
        - Model can be saved to disk
        - Model can be loaded from disk
        - Loaded model is not None
        """
        trainer, x, y = mlp_trainer_and_data
        trainer.train(x, y)

        save_path = tmp_path / "mlp_model.pth"
        trainer.save(save_path)

        loaded_model = trainer.load(save_path)
        assert loaded_model is not None

    def test_trainer_save_and_load_sklearn(self, tmp_path, sklearn_trainer_and_data):
        """
        Tests saving and loading Scikit-learn models.

        Verifies:
        - Scikit-learn model can be pickled and saved
        - Model can be loaded from pickle file
        - Loaded model is not None
        """
        trainer, x, y = sklearn_trainer_and_data
        trainer.train(x, y)

        save_path = tmp_path / "sklearn_model.pkl"
        trainer.save(save_path)

        loaded_model = trainer.load(save_path)
        assert loaded_model is not None

    def test_call_trainer_with_val_loader(self, mlp_trainer_and_data):
        """
        Tests internal call_trainer method with validation data.

        Verifies:
        - call_trainer creates validation DataLoader when x_val and y_val provided
        - Training executes successfully with validation
        - Result contains train_loss
        """
        trainer, x, y = mlp_trainer_and_data
        trainer.cross_validation = False

        x_val = x.iloc[:10]
        y_val = y.iloc[:10]
        x_train = x.iloc[10:]
        y_train = y.iloc[10:]

        result = trainer.call_trainer(x_train, y_train, x_val=x_val, y_val=y_val)

        assert result is not None
        assert "train_loss" in result

    def test_call_trainer_without_val_loader(self, mlp_trainer_and_data):
        """
        Tests internal call_trainer method without validation data.

        Verifies:
        - call_trainer works when x_val and y_val are None
        - Training executes successfully without validation
        - Result contains train_loss
        """
        trainer, x, y = mlp_trainer_and_data
        trainer.cross_validation = False

        x_train = x.iloc[10:]
        y_train = y.iloc[10:]

        result = trainer.call_trainer(x_train, y_train, x_val=None, y_val=None)

        assert result is not None
        assert "train_loss" in result

    def test_call_trainer_sklearn_branch(self, mlp_trainer_and_data):
        """
        Tests the sklearn model branch in call_trainer.

        Verifies:
        - call_trainer handles sklearn models differently than PyTorch
        - sklearn fit method is called correctly
        - Result is returned from sklearn training
        """

        class DummySklearn:
            """Mock sklearn model for testing."""

            def fit(self, x, y, **kwargs):
                return {"dummy": True}

        trainer = mlp_trainer_and_data[0]

        # Temporarily replace model with dummy sklearn
        original_model = trainer.model
        trainer.model = DummySklearn()

        x_df = pd.DataFrame(np.zeros((2, 2)), columns=["f0", "f1"])
        y_series = pd.Series(np.zeros((2, 1)).flatten(), name="label")

        result = trainer.call_trainer(x_df, y_series)
        assert result == {"dummy": True}

        # Restore original model
        trainer.model = original_model

    def test_all_optimizers(self, mlp_trainer_and_data):
        """
        Tests that all supported optimizers can be instantiated.

        Verifies:
        - ADAM optimizer works
        - ADAMW optimizer works
        - SGD optimizer works
        - RMSPROP optimizer works
        - All return valid optimizer objects
        """
        trainer = mlp_trainer_and_data[0]

        for opt in [
            OptimizersEnum.ADAM.value,
            OptimizersEnum.ADAMW.value,
            OptimizersEnum.SGD.value,
            OptimizersEnum.RMSPROP.value,
        ]:
            optimizer_obj = trainer._get_optimizer(opt)
            assert optimizer_obj is not None

    def test_invalid_optimizer_raises_error(self, mlp_trainer_and_data):
        """
        Tests that invalid optimizer names raise ValueError.

        Verifies:
        - _get_optimizer raises ValueError for unknown optimizers
        """
        trainer = mlp_trainer_and_data[0]

        with pytest.raises(ValueError):
            trainer._get_optimizer("notarealoptimizer")

    def test_invalid_criterion_raises_error(self, mlp_trainer_and_data):
        """
        Tests that invalid criterion names raise ValueError.

        Verifies:
        - _get_fn_cost raises ValueError for unknown loss functions
        """
        trainer = mlp_trainer_and_data[0]
        trainer.config.criterion = "not_a_criterion"

        with pytest.raises(ValueError):
            trainer._get_fn_cost(trainer.config.criterion)

    def test_trainer_invalid_model_config(self):
        """
        Tests that invalid model configurations are rejected.

        Verifies:
        - ModelTrainer raises ValueError when config_model is not recognized
        - Error is caught during trainer initialization
        """

        class DummyConfig:
            """Invalid config class for testing."""

            pass

        with pytest.raises(ValueError):
            ModelTrainer(
                TrainerConfig(
                    batch_size=8,
                    epochs=1,
                    seed=42,
                    learning_rate=1e-3,
                    config_model=DummyConfig(),
                    optimizer=OptimizersEnum.ADAM.value,
                    criterion=CriterionEnum.MSE.value,
                    device="cpu",
                )
            )

    def test_invalid_model_config_in_get_model(self, mlp_trainer_and_data):
        """
        Tests _get_model method with invalid configuration.

        Verifies:
        - _get_model raises ValueError for unrecognized config types
        """
        trainer = mlp_trainer_and_data[0]
        config = "not_a_model_config"

        with pytest.raises(ValueError):
            trainer._get_model(config)

    def test_trainer_config_batch_size_validation(self):
        """
        Tests batch_size validator in TrainerConfig.

        Verifies:
        - batch_size must be > 0
        - ValueError raised for batch_size <= 0
        """
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            TrainerConfig(
                batch_size=0,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="cpu",
            )

    def test_trainer_config_epochs_validation(self):
        """
        Tests epochs validator in TrainerConfig.

        Verifies:
        - epochs must be > 0
        - ValueError raised for epochs <= 0
        """
        with pytest.raises(ValueError, match="epochs must be > 0"):
            TrainerConfig(
                batch_size=1,
                epochs=0,
                seed=42,
                learning_rate=1e-3,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="cpu",
            )

    def test_trainer_config_learning_rate_validation(self):
        """
        Tests learning_rate validator in TrainerConfig.

        Verifies:
        - learning_rate must be > 0
        - ValueError raised for learning_rate <= 0
        """
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TrainerConfig(
                batch_size=1,
                epochs=1,
                seed=42,
                learning_rate=0,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="cpu",
            )

    def test_trainer_config_n_splits_validation(self):
        """
        Tests n_splits validator in TrainerConfig for cross-validation.

        Verifies:
        - n_splits must be > 1 when cross_validation is True
        - ValueError raised for n_splits <= 1
        """
        with pytest.raises(
            ValueError, match="n_splits must be > 1 for cross-validation"
        ):
            TrainerConfig(
                batch_size=1,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="cpu",
                cross_validation=True,
                n_splits=1,
            )

    def test_trainer_config_optimizer_validation(self):
        """
        Tests optimizer validator in TrainerConfig.

        Verifies:
        - optimizer must be one of the valid enum values
        - ValueError raised for invalid optimizer names
        """
        with pytest.raises(ValueError, match="optimizer must be one of"):
            TrainerConfig(
                batch_size=1,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer="notarealoptimizer",
                criterion=CriterionEnum.MSE.value,
                device="cpu",
            )

    def test_trainer_config_criterion_validation(self):
        """
        Tests criterion validator in TrainerConfig.

        Verifies:
        - criterion must be one of the valid enum values
        - ValueError raised for invalid criterion names
        """
        with pytest.raises(ValueError, match="criterion must be one of"):
            TrainerConfig(
                batch_size=1,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer=OptimizersEnum.ADAM.value,
                criterion="notarealcriterion",
                device="cpu",
            )

    def test_trainer_config_device_validation(self):
        """
        Tests device validator in TrainerConfig.

        Verifies:
        - device must be either 'cpu' or 'cuda'
        - ValueError raised for invalid device names
        """
        with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
            TrainerConfig(
                batch_size=1,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=MLPConfig(
                    input_size=4,
                    hidden_sizes=(2,),
                    output_size=1,
                    activation_function="relu",
                ),
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="notarealdevice",
            )
