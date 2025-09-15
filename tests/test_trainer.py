import pytest
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from ThreeWToolkit.trainer.trainer import ModelTrainer, TrainerConfig
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.models.sklearn_models import SklearnModelsConfig
from ThreeWToolkit.core.enums import OptimizersEnum, CriterionEnum, ModelTypeEnum


@pytest.fixture
def mlp_trainer_and_data():
    np.random.seed(42)
    torch.manual_seed(42)
    import pandas as pd

    x = np.random.rand(50, 8).astype(np.float32)
    # Create a binary label for demonstration, as int
    y = (np.random.rand(50) > 0.5).astype(int)
    x_df = pd.DataFrame(x, columns=[f"f{i}" for i in range(8)])
    y_series = pd.Series(y, name="label")
    config = MLPConfig(
        input_size=8,
        hidden_sizes=(16, 8),
        output_size=1,
        activation_function="relu",
        regularization=None,
        random_seed=42,
    )
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
def sklearn_trainer_and_data():
    from sklearn.datasets import make_classification

    x, y = make_classification(n_samples=40, n_features=6, n_classes=2, random_state=42)
    config = SklearnModelsConfig(
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42, model_params={}
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


def test_mlp_trainer_train_and_history(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.train(x, y)
    assert hasattr(trainer, "history")
    assert isinstance(trainer.history, list)
    assert "train_loss" in trainer.history[0]
    assert len(trainer.history[0]["train_loss"]) == trainer.epochs


def test_mlp_trainer_predict(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.train(x, y)
    preds = trainer.predict(x)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == x.shape[0]


def test_mlp_trainer_test(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.train(x, y)
    test_loss, test_metrics = trainer.test(x, y, metrics=[mean_squared_error])
    assert isinstance(test_loss, float)
    assert "mean_squared_error" in test_metrics


def test_mlp_trainer_save_and_load(tmp_path, mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.train(x, y)
    save_path = tmp_path / "mlp_model.pth"
    trainer.save(save_path)
    loaded_model = trainer.load(save_path)
    assert loaded_model is not None


def test_mlp_trainer_cross_validation(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.cross_validation = True
    trainer.n_splits = 3
    trainer.train(x, y)
    assert isinstance(trainer.history, list)
    assert len(trainer.history) == 3
    for fold_hist in trainer.history:
        assert "train_loss" in fold_hist
        assert len(fold_hist["train_loss"]) == trainer.epochs


def test_sklearn_trainer_train_and_predict(sklearn_trainer_and_data):
    trainer, x, y = sklearn_trainer_and_data
    trainer.train(x, y)
    preds = trainer.predict(x)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == x.shape[0]


def test_sklearn_trainer_test(sklearn_trainer_and_data):
    trainer, x, y = sklearn_trainer_and_data
    trainer.train(x, y)
    from ThreeWToolkit.metrics import _classification

    test_metrics = trainer.test(x, y, metrics=[_classification.accuracy_score])
    assert "accuracy_score" in test_metrics


def test_sklearn_trainer_cross_validation(sklearn_trainer_and_data):
    trainer, x, y = sklearn_trainer_and_data
    trainer.cross_validation = True
    trainer.n_splits = 2
    trainer.train(x, y)
    assert isinstance(trainer.history, list)
    assert len(trainer.history) == 2


def test_trainer_invalid_model_config():
    class DummyConfig:
        pass

    # Should raise ValueError in ModelTrainer, not TrainerConfig
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


def test_trainer_crossval_default_n_splits():
    # Explicitly test __init__ branch: cross_validation=True, n_splits omitted
    from sklearn.datasets import make_classification

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
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42, model_params={}
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
        # n_splits omitted
    )
    trainer = ModelTrainer(trainer_config)
    assert trainer.n_splits == 5
    assert trainer.metrics == metrics


def test_trainer_unknown_model_config_explicit():
    # Explicitly test the case _ branch in _get_model
    class NotAConfig:
        pass

    config = NotAConfig()
    with pytest.raises(ValueError):
        ModelTrainer(
            TrainerConfig(
                batch_size=2,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=config,
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="cpu",
            )
        )

    # Use a valid config for the rest of the test
    valid_config = MLPConfig(
        input_size=4,
        hidden_sizes=(2,),
        output_size=1,
        activation_function="relu",
        regularization=None,
        random_seed=42,
    )
    # Covers all criterion branches (cross_entropy, binary_cross_entropy, mse, mae, else)
    for crit in [
        CriterionEnum.CROSS_ENTROPY.value,
        CriterionEnum.BINARY_CROSS_ENTROPY.value,
        CriterionEnum.MSE.value,
        CriterionEnum.MAE.value,
    ]:
        trainer_config = TrainerConfig(
            batch_size=2,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=valid_config,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=crit,
            device="cpu",
        )
        trainer = ModelTrainer(trainer_config)
        fn = trainer._get_fn_cost(crit)
        assert fn is not None
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="criterion must be one of"):
        TrainerConfig(
            batch_size=2,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=valid_config,
            optimizer=OptimizersEnum.ADAM.value,
            criterion="notarealcriterion",
            device="cpu",
        )

    # Covers else branch in predict (sklearn)
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=10,
        n_features=3,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    config = SklearnModelsConfig(
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42, model_params={}
    )
    trainer_config = TrainerConfig(
        batch_size=2,
        epochs=1,
        seed=42,
        learning_rate=1e-3,
        config_model=config,
        optimizer=OptimizersEnum.ADAM.value,
        criterion=CriterionEnum.MSE.value,
        device="cpu",
    )
    trainer = ModelTrainer(trainer_config)
    trainer.train(x, y)
    preds = trainer.predict(x)
    assert isinstance(preds, np.ndarray)

    # Covers else branch in test (sklearn)
    from ThreeWToolkit.metrics import _classification

    x, y = make_classification(
        n_samples=10,
        n_features=3,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    config = SklearnModelsConfig(
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42, model_params={}
    )
    trainer_config = TrainerConfig(
        batch_size=2,
        epochs=1,
        seed=42,
        learning_rate=1e-3,
        config_model=config,
        optimizer=OptimizersEnum.ADAM.value,
        criterion=CriterionEnum.MSE.value,
        device="cpu",
    )
    trainer = ModelTrainer(trainer_config)
    trainer.train(x, y)
    metrics = [_classification.accuracy_score]
    result = trainer.test(x, y, metrics=metrics)
    assert "accuracy_score" in result


def test_trainer_all_missing_branches(mlp_trainer_and_data):
    # Cover TrainerConfig batch_size validator (<=0)
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        TrainerConfig(
            batch_size=0,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
        )

    # Cover TrainerConfig epochs validator (<=0)
    with pytest.raises(ValueError, match="epochs must be > 0"):
        TrainerConfig(
            batch_size=1,
            epochs=0,
            seed=42,
            learning_rate=1e-3,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
        )

    # Cover TrainerConfig learning_rate validator (<=0)
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        TrainerConfig(
            batch_size=1,
            epochs=1,
            seed=42,
            learning_rate=0,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
        )

    # Cover TrainerConfig n_splits validator (<=1 with cross_validation)
    with pytest.raises(ValueError, match="n_splits must be > 1 for cross-validation"):
        TrainerConfig(
            batch_size=1,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="cpu",
            cross_validation=True,
            n_splits=1,
        )

    # Cover TrainerConfig optimizer validator (invalid value)
    with pytest.raises(ValueError, match="optimizer must be one of"):
        TrainerConfig(
            batch_size=1,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer="notarealoptimizer",
            criterion=CriterionEnum.MSE.value,
            device="cpu",
        )

    # Cover TrainerConfig criterion validator (invalid value)
    with pytest.raises(ValueError, match="criterion must be one of"):
        TrainerConfig(
            batch_size=1,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer=OptimizersEnum.ADAM.value,
            criterion="notarealcriterion",
            device="cpu",
        )

    # Cover TrainerConfig device validator (invalid value)
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        TrainerConfig(
            batch_size=1,
            epochs=1,
            seed=42,
            learning_rate=1e-3,
            config_model=mlp_trainer_and_data[0].config.config_model,
            optimizer=OptimizersEnum.ADAM.value,
            criterion=CriterionEnum.MSE.value,
            device="notarealdevice",
        )

    # Cover else branch in call_trainer (sklearn)
    class DummySklearn:
        def fit(self, x, y, **kwargs):
            return {"dummy": True}

    trainer = mlp_trainer_and_data[0]
    import pandas as pd

    # Save original model
    original_model = trainer.model
    trainer.model = DummySklearn()
    # Use DataFrame/Series for compatibility
    x_df = pd.DataFrame(np.zeros((2, 2)), columns=["f0", "f1"])
    y_series = pd.Series(np.zeros((2, 1)).flatten(), name="label")
    result = trainer.call_trainer(x_df, y_series)
    assert result == {"dummy": True}

    # Restore to a valid MLP model for error branch tests
    trainer.model = original_model

    # Cover else branch in _get_fn_cost
    with pytest.raises(ValueError):
        trainer._get_fn_cost("notarealcriterion")

    # Cover else branch in _get_optimizer
    with pytest.raises(ValueError):
        trainer._get_optimizer("notarealoptimizer")

    # Cover else branch in _get_model
    with pytest.raises(ValueError):
        trainer._get_model("not_a_model_config")
    # Covers default n_splits branch
    config = SklearnModelsConfig(
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION, random_seed=42, model_params={}
    )
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
        # n_splits omitted
    )
    trainer = ModelTrainer(trainer_config)
    assert trainer.n_splits == 5

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
        n_splits=5,
    )
    trainer = ModelTrainer(trainer_config)
    assert trainer.n_splits == 5

    # Covers all optimizer branches and else
    valid_config = MLPConfig(
        input_size=4,
        hidden_sizes=(2,),
        output_size=1,
        activation_function="relu",
        regularization=None,
        random_seed=42,
    )
    trainer_config = TrainerConfig(
        batch_size=2,
        epochs=1,
        seed=42,
        learning_rate=1e-3,
        config_model=valid_config,
        optimizer=OptimizersEnum.ADAM.value,
        criterion=CriterionEnum.MSE.value,
        device="cpu",
    )
    trainer = ModelTrainer(trainer_config)
    for opt in [
        OptimizersEnum.ADAM.value,
        OptimizersEnum.ADAMW.value,
        OptimizersEnum.SGD.value,
        OptimizersEnum.RMSPROP.value,
    ]:
        optimizer_obj = trainer._get_optimizer(opt)
        assert optimizer_obj is not None
    with pytest.raises(ValueError):
        trainer._get_optimizer("notarealoptimizer")

    # Covers unknown model config branch
    class DummyConfig:
        pass

    dummy = DummyConfig()
    with pytest.raises(ValueError):
        ModelTrainer(
            TrainerConfig(
                batch_size=2,
                epochs=1,
                seed=42,
                learning_rate=1e-3,
                config_model=dummy,
                optimizer=OptimizersEnum.ADAM.value,
                criterion=CriterionEnum.MSE.value,
                device="cpu",
            )
        )
    # Covers else branch in _get_optimizer for MLP
    trainer, x, y = mlp_trainer_and_data
    config = "not_a_model_config"
    with pytest.raises(ValueError):
        trainer._get_model(config)


def test_trainer_invalid_criterion(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.config.criterion = "not_a_criterion"
    with pytest.raises(ValueError):
        trainer._get_fn_cost(trainer.config.criterion)


def test_trainer_metrics_assignment():
    # Covers self.metrics = config.metrics
    config = MLPConfig(
        input_size=4,
        hidden_sizes=(2,),
        output_size=1,
        activation_function="relu",
        regularization=None,
        random_seed=42,
    )
    trainer_config = TrainerConfig(
        batch_size=2,
        epochs=1,
        seed=42,
        learning_rate=1e-3,
        config_model=config,
        optimizer=OptimizersEnum.ADAM.value,
        criterion=CriterionEnum.MSE.value,
        device="cpu",
        metrics=[lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()],
    )
    trainer = ModelTrainer(trainer_config)
    assert trainer.metrics is not None


def test_trainer_train_with_val(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.cross_validation = False
    # Use a split for validation
    x_val = x.iloc[:10]
    y_val = y.iloc[:10]
    x_train = x.iloc[10:]
    y_train = y.iloc[10:]
    trainer.train(x_train, y_train, x_val=x_val, y_val=y_val)
    assert isinstance(trainer.history, list)
    assert len(trainer.history) == 1
    assert "train_loss" in trainer.history[0]
    assert len(trainer.history[0]["train_loss"]) == trainer.epochs


def test_call_trainer_with_val_loader(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.cross_validation = False
    # Use a split for validation
    x_val = x.iloc[:10]
    y_val = y.iloc[:10]
    x_train = x.iloc[10:]
    y_train = y.iloc[10:]
    # Directly call call_trainer to ensure val_loader is created
    result = trainer.call_trainer(x_train, y_train, x_val=x_val, y_val=y_val)
    assert result is not None
    assert "train_loss" in result


def test_call_trainer_without_val_loader(mlp_trainer_and_data):
    trainer, x, y = mlp_trainer_and_data
    trainer.cross_validation = False
    # Use a split for validation
    x_train = x.iloc[10:]
    y_train = y.iloc[10:]
    # Directly call call_trainer to ensure val_loader is created
    result = trainer.call_trainer(x_train, y_train, x_val=None, y_val=None)
    assert result is not None
    assert "train_loss" in result


def test_trainer_save_and_load_sklearn(tmp_path, sklearn_trainer_and_data):
    trainer, x, y = sklearn_trainer_and_data
    trainer.train(x, y)
    save_path = tmp_path / "sklearn_model.pkl"
    trainer.save(save_path)
    loaded_model = trainer.load(save_path)
    assert loaded_model is not None
