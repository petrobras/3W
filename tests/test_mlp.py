import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from ThreeWToolkit.models.mlp import MLP, ActivationFunction, MLPTrainer, MLPConfig


# @pytest.fixture
def trainer_setup():
    num_samples = 100
    input_size = 10
    cols = pd.Index(
        [f"feature_{i}" for i in range(input_size)]
    )  # Use Index for columns
    data = np.random.rand(num_samples, input_size)
    features_data = pd.DataFrame(data=data, columns=cols)
    target_data = pd.Series(np.random.randint(0, 2, num_samples), name="target")
    x_tensor = torch.tensor(features_data.values, dtype=torch.float32)
    y_tensor = torch.tensor(target_data.values, dtype=torch.long)
    config = MLPConfig(
        input_size=input_size,
        hidden_sizes=(32, 16),
        output_size=2,
        activation_function=ActivationFunction.RELU,
    )
    trainer = MLPTrainer(
        config=config,
        batch_size=16,
        lr=1e-3,
        nfolds=2,
        seed=42,
    )
    return {
        "x_tensor": x_tensor,
        "y_tensor": y_tensor,
        "config": config,
        "trainer": trainer,
    }


class TestMLP:
    @pytest.mark.parametrize(
        "activation_enum, expected_type",
        [
            (ActivationFunction.RELU, nn.ReLU),
            (ActivationFunction.SIGMOID, nn.Sigmoid),
            (ActivationFunction.TANH, nn.Tanh),
        ],
    )
    def test_mlp_activation_function(
        self,
        activation_enum,
        expected_type,
    ):
        # Creates a config with the activation function
        config = MLPConfig(
            input_size=10,
            hidden_sizes=(8, 3),
            output_size=2,
            activation_function=activation_enum,
        )
        model = MLP(config)
        assert isinstance(model.model[1], expected_type)
        assert isinstance(model.model[3], expected_type)

    def test_init_type_error(self):
        with pytest.raises(
            AttributeError, match="'dict' object has no attribute 'input_size'"
        ):
            MLP(config={"input_size": 10})  # type: ignore


class TestMLPTrainer:
    def test_trainer_initialization(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        assert trainer.batch_size == 16
        assert isinstance(trainer.config, MLPConfig)

    def test_train_loop(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        trainer.train(epochs=2)
        assert len(trainer.models) == trainer.nfolds
        assert len(trainer.history["train_loss"]) == 2

    def test_evaluate(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        trainer.train(epochs=1)
        model_to_eval = trainer.models[0]
        test_loader = trainer.create_dataloader(trainer.test_dataset, shuffle=False)
        preds, accuracy = trainer.evaluate(model_to_eval, test_loader)
        assert isinstance(preds, np.ndarray)
        assert 0.0 <= accuracy <= 1.0


params = trainer_setup()
a = TestMLPTrainer()
a.test_train_loop(params)
x = 123
