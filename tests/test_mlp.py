import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pydantic import BaseModel
from typing import Tuple

from ThreeWToolkit.models.mlp import MLP, ActivationFunction, MLPTrainer, MLPConfig 

@pytest.fixture
def trainer_setup():
    """Prepara um cenário de teste completo usando as classes locais."""
    num_samples = 100
    input_size = 10
    features_data = pd.DataFrame(
        np.random.rand(num_samples, input_size),
        columns=[f'feature_{i}' for i in range(input_size)]
    )
    target_data = pd.Series(np.random.randint(0, 2, num_samples), name='target')

    X_tensor = torch.tensor(features_data.values, dtype=torch.float32)
    y_tensor = torch.tensor(target_data.values, dtype=torch.long)
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # Usando a MLPConfig local, sem 'model_type'
    config = MLPConfig(
        input_size=input_size,
        hidden_sizes=(32, 16),
        output_size=2,
        activation_function=ActivationFunction.RELU
    )
    
    # MLPTrainer usará as classes MLP e MLPConfig definidas aqui no teste
    trainer = MLPTrainer(
        train_dataset=full_dataset,
        test_dataset=full_dataset,
        config=config,
        batch_size=16,
        lr=1e-3,
        nfolds=2,
        seed=42
    )
    
    # Sobrescreve o método do trainer para que ele use o nosso MLP local
    trainer._get_model = lambda: MLP(config).to(trainer.device)
    
    return {
        "trainer": trainer,
        "config": config,
        "dataframe": features_data.join(target_data),
        "dataset": full_dataset,
    }

# --- Testes (Agora funcionam com as classes locais) ---

class TestMLP:
    @pytest.mark.parametrize("activation_enum, expected_type", [
        (ActivationFunction.RELU, nn.ReLU),
        (ActivationFunction.SIGMOID, nn.Sigmoid),
        (ActivationFunction.TANH, nn.Tanh)
    ])
    def test_mlp_activation_function(self, activation_enum, expected_type):
        config = MLPConfig(
            input_size=10,
            hidden_sizes=(8,),
            output_size=2,
            activation_function=activation_enum
        )
        model = MLP(config)
        assert isinstance(model.model[1], expected_type)

    def test_init_type_error(self):
        with pytest.raises(TypeError, match="Expected MLPConfig."):
            MLP(config={'input_size': 10})

class TestMLPTrainer:
    def test_trainer_initialization(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        assert trainer.batch_size == 16
        assert isinstance(trainer.config, MLPConfig)

    def test_train_loop(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        trainer.train(epochs=2)
        assert len(trainer.models) == trainer.nfolds
        assert len(trainer.history['train_loss']) == 2

    def test_evaluate(self, trainer_setup):
        trainer = trainer_setup["trainer"]
        trainer.train(epochs=1)
        model_to_eval = trainer.models[0]
        test_loader = trainer.create_dataloader(trainer.test_dataset, shuffle=False)
        preds, accuracy = trainer.evaluate(model_to_eval, test_loader)
        assert isinstance(preds, np.ndarray)
        assert 0.0 <= accuracy <= 1.0

a = TestMLP()
a.test_init_type_error()