import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from ThreeWToolkit.models.mlp import MLP, ActivationFunction, MLPTrainer, MLPConfig, LabeledSubset


@pytest.fixture
def trainer_setup():
    """
    Pytest fixture to set up a standard MLPTrainer and data for testing.
    """
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


class TestLabeledSubset:
    def test_init_value_error(self):
        """
        Tests if LabeledSubset raises ValueError for inputs of different lengths.
        """
        samples = torch.rand(10, 2)
        labels = torch.rand(9)
        with pytest.raises(
            ValueError, match="Samples and labels must have the same length."
        ):
            LabeledSubset(samples, labels)

class TestMLP:
    @pytest.mark.parametrize(
        "activation_enum, expected_type",
        [
            (ActivationFunction.RELU, nn.ReLU),
            (ActivationFunction.SIGMOID, nn.Sigmoid),
            (ActivationFunction.TANH, nn.Tanh),
        ],
    )
    def test_mlp_activation_function(self,
        activation_enum,
        expected_type,
    ):
        """
        Tests that the MLP model is correctly constructed with various activation functions.
        """
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
        """
        Tests that initializing an MLP with an invalid config (dict) raises an AttributeError.
        """
        with pytest.raises(
            AttributeError, match="'dict' object has no attribute 'input_size'"
        ):
            MLP(config={"input_size": 10})  # type: ignore


class TestMLPTrainer:
    def test_trainer_initialization(self, trainer_setup):
        """
        Tests the basic initialization attributes of the MLPTrainer.
        """
        trainer = trainer_setup["trainer"]
        assert trainer.batch_size == 16    
        assert isinstance(trainer.config, MLPConfig)

    def test_train_loop(self, trainer_setup):
        """
        Tests that the main training loop runs and populates the history dictionary.
        """
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        epochs = 3
        trainer.train(x, y, epochs=epochs)
        # Check if the trainer saved the history
        for metrics_all_epochs in trainer.history.values():
            assert len(metrics_all_epochs) == epochs

    def test_evaluate(self, trainer_setup):
        """
        Tests the evaluation of a trained model, checking loss and accuracy scores.
        """
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y, epochs=1)

        # Simulate a test dataset to evaluate the best model with the training samples
        model_to_eval = trainer.best_model
        test_loader = trainer.create_dataloader(x, y, shuffle=False)
        # Evaluate the test dataset with accuracy metric
        avg_loss, metrics = trainer.run_evaluation_epoch(model_to_eval, test_loader)
        assert 0.0 <= avg_loss
        assert 0.0 <= metrics["accuracy_score"] <= 1.0
    
    def test_get_params(self, trainer_setup):
        """
        Tests the get_params method.
        """
        trainer = trainer_setup["trainer"]
        config = trainer_setup["config"]

        params = trainer.get_params()

        # Check that the returned object is a dictionary
        assert isinstance(params, dict)
        # Check that it contains the expected configuration keys and values
        assert params["input_size"] == config.input_size
        assert params["output_size"] == config.output_size
    
    def test_set_params(self, trainer_setup):
        """
        Tests setting valid and invalid parameters.
        """
        trainer = trainer_setup["trainer"]

        # Test setting a valid parameter from the config
        new_seed = 101
        trainer.set_params(random_seed=new_seed)
        assert trainer.config.random_seed == new_seed

        # Test setting an invalid parameter
        with pytest.raises(ValueError, match="Invalid parameter: invalid_key"):
            trainer.set_params(invalid_key=123)

    def test_evaluate_method(self, trainer_setup):
        """
        Directly tests the evaluate method.
        """
        trainer = trainer_setup["trainer"]
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]

        def dummy_metric(y_t, y_p):
            return np.sum(np.array(y_t) == np.array(y_p)) / len(y_t)

        metrics_result = trainer.evaluate(y_pred, y_true, metrics=[dummy_metric])
        assert "dummy_metric" in metrics_result
        assert metrics_result["dummy_metric"] == 0.75

    def test_train_non_sized_input(self, trainer_setup):
        """
        Tests that train raises TypeError for non-Sized datasets.
        """
        trainer = trainer_setup["trainer"]
        # A generator is an iterator, which is not Sized
        x_generator = (i for i in range(10))
        y_dummy = [0] * 10
        with pytest.raises(TypeError, match="Expected Sized Dataset."):
            trainer.train(x_generator, y_dummy)

    def test_predict(self, trainer_setup):
        """
        Tests the predict method after a brief training.
        """
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]
        trainer.train(x, y, epochs=1)

        assert trainer.best_model is not None
        
        # Create a dataloader for prediction
        pred_loader = trainer.create_dataloader(x, y, shuffle=False)
        predictions = trainer.predict(trainer.best_model, pred_loader)
        
        # Assertions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(x)
        # Check if predictions are class indices (0 or 1 in this setup)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_train_raises_type_error_on_invalid_best_model(
        self, trainer_setup, monkeypatch
    ):
        """
        Tests the internal TypeError if the best_model is not an MLP instance.
        This is an edge case that's hard to trigger naturally.
        """
        trainer = trainer_setup["trainer"]
        x = trainer_setup["x_tensor"]
        y = trainer_setup["y_tensor"]

        # Mock model.eval() to return an invalid object (e.g., a string)
        # This will cause best_model['model'] to be a string inside train()
        def mock_eval(self):
            return "not a model"

        # The patch needs to target the MLP class from mlp.py
        monkeypatch.setattr(MLP, "eval", mock_eval)

        with pytest.raises(TypeError, match="is not an MLP or None"):
            trainer.train(x, y, epochs=1)
            
    def test_predict_with_empty_loader(self, trainer_setup):
        """
        Tests that predict returns an empty array for an empty data loader.
        """
        trainer = trainer_setup["trainer"]
        model = trainer._get_model()  # Get a model instance

        # Create empty tensors for the dataloader
        x_empty = torch.empty(0, trainer.config.input_size)
        y_empty = torch.empty(0)
        
        # Create an empty dataloader
        empty_loader = trainer.create_dataloader(x_empty, y_empty, shuffle=False)
        
        predictions = trainer.predict(model, empty_loader)
        
        # Assert that the output is an empty numpy array
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 0
