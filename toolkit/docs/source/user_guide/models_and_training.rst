===================
Models and Training
===================

ThreeWToolkit provides wrappers and trainers to run models using PyTorch or Scikit-Learn.

Model Architecture Specs
========================

Neural network architectures are defined via declarative configuration classes such as ``MLPConfig``:

.. code-block:: python

   from ThreeWToolkit.models import MLPConfig

   model_config = MLPConfig(
       hidden_sizes=(64, 32),
       output_size=5,
   )

Training Execution
==================

Models are trained using dedicated trainer configurations like ``TorchTrainerConfig``:

.. code-block:: python

   from ThreeWToolkit.trainer import TorchTrainerConfig

   trainer_config = TorchTrainerConfig(
       config_model=model_config,
       seed=42,
       epochs=50,
       batch_size=32,
       learning_rate=1e-4,
       device="cpu",
   )