==========
Quickstart
==========

This guide walks you through building, executing, and visualizing a complete machine learning pipeline for oil well fault detection using **ThreeWToolkit**.

Architecture Pattern
====================

ThreeWToolkit follows a clear separation between **Configuration** and **Execution**:

* **Core Layer (`Config` classes)**: You define the entire declarative blueprint of your pipeline (dataset targets, signal cleaning thresholds, feature extraction techniques, neural network topology, and trainer hyper-parameters).
* **Application Layer (`.build()`)**: The ``PipelineConfig.build()`` factory parses the configuration, instantiates all operational transformers/adapters, and returns an executable ``Pipeline`` instance.

End-to-End Pipeline Example
===========================

The example below demonstrates how to configure a multi-stage workflow: ingesting Parquet data, chaining preprocessors, extracting statistical/wavelet features, training a PyTorch MLP model, and plotting the loss curve.

.. code-block:: python

   import matplotlib.pyplot as plt

   from ThreeWToolkit.pipeline import PipelineConfig
   from ThreeWToolkit.dataset import ParquetDatasetConfig, TransformConfig
   from ThreeWToolkit.preprocessing import (
       SequentialPreprocessingAdapterConfig,
       CleanSignalsConfig,
       FillLabelsConfig,
       RemapClassConfig,
       NormalizeConfig,
       ImputeMissingConfig,
   )
   from ThreeWToolkit.feature_extraction import (
       WindowingConfig,
       SequentialFeatureAdapterConfig,
       ConcatFeatureAdapterConfig,
       StatisticalConfig,
       EWStatisticalConfig,
       WaveletConfig,
   )
   from ThreeWToolkit.models import MLPConfig
   from ThreeWToolkit.trainer import TorchTrainerConfig

   # 1. Path to local or downloaded 3W dataset directory
   path = "../../dataset"

   # 2. Define the complete declarative pipeline configuration
   pipeline_config = PipelineConfig(
       train_dataset_config=ParquetDatasetConfig(
           path=path,
           version="2.0.0",
           target_column="class",
           target_class=[1, 2],
           force_download=False,
           event_type=["real"],
       ),
       test_dataset_config=ParquetDatasetConfig(
           path=path,
           version="2.0.0",
           target_column="class",
           target_class=[1, 2],
           force_download=False,
           event_type=["drawn"],
       ),
       trainer_config=TorchTrainerConfig(
           config_model=MLPConfig(
               hidden_sizes=(64, 32),
               output_size=5,
           ),
           seed=42,
           epochs=50,
           batch_size=32,
           learning_rate=1e-4,
           device="cpu",
       ),
       pre_transform_config=TransformConfig(
           pre_processing=SequentialPreprocessingAdapterConfig(
               steps=[
                   CleanSignalsConfig(missing_column_threshold=0.65),
                   FillLabelsConfig(),
                   RemapClassConfig(),
               ],
           )
       ),
       transform_config=TransformConfig(
           pre_processing=SequentialPreprocessingAdapterConfig(
               steps=[
                   NormalizeConfig(),
                   ImputeMissingConfig(),
               ]
           ),
           feature_extraction=SequentialFeatureAdapterConfig(
               steps=[
                   WindowingConfig(),
                   ConcatFeatureAdapterConfig(
                       steps=[
                           StatisticalConfig(),
                           EWStatisticalConfig(),
                           WaveletConfig(),
                       ]
                   ),
               ]
           ),
       ),
   )

   # 3. Build operational pipeline objects from config
   pipeline = pipeline_config.build()

   # 4. Run the pipeline (Data ingestion -> Processing -> Feature extraction -> Model training)
   results = pipeline.run()

   # 5. Extract training history and plot loss curve
   history = results.training_result.history

   plt.figure(figsize=(10, 5))
   plt.plot(history.train_loss, label="Train Loss")
   plt.title("Training History")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend()
   plt.show()

Next Steps
==========

* **3W Dataset Structure**: Learn more about the raw sensor time-series formats and class definitions in the :doc:`user_guide/dataset`.
* **Preprocessing & Features**: Discover all available signal transformers in the :doc:`user_guide/preprocessing` and :doc:`user_guide/feature_extraction`.
* **API Reference**: Inspect module signatures directly in the :doc:`api`.