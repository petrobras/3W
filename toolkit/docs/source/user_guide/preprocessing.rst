=============
Preprocessing
=============

Signal preprocessing transforms raw time-series data into clean, normalized formats suitable for feature extraction and model training.

Available Preprocessors
=======================

* **CleanSignalsConfig**: Filters out invalid channels based on missing data thresholds.
* **FillLabelsConfig**: Imputes or forward-fills missing target class labels across time windows.
* **RemapClassConfig**: Re-maps original dataset class IDs to custom target indices.
* **NormalizeConfig**: Applies scaling (e.g., MinMax or Z-Score) across sensor channels.
* **ImputeMissingConfig**: Handles missing value interpolation across continuous time-series.

Chaining Preprocessors
======================

Preprocessors are combined sequentially using ``SequentialPreprocessingAdapterConfig``:

.. code-block:: python

   from ThreeWToolkit.preprocessing import (
       SequentialPreprocessingAdapterConfig,
       CleanSignalsConfig,
       NormalizeConfig,
       ImputeMissingConfig,
   )

   preprocessor_config = SequentialPreprocessingAdapterConfig(
       steps=[
           CleanSignalsConfig(missing_column_threshold=0.65),
           NormalizeConfig(),
           ImputeMissingConfig(),
       ]
   )