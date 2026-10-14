==================
Feature Extraction
==================

Feature extraction converts continuous sensor signals into structured feature spaces using sliding windows.

Windowing
=========

Time-series windowing segments long sensor signals into fixed-width frames defined by ``WindowingConfig``.

Feature Adapters
================

ThreeWToolkit supports multi-domain feature extraction using adapters:

* **StatisticalConfig**: Standard time-domain statistics (mean, variance, skewness, kurtosis).
* **EWStatisticalConfig**: Exponentially weighted time-domain statistics.
* **WaveletConfig**: Frequency and time-frequency domain decomposition via wavelets.

Combining Feature Extractor Steps
=================================

Multiple extractors can be combined in parallel using ``ConcatFeatureAdapterConfig``:

.. code-block:: python

   from ThreeWToolkit.feature_extraction import (
       WindowingConfig,
       SequentialFeatureAdapterConfig,
       ConcatFeatureAdapterConfig,
       StatisticalConfig,
       EWStatisticalConfig,
       WaveletConfig,
   )

   feature_config = SequentialFeatureAdapterConfig(
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
   )