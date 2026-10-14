=======
Dataset
=======

The **3W Dataset** serves as the standard reference dataset for **ThreeWToolkit**. It consists of multivariate time-series data acquired from real and simulated oil wells operated by Petrobras.

Dataset Structure
=================

Data is organized by operational event types (e.g., real, simulated, or hand-drawn) and categorized into different fault classes such as severe slugging, hydrates, and pump failures.

Configuration & Ingestion
=========================

Dataset access is managed through declarative configuration classes, such as ``ParquetDatasetConfig``:

.. code-block:: python

   from ThreeWToolkit.dataset import ParquetDatasetConfig

   dataset_config = ParquetDatasetConfig(
       path="../../dataset",
       version="2.0.0",
       target_column="class",
       target_class=[1, 2],
       force_download=False,
       event_type=["real"],
   )

For complete details on column specifications and raw storage formats, refer to the official repository's `3W_DATASET_STRUCTURE.md <https://github.com/petrobras/3W/blob/main/3W_DATASET_STRUCTURE.md>`_.