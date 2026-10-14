============
Architecture
============

The **3W Toolkit** is designed around a modular, highly extensible, and configuration-driven software architecture. Its core design philosophy enforces a strict separation between component specifications and operational execution, allowing modules to be updated, replaced, or expanded independently.

Core Principles
===============

* **Separation of Concerns**: Each module (data loading, preprocessing, feature engineering, model definition, training, evaluation) operates independently with minimal coupling.
* **Configuration-Driven Execution**: Workflows and components are specified using dedicated configuration objects (Pydantic-backed), ensuring 100% reproducibility.
* **Standardized Data Contracts**: Modular communication relies on lightweight, immutable data containers rather than raw dictionaries or arbitrary objects.

Abstraction Layers
==================

The system architecture is organized into two primary abstraction layers:

1. Core Layer (Specification & Contracts)
-----------------------------------------

The **Core** layer establishes the fundamental contracts, base abstractions, and configuration blueprints. It does not execute complex business logic directly.

* **Base Interfaces**: Abstract base classes like ``BaseDataset``, ``BasePreprocessing``, ``BaseFeatureExtractor``, ``BaseModels``, ``BaseTrainer``, and ``BasePipeline`` define consistent interfaces across the system.
* **Data Containers**: Lightweight structures (e.g., ``DatasetOutputs``, ``TrainingResult``, ``PredictionResult``, ``AssessmentOutput``) standardize inputs/outputs between stages, ensuring traceability.
* **Configuration Blueprints**: Classes ending in ``Config`` hold operational hyperparameters and handle schema validation prior to execution.

2. Application Layer (Concrete Implementations)
-----------------------------------------------

The **Application** layer implements the functional operational logic. Components are constructed from their corresponding ``Config`` blueprints via the ``.build()`` dynamic instantiation pattern:

.. code-block:: text

   +--------------------------+
   |   PipelineConfig (Core)  |
   +--------------------------+
                |
                | .build()
                v
   +--------------------------+
   |   Pipeline (Application) |
   +--------------------------+

Component Overview
==================

Data Management & Processing
----------------------------

* **Dataset Handling**: Classes like ``ParquetDataset`` manage memory-efficient ingestion and parsing of telemetry files.
* **Data Transformation**: Non-destructive transformations such as ``Normalize`` (preprocessing) or ``Windowing`` (feature extraction) produce new dataset representations without altering raw underlying streams.

Model & Training Isolation
--------------------------

To support heterogeneous operational demands, model definitions are kept distinct from training logic:

* **Model Encodings**: Unified wrappers support traditional machine learning via ``SklearnModels`` and deep learning architectures through ``TorchModels``.
* **Dedicated Trainers**: Framework-specific execution engines (e.g., ``SklearnTrainer``, ``TorchTrainer``) inherit from ``BaseTrainer`` to execute training pipelines independently of model definitions.

Evaluation & Orchestration
--------------------------

* **Model Assessment**: Standardized evaluation is performed by ``ModelAssessment``, generating model-agnostic performance metrics and visualization structures.
* **Pipeline Orchestration**: The higher-level ``Pipeline`` class encapsulates end-to-end workflows—from raw sensor ingestion to final metric assessment—enabling reproducible multi-stage experiment execution.

Architecture Overview Diagram
=============================

Below is the class and interaction schema illustrating the main components of the **3W Toolkit**:

.. image:: ../../../../paper/assets/diagrama_classes_joss-background.drawio.svg
   :align: center
   :alt: ThreeWToolkit Class Diagram
   :width: 100%

---