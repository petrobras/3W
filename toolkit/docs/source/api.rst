=============
API Reference
=============

Core Layer
==========

.. currentmodule:: ThreeWToolkit.core

.. autosummary::
   :toctree: api
   :nosignatures:

   BaseDataset
   BasePreprocessing
   BaseFeatureExtractor
   BaseModels
   BaseTrainer
   BasePipeline
   DatasetOutputs
   TrainingResult
   PredictionResult

Data & Preprocessing
====================

.. currentmodule:: ThreeWToolkit.preprocessing

.. autosummary::
   :toctree: api
   :nosignatures:

   CleanSignals
   ImputeMissing
   Normalize
   RenameColumns

Feature Extraction
==================

.. currentmodule:: ThreeWToolkit.feature_extraction

.. autosummary::
   :toctree: api
   :nosignatures:

   Windowing
   StatisticalFeatures
   WaveletFeatures

Models & Training
=================

.. currentmodule:: ThreeWToolkit.models

.. autosummary::
   :toctree: api
   :nosignatures:

   SklearnModels
   TorchModels

Assessment & Evaluation
=======================

.. currentmodule:: ThreeWToolkit.assessment

.. autosummary::
   :toctree: api
   :nosignatures:

   ModelAssessment
   AssessmentVisualization
