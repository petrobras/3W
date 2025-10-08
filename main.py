"""
Main pipeline configuration and execution script.

This script demonstrates a complete machine learning pipeline for time-series
classification using the 3W Toolkit. It includes data loading, preprocessing,
feature extraction, model training, and evaluation.
"""

from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.base_dataset import ParquetDatasetConfig
from ThreeWToolkit.core.base_feature_extractor import EWStatisticalConfig, StatisticalConfig, WaveletConfig
from ThreeWToolkit.core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    RenameColumnsConfig,
    WindowingConfig,
)
from ThreeWToolkit.core.enums import ModelTypeEnum, TaskType
from ThreeWToolkit.models.mlp import MLPConfig
from ThreeWToolkit.models.sklearn_models import SklearnModelsConfig
from ThreeWToolkit.trainer.trainer import TrainerConfig
from ThreeWToolkit.pipeline import Pipeline


if __name__ == "__main__":
    # Alternative model configuration for MLP (commented out for testing)
    # config_model = MLPConfig(
    #     hidden_sizes=(32, 16),
    #     output_size=2,
    #     random_seed=42,
    #     activation_function="relu",
    #     regularization=None,
    # )

    # Configure scikit-learn model
    config_model = SklearnModelsConfig(
        model_type=ModelTypeEnum.LOGISTIC_REGRESSION,
        random_seed=42
    )

    # Build the complete pipeline
    pipeline = Pipeline(
        [
            # --- Dataset Loading ---
            ParquetDatasetConfig(
                path="./data/raw",
                split=None,
                download=False,
                columns=["T-JUS-CKP", "T-MON-CKP"],
                target_column="class",
                target_class=[0, 1]
            ),
            
            # --- Preprocessing Steps ---
            ImputeMissingConfig(strategy="median", columns=["T-JUS-CKP"]),
            NormalizeConfig(norm="l2"),
            WindowingConfig(window_size=10),
            RenameColumnsConfig(columns_map={"T-JUS-CKP": "T-JUS-NOVO"}),  # Alternative preprocessing
            
            # --- Feature Extraction ---
            # StatisticalConfig(),  # Alternative: Basic statistical features
            EWStatisticalConfig(
                selected_features=['ew_mean', 'ew_std', 'ew_skew'],
                overlap=0.5,
                offset=0,
                decay=0.9,  # Exponential decay factor
                eps=1e-8
            ),
            # WaveletConfig(  # Alternative: Wavelet-based features
            #     level=3,
            #     overlap=0.5,
            #     offset=0,
            #     wavelet="haar"
            # ),
            
            # --- Model Training ---
            TrainerConfig(
                optimizer="adam",
                criterion="cross_entropy",
                batch_size=32,
                epochs=20,
                seed=11,
                config_model=config_model,
                learning_rate=0.001,
                cross_validation=True,
                shuffle_train=True,
            ),
            
            # --- Model Evaluation ---
            ModelAssessmentConfig(
                metrics=["balanced_accuracy", "precision", "recall", "f1"],
                task_type=TaskType.CLASSIFICATION,
                export_results=True,
                generate_report=False,
            ),
        ]
    )

    # Execute the complete pipeline
    pipeline.run()