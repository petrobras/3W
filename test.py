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
from ThreeWToolkit.assessment import ModelAssessmentConfig

path = "dataset"
pipeline = PipelineConfig(
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
).build()

results = pipeline.run()

history = results.training_result.history

plt.figure(figsize=(10, 5))
plt.plot(history.train_loss, label="Train Loss")

plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()