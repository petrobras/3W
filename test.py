import numpy as np
import matplotlib.pyplot as plt

from ThreeWToolkit.dataset import ParquetDatasetConfig, TransformConfig

from ThreeWToolkit.preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    RenameColumnsConfig,
    CleanSignalsConfig,
    FillLabelsConfig,
    RemapClassConfig,
    SequentialPreprocessingAdapterConfig,
)
from ThreeWToolkit.feature_extraction import (
    WindowingConfig,
    StatisticalConfig,
    WaveletConfig,
    EWStatisticalConfig,
    SequentialFeatureAdapterConfig,
    ConcatFeatureAdapterConfig,
)

from ThreeWToolkit.data_visualization import DataVisualization

from ThreeWToolkit.models import MLPConfig

from ThreeWToolkit.trainer import TorchTrainerConfig

from ThreeWToolkit.assessment import ModelAssessmentConfig

dataset_path = "dataset"

# Create and load
ds = ParquetDatasetConfig(path=dataset_path).build()

# Count events
total_events = len(ds)
print(f"Total events: {total_events}")

event_id = 1200
event = ds[event_id]

print(f"Event type: {type(event)}")

print("Available fields:", list(event.__class__.model_fields.keys()))

signal = event.signal

print(f"Signal type: {type(signal)}")
print(f"Signal shape: {np.shape(signal)}")

# Show a small preview
print("Signal preview:")
signal[:5]

label = event.label

print(f"Label type: {type(label)}")
print(f"Label shape: {np.shape(label)}")

print("Unique label values:")
print(np.unique(label))

print("Label preview:")
print(label[5000:5010])

metadata = event.metadata

print("Metadata:")
print(metadata)

ds = ParquetDatasetConfig(path=dataset_path, event_type=["drawn", "simulated"]).build()

print(f"\nFiltered dataset size: {len(ds)}")

ds = ParquetDatasetConfig(path=dataset_path, event_type=["real"]).build()

print(f"\nFiltered dataset size: {len(ds)}")

target_class = [0]

ds = ParquetDatasetConfig(path=dataset_path, target_class=target_class).build()

print(f"Filtered dataset size: {len(ds)}")
print("Unique labels:", np.unique(ds[0].label))

target_class = [2]

ds = ParquetDatasetConfig(path=dataset_path, target_class=target_class).build()

print(f"Filtered dataset size: {len(ds)}")
print("Unique labels:", np.unique(ds[0].label))

target_class = [0, 2]

ds = ParquetDatasetConfig(path=dataset_path, target_class=target_class).build()

print(f"Filtered dataset size: {len(ds)}")

target_class = [2]

ds = ParquetDatasetConfig(
    path=dataset_path, event_type=["real"], target_class=target_class
).build()

print(f"Filtered dataset size: {len(ds)}")
print("Unique labels:", np.unique(ds[0].label))

my_split = [
    "./0/WELL-00008_20170817140222.parquet",
    "./3/SIMULATED_00061.parquet",
    "./4/WELL-00004_20140806090103.parquet",
    "./6/SIMULATED_00117.parquet",
    "./0/WELL-00001_20170201110124.parquet",
    "./5/SIMULATED_00138.parquet",
    "./4/WELL-00005_20170624070158.parquet",
    "./8/SIMULATED_00044.parquet",
    "./5/SIMULATED_00303.parquet",
    "./9/SIMULATED_00028.parquet",
    "./8/SIMULATED_00072.parquet",
    "./7/WELL-00022_20180802233838.parquet",
    "./0/WELL-00003_20170812110000.parquet",
    "./9/SIMULATED_00115.parquet",
    "./1/SIMULATED_00025.parquet",
    "./9/SIMULATED_00065.parquet",
    "./6/SIMULATED_00041.parquet",
    "./5/SIMULATED_00329.parquet",
    "./4/WELL-00004_20141118160016.parquet",
    "./6/SIMULATED_00095.parquet",
]

print(f"Number of files in split: {len(my_split)}")

ds = ParquetDatasetConfig(path=dataset_path, split="list", file_list=my_split).build()

print(f"Filtered dataset size: {len(ds)}")

# Inspect one sample
print("Sample label:", ds[2].label)

event_id = 2
event = ds[event_id]

print("Signal preview:")
event.signal.head()

ds = ParquetDatasetConfig(path=dataset_path).build()

print(f"\nNumber of columns before cleaning: {ds[0].signal.shape[1]}")
print("Columns before cleaning:")
print(ds[0].signal.columns.tolist())

clean_signal = CleanSignalsConfig()

transformer = TransformConfig(pre_processing=clean_signal).build()
transformer.fit(ds)
transformed_ds = transformer.transform(ds)

print(f"\nNumber of columns after cleaning: {transformed_ds[0].signal.shape[1]}")
print("Columns after cleaning:")
print(transformed_ds[0].signal.columns.tolist())

print("Before imputation:")
ds[0].signal.head()

clean_signal = CleanSignalsConfig()
impute_missing = ImputeMissingConfig(strategy="mean")

transformer = TransformConfig(
    pre_processing=SequentialPreprocessingAdapterConfig(
        steps=[clean_signal, impute_missing]
    )
).build()

transformer.fit(ds)
transformed_ds = transformer.transform(ds)

print("\nAfter mean imputation:")
transformed_ds[0].signal.head()

print("Before normalization:")
ds[0].signal["T-TPT"].head(10)

normalize_step = NormalizeConfig(norm="l2")

transformer = TransformConfig(
    pre_processing=SequentialPreprocessingAdapterConfig(
        steps=[clean_signal, normalize_step]
    )
).build()

transformer.fit(ds)
transformed_ds = transformer.transform(ds)

print("\nAfter normalization:")
transformed_ds[0].signal["T-TPT"].head(10)

print("Before renaming:")
print(ds[0].signal.columns.tolist())

columns_map = {"ABER-CKGL": "sensor_A", "ABER-CKP": "sensor_B"}

rename_step = RenameColumnsConfig(columns_map=columns_map)

transformer = TransformConfig(pre_processing=rename_step).build()
transformer.fit(ds)
transformed_ds = transformer.transform(ds)

print("\nAfter renaming:")
print(transformed_ds[0].signal.columns.tolist())

pipeline = SequentialPreprocessingAdapterConfig(
    steps=[
        CleanSignalsConfig(),
        ImputeMissingConfig(strategy="mean"),
        NormalizeConfig(norm="l2"),
    ]
)

transformer = TransformConfig(pre_processing=pipeline).build()
transformer.fit(ds)
transformed_ds = transformer.transform(ds)

print("Pipeline applied successfully.")

ds = ParquetDatasetConfig(
    path=dataset_path,
    event_type=["real"],
).build()

print(f"\nNumber of events in the dataset: {len(ds)}")

transformer = TransformConfig(
    feature_extraction=SequentialFeatureAdapterConfig(
        steps=[
            # Step 1: Windowing (required)
            WindowingConfig(),
            # Step 2: Feature extraction
            ConcatFeatureAdapterConfig(
                steps=[
                    StatisticalConfig(),
                    EWStatisticalConfig(),
                    WaveletConfig(),
                ]
            ),
        ]
    ),
).build()

pre_processing_pipeline = SequentialPreprocessingAdapterConfig(
    steps=[
        CleanSignalsConfig(),
        ImputeMissingConfig(),
    ]
)

transformer = TransformConfig(
    pre_processing=pre_processing_pipeline,
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
).build()

transformer.fit(ds)
transformed_ds = transformer.transform(ds)

print("Feature extraction completed.")

#### Example: Applying windowing

data_processor = TransformConfig(
    pre_processing=pre_processing_pipeline,
    feature_extraction=WindowingConfig(window_size=128),
).build()

data_processor.fit(ds)
transformed_data = data_processor.transform(ds)

single_event = transformed_data[0]

print(f"Windowed signal shape: {single_event.signal.shape}")
print("-" * 50)

print("Columns:")
print(single_event.signal.columns)
print("-" * 50)

print("Preview:")
print(single_event.signal.head())
print("-" * 50)

#### Example: Windowing + Statistical features

windowing_config = WindowingConfig(window_size=128)

data_processor = TransformConfig(
    pre_processing=ImputeMissingConfig(),
    feature_extraction=SequentialFeatureAdapterConfig(
        steps=[
            windowing_config,
            StatisticalConfig(features=["mean", "std", "min", "max"]),
        ]
    ),
).build()

data_processor.fit(ds)
transformed_data = data_processor.transform(ds)

single_event = transformed_data[0]

print(f"Feature matrix shape: {single_event.signal.shape}")
print("-" * 50)

print("Feature columns:")
print(single_event.signal.columns)
print("-" * 50)

print("Preview:")
print("-" * 50)
single_event.signal.head()

# Example: Windowing + Wavelet features
# Shared parameters
LEVEL = 7
WINDOW_SIZE = 128
OVERLAP = 0.875

# Windowing configuration
windowing_config = WindowingConfig(
    window_size=WINDOW_SIZE, overlap=OVERLAP, window="boxcar"
)

# Feature extraction pipeline
data_processor = TransformConfig(
    pre_processing=pre_processing_pipeline,
    feature_extraction=SequentialFeatureAdapterConfig(
        steps=[
            windowing_config,
            WaveletConfig(level=LEVEL),
        ]
    ),
).build()

data_processor.fit(ds)
transformed_data = data_processor.transform(ds)

single_event = transformed_data[0]

print(f"Feature matrix shape: {single_event.signal.shape}")
print("-" * 50)

print("Feature columns:")
print(single_event.signal.columns)
print("-" * 50)

print("Preview:")
print("-" * 50)
single_event.signal.head()

# Example: Windowing + EW statistical features
WINDOW_SIZE = 128

data_processor = TransformConfig(
    pre_processing=pre_processing_pipeline,
    feature_extraction=SequentialFeatureAdapterConfig(
        steps=[
            WindowingConfig(window_size=WINDOW_SIZE),
            EWStatisticalConfig(decay=0.9),
        ]
    ),
).build()

data_processor.fit(ds)
transformed_data = data_processor.transform(ds)

single_event = transformed_data[0]

print(f"Feature matrix shape: {single_event.signal.shape}")
print("-" * 50)

print("Feature columns:")
print(single_event.signal.columns)
print("-" * 50)

print("Preview:")
print("-" * 50)
single_event.signal.head()

ds = ParquetDatasetConfig(
    path=dataset_path,
    event_type=["real"],
).build()

sig = ds[2].signal.copy()

print("Selected event shape:", sig.shape)
print("Available columns:", sig.columns.tolist())

series = sig["T-TPT"]

fig, path = DataVisualization.plot_series(
    series=series,
    title="T-TPT",
    xlabel="Timestamp",
    ylabel="T-TPT",
    overlay_events=False,
    color="green",
)

plt.show()

features = ["T-JUS-CKP", "T-TPT"]
series_list = [sig[f] for f in features]

fig, ax = plt.subplots(figsize=(12, 5))

DataVisualization.plot_multiple_series(
    series_list=series_list,
    labels=features,
    title="T-JUS-CKP vs T-TPT",
    xlabel="Timestamp",
    ylabel="Value",
    ax=ax,
)

plt.show()

features = [
    "P-ANULAR",
    "P-JUS-CKGL",
    "P-MON-CKP",
    "P-TPT",
    "T-JUS-CKP",
    "T-TPT",
]

subset = sig[features]

fig = DataVisualization.correlation_heatmap(
    df_of_series=subset,
    title="Correlation Heatmap of Selected Features",
)

plt.show()

RANDOM_SEED = 2026

dataset_path = "dataset"
classes = [0, 1, 2]

ds = ParquetDatasetConfig(
    path=dataset_path,
    event_type=["real"],
    target_class=classes,
).build()

print(f"\nTotal events: {len(ds)}")

np.random.seed(RANDOM_SEED)

all_files = ds.files_events

indices = np.arange(len(all_files))
np.random.shuffle(indices)

n = len(indices)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

train_files = [all_files[i] for i in train_idx]
val_files = [all_files[i] for i in val_idx]
test_files = [all_files[i] for i in test_idx]

ds_train = ParquetDatasetConfig(
    path=dataset_path,
    split="list",
    file_list=train_files,
).build()

ds_val = ParquetDatasetConfig(
    path=dataset_path,
    split="list",
    file_list=val_files,
).build()

ds_test = ParquetDatasetConfig(
    path=dataset_path,
    split="list",
    file_list=test_files,
).build()

window_size = 128

dataset_processor = TransformConfig(
    pre_processing=SequentialPreprocessingAdapterConfig(
        steps=[
            CleanSignalsConfig(missing_columns_threshold=0.65),
            ImputeMissingConfig(),
            NormalizeConfig(),
            FillLabelsConfig(),
            RemapClassConfig(),
        ]
    ),
    feature_extraction=SequentialFeatureAdapterConfig(
        steps=[
            WindowingConfig(window_size=window_size),
            ConcatFeatureAdapterConfig(
                steps=[
                    StatisticalConfig(),
                    EWStatisticalConfig(),
                    WaveletConfig(),
                ]
            ),
        ]
    ),
).build()

dataset_processor.fit(ds_train)

ds_train_transformed = dataset_processor.transform(ds_train)
ds_val_transformed = dataset_processor.transform(ds_val)
ds_test_transformed = dataset_processor.transform(ds_test)

mlp_config = MLPConfig(
    hidden_sizes=(32, 16),
    output_size=dataset_processor.num_classes,
)

trainer = TorchTrainerConfig(
    seed=RANDOM_SEED,
    config_model=mlp_config,
    learning_rate=0.0001,
    batch_size=32,
    epochs=30,
).build()

train_results = trainer.train(
    train_dataset=ds_train_transformed,
    val_dataset=ds_val_transformed,
)

test_predictions = trainer.predict(ds_test_transformed)

assessment = ModelAssessmentConfig(
    metrics=["accuracy"],
).build()

results = assessment.evaluate(
    training_results=train_results,
    predictions=test_predictions,
)

print("Evaluation results:", results)

history = train_results.history

plt.figure(figsize=(10, 5))
plt.plot(history.train_loss, label="Train Loss")
plt.plot(history.val_loss, label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

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

from ThreeWToolkit.reports.report_generation import ReportGeneration

train_len = sum(len(event.signal) for event in ds_train_transformed)
test_len = sum(len(event.signal) for event in ds_test_transformed)

predictions = trainer.predict(ds_test_transformed)

assessment = ModelAssessmentConfig(
    metrics=["accuracy", "precision", "recall", "f1"],
).build()

metrics_to_include = assessment.evaluate(
    training_results=train_results,
    predictions=predictions,
)

sig = ds_test[0].signal.copy()

plot_config = {
    "PlotSeries": {
        "series": sig["T-TPT"],
        "title": "T-TPT",
        "xlabel": "Timestamp",
        "ylabel": "T-TPT",
        "overlay_events": False,
    },
    "PlotMultipleSeries": {
        "series_list": [sig["P-MON-CKP"], sig["P-TPT"]],
        "labels": ["P-MON-CKP", "P-TPT"],
        "title": "Pressure Signals",
        "xlabel": "Timestamp",
        "ylabel": "Value",
    },
    "PlotCorrelationHeatmap": {
        "df_of_series": sig[
            ["P-ANULAR", "P-JUS-CKGL", "P-MON-CKP", "P-TPT", "T-JUS-CKP", "T-TPT"]
        ],
        "title": "Correlation Heatmap",
    },
}

report_generation = ReportGeneration(
    model=trainer.model,
    train_len=train_len,
    test_len=test_len,
    predictions=predictions,
    calculated_metrics=metrics_to_include.metrics,
    plot_config=plot_config,
    title="3WToolkit Signal Analysis Report",
    author="Your Name",
    export_report_after_generate=False,
)

html_report = report_generation.generate_summary_report(
    template_name="report_template.html",
    format="html",
)

latex_report = report_generation.generate_summary_report(format="latex")

report_generation.save_report(html_report, "signal_analysis_report", format="html")

report_generation.save_report(latex_report, "signal_analysis_report", format="latex")

from pathlib import Path

p = Path("/usr/local/lib/python3.10/site-packages/ThreeWToolkit/reports/html")
print(p.exists())
print(list(p.glob("*")))