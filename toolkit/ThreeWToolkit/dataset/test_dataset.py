from ThreeWToolkit.dataset import (
    ParquetDataset,
    ParquetDatasetConfig,
    TransformDatasetConfig,
    TransformDataset,
)
from ThreeWToolkit.preprocessing import (
    NormalizeConfig,
    RenameColumnsConfig,
    ImputeMissingConfig,
    RemapClassConfig,
    FillLabelsConfig,
    SequentialPreprocessingAdapterConfig,
)
from ThreeWToolkit.feature_extraction import (
    WindowingConfig,
    StatisticalConfig,
    EWStatisticalConfig,
    WaveletConfig,
    ConcatFeatureAdapterConfig,
    SequentialFeatureAdapterConfig,
)
from ThreeWToolkit.core.enums import EventPrefixEnum
from ThreeWToolkit.trainer.trainer import TrainerConfig, ModelTrainer
from ThreeWToolkit.models.mlp import MLPConfig, MLP

if __name__ == "__main__":
    config_train = ParquetDatasetConfig(
        path="/home/eduardo/3W/dataset",
        version="2.0.0",
        columns=["T-JUS-CKP", "T-MON-CKP"],  # ,
        target_column="class",
        target_class=[1, 2, 3, 4, 5],
        force_download=False,
        # file_list="dataset_files.txt",
        event_type=[EventPrefixEnum.REAL],
        # split="train", "test",
    )
    config_val = ParquetDatasetConfig(
        path="/home/eduardo/3W/dataset",
        version="2.0.0",
        columns=["T-JUS-CKP", "T-MON-CKP"],  # ,
        target_column="class",
        target_class=[1, 2, 3, 4, 5],
        force_download=False,
        # file_list="dataset_files.txt",
        event_type=[EventPrefixEnum.DRAWN],
        # split="train", "test",
    )
    ds_train = ParquetDataset(config_train)
    # ds_val = ParquetDataset(config_val)

    # ((t1, v1), 2, 3, 4, 5) = Subset(ParquetDataset, n_folds=5)

    process_cfg = TransformDatasetConfig(
        pre_processing=SequentialPreprocessingAdapterConfig(
            steps=[
                # FillLabelsConfig(),
                # ImputeMissingConfig(),
                # NormalizeConfig(),
                # RenameColumnsConfig(),
                RemapClassConfig(),
            ]
        ),
        feature_extraction=ConcatFeatureAdapterConfig(
            transforms=[
                # WindowingConfig(),
                StatisticalConfig(),
                # EWStatisticalConfig(),
                # WaveletConfig(),
            ]
        ),
    )

    processed_ds = TransformDataset(process_cfg)
    ds_train_transformed = processed_ds.fit_and_transform(ds_train)

    print(f"Number of events after processing: {len(ds_train_transformed)}")
    print("--------------------")

    ds_val_transformed = processed_ds.transform(ds_val)
    print(f"Number of events after processing: {len(ds_val_transformed)}")
    print("--------------------")

    # get the x and y as dataframes
    y_train = ds_train_transformed["label"].astype(int)
    x_train = ds_train_transformed.drop(columns=["label"])

    # get y_train max value
    max_value = y_train.max()
    print(f"Max value in y_train: {max_value}")

    mlp_config = MLPConfig(
        random_seed=42,
        input_size=(ds_train_transformed.shape[1] - 1),
        hidden_sizes=(64, 32),
        output_size=max_value + 1,
    )
    print(f"model: {mlp_config}")

    trainer_cfg = TrainerConfig(
        config_model=mlp_config,
        epochs=10,
        batch_size=32,
    )
    print(f"trainer config: {trainer_cfg}")

    model_trainer = ModelTrainer(trainer_cfg)

    model_trainer.train(x_train=x_train, y_train=y_train)
    print("Training completed.")

    # for idx in range(len(input_dataclass)):
    #     train = Dataloader(input_dataclass[idx].train, batch_size=32, shuffle=True)
    #     val = Dataloader(input_dataclass[idx].val, batch_size=32, shuffle=True)
    #     test = Dataloader(input_dataclass.test, batch_size=32, shuffle=True)

    #     x, y = input_dataclass[idx].to_dataframe()
