from ThreeWToolkit.dataset import (
    ParquetDatasetConfig,
)
from ..preprocessing import *
from ..feature_extraction import *
from ThreeWToolkit.dataset import TransformConfig, TransformDataset

from ThreeWToolkit.core.enums import EventPrefixEnum
from ThreeWToolkit.trainer.trainer import TrainerConfig, ModelTrainer
from ThreeWToolkit.models.mlp import MLPConfig, MLP

if __name__ == "__main__":
    config_train = ParquetDatasetConfig(
        path="../../../dataset/",
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
        path="../../../dataset/",
        version="2.0.0",
        columns=["T-JUS-CKP", "T-MON-CKP"],  # ,
        target_column="class",
        target_class=[1, 2, 3, 4, 5],
        force_download=False,
        # file_list="dataset_files.txt",
        event_type=[EventPrefixEnum.DRAWN],
        # split="train", "test",
    )
    ds_train = config_train.build()
    ds_val = config_val.build()

    # ((t1, v1), 2, 3, 4, 5) = Subset(ParquetDataset, n_folds=5)

    dataset_processor = TransformConfig(
        pre_processing=SequentialPreprocessingAdapterConfig(steps=[ ImputeMissingConfig(), FillLabelsConfig(),]),
        feature_extraction=ConcatFeatureAdapterConfig(steps=[ StatisticalConfig(),]),
    ).build()

    dataset_processor.fit(ds_train)

    transformed_ds_train = dataset_processor.transform(ds_train)
    print(f"Number of events after processing: {len(transformed_ds_train)}")
    print("--------------------")

    transformed_ds_valid = dataset_processor.transform(ds_val)


    print(f"Number of events after processing: {len(transformed_ds_valid)}")
    print("--------------------")

    print("Sample transformed event:")
    sample_event = transformed_ds_train[0]
    print(sample_event)

    print("Sample validation event:")
    sample_val_event = transformed_ds_valid[0]
    print(sample_val_event)

    exit(0)

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
