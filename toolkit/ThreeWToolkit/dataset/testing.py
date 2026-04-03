from ThreeWToolkit.dataset import (
    ParquetDatasetConfig,
)
from ThreeWToolkit.preprocessing import (
    FillLabelsConfig,
    ImputeMissingConfig,
    NormalizeConfig,
    RemapClassConfig,
    RenameColumnsConfig,
    CleanSignalsConfig,
    SequentialPreprocessingAdapterConfig,
)
from ThreeWToolkit.feature_extraction import (
    WindowingConfig,
    StatisticalConfig,
    WaveletConfig,
    EWStatisticalConfig,
    ConcatFeatureAdapterConfig,
    SequentialFeatureAdapterConfig,
)
from ThreeWToolkit.pipeline import PipelineConfig, Pipeline
from ThreeWToolkit.dataset import TransformConfig
from ThreeWToolkit.core.enums import EventPrefixEnum
from ThreeWToolkit.trainer import (
    TorchTrainerConfig,
    SklearnTrainerConfig,
)
from ThreeWToolkit.models.mlp import MLPConfig, MLP
from ThreeWToolkit.models.sklearn_models import SklearnModels, SklearnModelsConfig
from ThreeWToolkit.assessment import ModelAssessment
from ThreeWToolkit.core.enums import ModelTypeEnum


if __name__ == "__main__":
    path = "/home/eduardo/hold_new_code/3W/dataset"

    config_train = ParquetDatasetConfig(
        path=path,
        version="2.0.0",
        columns=[
            "T-JUS-CKP",
            "T-MON-CKP",
            "P-JUS-BS",
            "P-JUS-CKGL",
            "P-JUS-CKP",
            "P-MON-CKGL",
            "P-MON-CKP",
            "P-MON-SDV-P",
        ],
        target_column="class",
        target_class=[1, 2, 3, 4, 5],
        force_download=False,
        # file_list="dataset_files.txt",
        event_type=[EventPrefixEnum.REAL],
        # split="train", "test",
    )
    # config_val = ParquetDatasetConfig(
    #     path=path,
    #     version="2.0.0",
    #     columns=[
    #         "T-JUS-CKP",
    #         "T-MON-CKP",
    #         "P-JUS-BS",
    #         "P-JUS-CKGL",
    #         "P-JUS-CKP",
    #         "P-MON-CKGL",
    #         "P-MON-CKP",
    #         "P-MON-SDV-P",
    #     ],
    #     target_column="class",
    #     target_class=[1, 2, 3, 4, 5],
    #     force_download=False,
    #     # file_list="dataset_files.txt",
    #     event_type=[EventPrefixEnum.DRAWN],
    #     # split="train", "test",
    # )
    ds_train = config_train.build()
    # ds_val = config_val.build()

    # # ((t1, v1), 2, 3, 4, 5) = Subset(ParquetDataset, n_folds=5)

    dataset_processor = TransformConfig(
        pre_processing=SequentialPreprocessingAdapterConfig(
            steps=[
                CleanSignalsConfig(missing_column_threshold=0.65),
                ImputeMissingConfig(),
                NormalizeConfig(),
                FillLabelsConfig(),
                RemapClassConfig(),
            ]
        ),
        feature_extraction=SequentialFeatureAdapterConfig(
            steps=[
                WindowingConfig(),
                ConcatFeatureAdapterConfig(
                    steps=[StatisticalConfig(), EWStatisticalConfig(), WaveletConfig()]
                ),
            ]
        ),
    ).build()

    dataset_processor.fit(ds_train)

    transformed_ds_train = dataset_processor.transform(ds_train)
    # transformed_ds_valid = dataset_processor.transform(ds_val)

    # print(f"Number of events after processing: {len(transformed_ds_train)}")
    # print("--------------------")

    # print(f"Number of events after processing: {len(transformed_ds_valid)}")
    # print("--------------------")

    # print("Sample transformed event:")
    # sample_event = transformed_ds_train[0]
    # print(sample_event)
    # print(
    #     f"sample event columns ({len(sample_event.signal.columns)}): {sample_event.signal.columns.tolist()}"
    # )

    # print("Sample validation event:")
    # sample_val_event = transformed_ds_valid[0]
    # print(sample_val_event)

    # print(dataset_processor.num_classes)
    # mlp_config = MLPConfig(
    #     random_seed=42, hidden_sizes=(64, 32), output_size=dataset_processor.num_classes
    # )

    # trainer = TorchTrainerConfig(
    #     config_model=mlp_config,
    #     seed=42,
    #     epochs=10,
    #     batch_size=32,
    #     learning_rate=1e-3,
    #     device="cuda",
    # )  # .build()

    # results = trainer.train(transformed_ds_train, transformed_ds_valid)
    # assessment = trainer.evaluate(transformed_ds_valid)
    # results.model.save("mlp_model.pth")

    # sklearn_config = SklearnModelsConfig(
    #     model_type=ModelTypeEnum.RANDOM_FOREST,
    # )

    # trainer = SklearnTrainerConfig(
    #     config_model=sklearn_config,
    # ).build()

    # results = trainer.train(transformed_ds_train)
    # results.model.save("model")
    # assessment = trainer.evaluate(transformed_ds_valid)

    pipeline = PipelineConfig(
        generate_report=True,
        train_dataset_config=ParquetDatasetConfig(
            path=path,
            version="2.0.0",
            columns=[
                "T-JUS-CKP",
                "T-MON-CKP",
                "P-JUS-BS",
                "P-JUS-CKGL",
                "P-JUS-CKP",
                "P-MON-CKGL",
                "P-MON-CKP",
                "P-MON-SDV-P",
            ],
            target_column="class",
            target_class=[1, 2, 3, 4, 5],
            force_download=False,
            # file_list="dataset_files.txt",
            event_type=[EventPrefixEnum.REAL],
            # split="train", "test",
        ),
        trainer_config=TorchTrainerConfig(
            config_model=MLPConfig(
                hidden_sizes=(64, 32),
                output_size=dataset_processor.num_classes,
            ),
            seed=42,
            epochs=10,
            batch_size=32,
            learning_rate=1e-3,
            device="cpu",
        ),
        transform_config=TransformConfig(
            pre_processing=SequentialPreprocessingAdapterConfig(
                steps=[
                    CleanSignalsConfig(missing_column_threshold=0.65),
                    ImputeMissingConfig(),
                    NormalizeConfig(),
                    FillLabelsConfig(),
                    RemapClassConfig(),
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

    pipeline.run()

    print("Training completed.")
    # print(f"Results: {assessment}")
