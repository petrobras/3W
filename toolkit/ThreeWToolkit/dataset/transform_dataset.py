from typing import Any
from pydantic import BaseModel, Field, field_validator
from typing import Sequence
from ThreeWToolkit.preprocessing import (
    NormalizeConfig,
    ImputeMissingConfig,
    RenameColumnsConfig,
    ImputeMissing,
    RenameColumns,
)
from ThreeWToolkit.feature_extraction import (
    WindowingConfig,
    Windowing,
    StatisticalConfig,
    StatisticalFeatures,
    WaveletConfig,
    WaveletFeatures,
    EWStatisticalConfig,
    EWStatisticalFeatures,
)
import pandas as pd
from tqdm import tqdm


class TransformDatasetConfig(BaseModel):
    """
    Configuration schema for processing a dataset.
    """

    # dataset_config: ParquetDatasetConfig
    pre_processing: (
        Sequence[NormalizeConfig | ImputeMissingConfig | RenameColumnsConfig] | None
    ) = Field(
        default=None,
        description="List of preprocessing steps to apply to the dataset.",
    )
    feature_extraction: (
        Sequence[
            WindowingConfig | StatisticalConfig | WaveletConfig | EWStatisticalConfig
        ]
        | None
    ) = Field(
        default=None,
        description="List of feature extraction steps to apply to the dataset after preprocessing.",
    )

    @field_validator("pre_processing", mode="before")
    @classmethod
    def validate_pre_processing(cls, v):
        if v is None:
            return None
        if isinstance(
            v,
            (
                NormalizeConfig,
                ImputeMissingConfig,
                RenameColumnsConfig,
            ),
        ):
            return [v]
        if isinstance(v, (list, tuple)):
            for item in v:
                if not isinstance(
                    item,
                    (
                        NormalizeConfig,
                        ImputeMissingConfig,
                        RenameColumnsConfig,
                    ),
                ):
                    raise TypeError(
                        "All items in pre_processing must be WindowingConfig, NormalizeConfig, ImputeMissingConfig, RenameColumnsConfig, or DropColumnsConfig"
                    )
            return list(v)
        raise TypeError("pre_processing must be None, a config, or a list of configs")

    @field_validator("feature_extraction", mode="before")
    @classmethod
    def validate_feature_extraction(cls, v):
        if v is None:
            return None
        if isinstance(
            v,
            (
                WindowingConfig,
                StatisticalConfig,
                WaveletConfig,
                EWStatisticalConfig,
            ),
        ):
            return [v]
        if isinstance(v, (list, tuple)):
            for item in v:
                if not isinstance(
                    item,
                    (
                        WindowingConfig,
                        StatisticalConfig,
                        WaveletConfig,
                        EWStatisticalConfig,
                    ),
                ):
                    raise TypeError(
                        "All items in feature_extraction must be WindowingConfig, StatisticalConfig, WaveletConfig, or EWStatisticalConfig"
                    )
            return list(v)
        raise TypeError(
            "feature_extraction must be None, a config, or a list of configs"
        )


class TransformDataset:
    """
    This class will be used as a wrapper to process datasets with various preprocessing steps.
    So it will call every given preprocessing step, collect statistics if needed (in a single run over the data)
    and have the statistics read to be used in the transformations (e.g. mean and std for normalization).

    The input data need to be a portion of the data already splitted, it will work as a dataset on its own.
    """

    def __init__(self, config: TransformDatasetConfig):
        super().__init__()
        self.config = config

        # Create preprocessing steps based on available configs
        self.pre_processing_steps = []
        if self.config.pre_processing is not None:
            for step_config in self.config.pre_processing:
                step_instance = step_config.target(step_config)
                self.pre_processing_steps.append(step_instance)

        # Validate and fix the order of preprocessing steps
        # self.pre_processing_steps = self._validate_and_fix_preprocessing_steps(
        #     self.pre_processing_steps
        # )

        # Create feature extraction steps based on available configs
        self.feature_extraction_steps = []
        if self.config.feature_extraction is not None:
            for step_config in self.config.feature_extraction:
                step_instance = step_config.target(step_config)
                self.feature_extraction_steps.append(step_instance)

    def fit(self, dataset: Any) -> None:
        """
        Fit preprocessing and feature extraction steps on the dataset, collecting necessary statistics.
        It needs to run through the entire list of events in the order given by the user,
        so it will be slow if there are many steps that require statistics collection,
        but it will ensure that the statistics are collected in the correct order.
        """
        for step in self.pre_processing_steps:
            for idx in range(len(dataset)):
                data = dataset[idx]
                if hasattr(step, "collect_statistics"):
                    step.collect_statistics(data)
            if hasattr(step, "compute_statistics"):
                step.compute_statistics()

    def transform(self, dataset: Any) -> Any:
        """
        Apply the fitted preprocessing and feature extraction steps to the dataset, using the collected statistics.
        This method is useful for applying the collected statistics from the training set to the validation and test sets,
        ensuring that the same transformations are applied consistently across all splits.
        """
        preprocessed_dataset = []
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            for step in self.pre_processing_steps:
                data = step.run(data)
            preprocessed_dataset.append(data)

        # here we have a list of dicts
        #   ['signal', 'label', 'file_name']

        # should we transform it to a dataframe before passing to feature extraction steps?
        # For this we would loop first by feature extraction steps

        # or should we do it by event and then concatenate the results?
        # For this we would loop first by events and then by feature extraction steps,
        # but it would be more computationally expensive
        all_features = []
        for idx in tqdm(range(len(preprocessed_dataset))):
            # original event data (df)
            data = preprocessed_dataset[idx]
            df = pd.concat([data["signal"], data["label"]], axis=1)

            # features_df hold the features for the current event,
            # we will concatenate the features from each step
            # as new columns
            feature_dict = {}
            for step in self.feature_extraction_steps:
                features = step.transform(df)
                # Only keep new columns (avoid duplicates)
                for col in features.columns:
                    if col not in feature_dict:
                        feature_dict[col] = features[col].values
            # Build DataFrame for this event
            features_df = pd.DataFrame(feature_dict)
            all_features.append(features_df)

        # concatenate all events features as rows
        final_df = pd.concat(all_features, ignore_index=True)
        print(
            f"final number of rows: {final_df.shape[0]}, final number of columns: {final_df.shape[1]}"
        )
        return final_df

    def fit_and_transform(self, dataset: Any) -> Any:
        """
        This method combines fit and transform, and should be called first on the training set,
        it will fit the steps and transform the dataset in a single run.
        """
        self.fit(dataset)
        return self.transform(dataset)

    # def _validate_and_fix_preprocessing_steps(self, steps):
    #     """
    #     Validate and automatically adjust the order of preprocessing steps.

    #     Enforces the following rules:
    #     - ImputeMissing should be the first (if present)
    #     - Only one RenameColumns is allowed
    #     - Windowing should be the last (if present)

    #     Args:
    #         steps (list): List of preprocessing steps

    #     Returns:
    #         list: Reordered list of preprocessing steps

    #     Raises:
    #         TypeError: If steps don't inherit from BaseStep
    #         ValueError: If multiple RenameColumns steps are provided
    #     """
    #     for step in steps:
    #         if not isinstance(step, BaseStep):
    #             raise TypeError(
    #                 f"Step '{step.__class__.__name__}' must inherit from BaseStep."
    #             )

    #     impute_steps = [s for s in steps if isinstance(s, ImputeMissing)]
    #     rename_steps = [s for s in steps if isinstance(s, RenameColumns)]
    #     windowing_steps = [s for s in steps if isinstance(s, Windowing)]

    #     # Only one RenameColumns is allowed
    #     if len(rename_steps) > 1:
    #         raise ValueError("Only one RenameColumns step is allowed.")

    #     # Reorder: ImputeMissing first, then other steps, then RenameColumns, then Windowing last
    #     reordered = []
    #     reordered.extend(impute_steps)
    #     for step in steps:
    #         if (
    #             step not in impute_steps
    #             and step not in rename_steps
    #             and step not in windowing_steps
    #         ):
    #             reordered.append(step)
    #     reordered.extend(rename_steps)
    #     reordered.extend(windowing_steps)
    #     return reordered
