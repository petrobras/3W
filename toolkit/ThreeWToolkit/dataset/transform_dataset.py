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
from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from ..core.base_transform import BaseTransform, BaseTransformConfig
import pandas as pd
from tqdm import tqdm


class TransformDatasetConfig(BaseTransformConfig):
    """
    Configuration for processing a dataset.
    """

    pre_processing: (
        Sequence[NormalizeConfig | ImputeMissingConfig | RenameColumnsConfig] | None
    ) = Field(
        default=None,
        description="List of preprocessing steps to apply to the dataset.",
    )
    feature_extraction: BaseFeatureExtractorConfig | None = Field(
        default=None,
        description="List of feature extraction steps to apply to the dataset after preprocessing.",
    )
    target_: type = Field(default_factory=lambda: TransformDataset)


class TransformDataset(BaseTransform):
    """ """

    def __init__(self, config: TransformDatasetConfig):
        super().__init__(config)
        self.config = config

        # Create preprocessing steps based on available configs
        self.pre_processing_steps = []
        if self.config.pre_processing is not None:
            for step_config in self.config.pre_processing:
                step_instance = step_config.build()
                self.pre_processing_steps.append(step_instance)

        if self.config.feature_extraction is not None:
            self.feature_extraction_step = self.config.feature_extraction.build()

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
                step.fit(data)
                step.compute()

    def transform(self, dataset: Any) -> pd.DataFrame:
        """
        Apply the fitted preprocessing and feature extraction steps to the dataset, using the collected statistics.
        This method is useful for applying the collected statistics from the training set to the validation and test sets,
        ensuring that the same transformations are applied consistently across all splits.
        """
        preprocessed_dataset = []
        for idx in tqdm(range(len(dataset))):
            data = dataset[idx]
            for step in self.pre_processing_steps:
                data = step.transform(data)
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
            data = preprocessed_dataset[idx]
            df = pd.concat([data["signal"], data["label"]], axis=1)

            if self.feature_extraction_step is not None:
                features_df = self.feature_extraction_step.transform(df)
                all_features.append(features_df)

        final_df = pd.concat(all_features, ignore_index=True)
        print(
            f"final number of rows: {final_df.shape[0]}, final number of columns: {final_df.shape[1]}"
        )
        return final_df
