from typing import Any
from pydantic import Field
from ..core.base_preprocessing import BasePreprocessingConfig
from ..core.base_feature_extractor import BaseFeatureExtractorConfig
from ..core.base_transform import BaseTransform, BaseTransformConfig
import pandas as pd
from tqdm import tqdm


class TransformDatasetConfig(BaseTransformConfig):
    """
    Configuration for transforming a dataset.
    """

    pre_processing: BasePreprocessingConfig | None = Field(
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

        if self.config.pre_processing is not None:
            self.pre_processing_step = self.config.pre_processing.build()

        if self.config.feature_extraction is not None:
            self.feature_extraction_step = self.config.feature_extraction.build()

    def fit(self, dataset: Any) -> None:
        """
        Fit preprocessing and feature extraction steps on the dataset, collecting necessary statistics.
        It needs to run through the entire list of events in the order given by the user,
        so it will be slow if there are many steps that require statistics collection,
        but it will ensure that the statistics are collected in the correct order.
        """
        for step in self.feature_extraction_step:
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
            for step in self.feature_extraction_step:
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
            signal = data.get("signal")
            label = data.get("label")
            # Ensure both are DataFrames
            if signal is not None and not isinstance(signal, pd.DataFrame):
                signal = pd.DataFrame(signal)
            if label is not None and not isinstance(label, pd.DataFrame):
                label = pd.DataFrame(label)
            # Only concat non-empty DataFrames
            dfs = [df for df in [signal, label] if df is not None and not df.empty]
            if not dfs:
                continue
            df = pd.concat(dfs, axis=1)
            print(df.head())

            if self.feature_extraction_step is not None:
                features_df = self.feature_extraction_step.transform(df)
                all_features.append(features_df)

        final_df = pd.concat(all_features, ignore_index=True)
        print(
            f"final number of rows: {final_df.shape[0]}, final number of columns: {final_df.shape[1]}"
        )
        return final_df
