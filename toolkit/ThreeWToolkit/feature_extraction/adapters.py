import pandas as pd

from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from ..core.dataset_outputs import DatasetOutputs

from pydantic import Field


class SequentialFeatureAdapterConfig(BaseFeatureExtractorConfig):
    transforms: list[BaseFeatureExtractorConfig]
    target_: type = Field(default_factory=lambda: SequentialFeatureAdapter)


class ConcatFeatureAdapterConfig(BaseFeatureExtractorConfig):
    transforms: list[BaseFeatureExtractorConfig]
    target_: type = Field(default_factory=lambda: ConcatFeatureAdapter)


class SequentialFeatureAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations sequentially to DatasetOutputs.
    Each transformation should be a BaseFeatureExtractor instance.
    """

    feature_extraction_steps: list = []

    def __init__(
        self,
        config: SequentialFeatureAdapterConfig,
    ):
        self.config = config
        self.feature_extraction_steps = []
        for step_config in self.config.transforms:
            step_instance = step_config.build()
            self.feature_extraction_steps.append(step_instance)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        for transform in self.feature_extraction_steps:
            data = transform.transform(data)
        return data


class ConcatFeatureAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations to DatasetOutputs and concatenates signal outputs.
    Labels from the first non-None label are used.
    """

    feature_extraction_steps: list = []

    def __init__(self, config: ConcatFeatureAdapterConfig):
        self.config = config
        self.feature_extraction_steps = []
        for step_config in self.config.transforms:
            step_instance = step_config.build()
            self.feature_extraction_steps.append(step_instance)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        feature_dict = {}
        final_label = None

        for transform in self.feature_extraction_steps:
            result = transform.transform(data)

            # Collect unique signal columns
            for col in result.signal.columns:
                if col not in feature_dict:
                    feature_dict[col] = result.signal[col].values

            # Keep first non-None label
            if final_label is None and result.label is not None:
                final_label = result.label

        features_df = pd.DataFrame(feature_dict)

        return DatasetOutputs(
            signal=features_df, label=final_label, metadata=data.metadata.copy()
        )
