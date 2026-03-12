import pandas as pd

from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)

from pydantic import Field


class SequentialAdapterConfig(BaseFeatureExtractorConfig):
    transforms: list[BaseFeatureExtractorConfig]
    target_: type = Field(default_factory=lambda: SequentialAdapter)


class ConcatAdapterConfig(BaseFeatureExtractorConfig):
    transforms: list[BaseFeatureExtractorConfig]
    target_: type = Field(default_factory=lambda: ConcatAdapter)


class SequentialAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations sequentially to the input data.
    Each transformation should be a callable that takes and returns data.
    """

    feature_extraction_steps: list = []

    def __init__(
        self,
        config: SequentialAdapterConfig,
    ):
        self.config = config
        self.feature_extraction_steps = []
        for step_config in self.config.transforms:
            step_instance = step_config.build()
            self.feature_extraction_steps.append(step_instance)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for transform in self.feature_extraction_steps:
            data = transform.transform(data)
        return data


class ConcatAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations to the input data and concatenates their outputs.
    Each transformation should be a callable that takes and returns data.
    The outputs must be compatible for concatenation (e.g., numpy arrays, pandas DataFrames, or tensors).
    """

    feature_extraction_steps: list = []

    def __init__(self, config: ConcatAdapterConfig):
        self.config = config
        self.feature_extraction_steps = []
        for step_config in self.config.transforms:
            step_instance = step_config.build()
            self.feature_extraction_steps.append(step_instance)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_dict = {}
        for transform in self.feature_extraction_steps:
            features = transform.transform(data)

            # Only keep new columns
            for col in features.columns:
                if col not in feature_dict:
                    feature_dict[col] = features[col].values

        features_df = pd.DataFrame(feature_dict)
        return features_df
