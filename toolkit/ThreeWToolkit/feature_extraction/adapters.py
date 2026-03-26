import pandas as pd

from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from ..core.dataset_outputs import DatasetOutputs

from pydantic import Field


class SequentialFeatureAdapterConfig(BaseFeatureExtractorConfig):
    steps: list[BaseFeatureExtractorConfig]
    target_: type = Field(default_factory=lambda: SequentialFeatureAdapter)


class ConcatFeatureAdapterConfig(BaseFeatureExtractorConfig):
    steps: list[BaseFeatureExtractorConfig]
    target_: type = Field(default_factory=lambda: ConcatFeatureAdapter)


class SequentialFeatureAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations sequentially to DatasetOutputs.
    Each transformation should be a BaseFeatureExtractor instance.
    """

    steps: list[BaseFeatureExtractor] = Field(default_factory=list)

    def __init__(
        self,
        config: SequentialFeatureAdapterConfig,
    ):
        self.config = config
        self.steps = [step_config.build() for step_config in self.config.steps]

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        for transform in self.steps:
            data = transform.transform(data)
        return data


class ConcatFeatureAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations to DatasetOutputs and concatenates signal outputs.
    Labels from the first non-None label are used.
    """

    steps: list[BaseFeatureExtractor] = Field(default_factory=list)

    def __init__(self, config: ConcatFeatureAdapterConfig):
        self.config = config
        self.steps = [step.build() for step in self.config.steps]

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:

        outputs = [step.transform(data) for step in self.steps]

        return DatasetOutputs(
            signal=pd.concat([output.signal for output in outputs], axis=1),
            label=outputs[0].label,
            metadata=outputs[0].metadata.copy(),
        )
