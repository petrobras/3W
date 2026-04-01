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

    Must have at least one step. The label of the first step is used for the output, and all signals are
    concatenated along the feature axis (axis=1). Metadata is concatenated, with first step's metadata taking precedence
    in case of conflicts.
    """

    steps: list[BaseFeatureExtractor] = Field(..., min_length=1)

    def __init__(self, config: ConcatFeatureAdapterConfig):
        self.config = config
        self.steps = [step.build() for step in self.config.steps]

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:

        outputs = [step.transform(data) for step in self.steps]

        if not all(
            output.signal.shape[0] == outputs[0].signal.shape[0] for output in outputs
        ):
            raise ValueError(
                "All output signals must have the same number of samples for concatenation."
            )

        output_dict = {}
        for output in reversed(
            outputs
        ):  # update in reverse order to give precedence to earlier steps
            output_dict.update(output.metadata)

        return DatasetOutputs(
            signal=pd.concat([output.signal for output in outputs], axis=1),
            label=outputs[0].label,
            metadata=output_dict,
        )
