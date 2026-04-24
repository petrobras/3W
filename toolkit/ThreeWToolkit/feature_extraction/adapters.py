import pandas as pd

from ..core.base_feature_extractor import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from ..core.dataset_outputs import DatasetOutputs

from pydantic import Field, PrivateAttr


class SequentialFeatureAdapterConfig(BaseFeatureExtractorConfig):
    """Configuration for sequential feature extraction."""

    steps: list[BaseFeatureExtractorConfig] = Field(
        ..., description="List of feature extractors to apply sequentially."
    )
    _target: type = PrivateAttr(default_factory=lambda: SequentialFeatureAdapter)


class ConcatFeatureAdapterConfig(BaseFeatureExtractorConfig):
    """Configuration for concatenating outputs from multiple feature extractors."""

    steps: list[BaseFeatureExtractorConfig] = Field(
        ...,
        description="List of feature extractors whose outputs will be concatenated.",
    )
    _target: type = PrivateAttr(default_factory=lambda: ConcatFeatureAdapter)


class SequentialFeatureAdapter(BaseFeatureExtractor):
    """
    Applies a list of transformations sequentially to DatasetOutputs.
    Each transformation should be a BaseFeatureExtractor instance.
    """

    steps: list[BaseFeatureExtractor] = Field(
        default_factory=list, description="List of feature extractors to apply."
    )

    def __init__(
        self,
        config: SequentialFeatureAdapterConfig,
    ):
        """Instantiate the adapter with the given configuration, building each feature extractor from its
        configuration.
        Args:
            config (SequentialFeatureAdapterConfig): The configuration for the adapter.
        """
        self.config = config
        self.steps = [step_config.build() for step_config in self.config.steps]

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply each feature extractor sequentially to the data, feeding the transformed data from the previous step
        to the next.
        Args:
            data (DatasetOutputs): The input data to transform.
        Returns:
            DatasetOutputs: The transformed data after applying all feature extractors sequentially.
        """
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

    Args:
        steps (list[BaseFeatureExtractor]): A list of feature extractors to apply and concatenate.
    """

    steps: list[BaseFeatureExtractor] = Field(
        ..., min_length=1, description="List of feature extractors to apply."
    )

    def __init__(self, config: ConcatFeatureAdapterConfig):
        """Instantiate the adapter with the given configuration, building each feature extractor from its
        configuration.
        Args:
            config (ConcatFeatureAdapterConfig): The configuration for the adapter.
        """
        self.config = config
        self.steps = [step.build() for step in self.config.steps]

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply each feature extractor to the data and concatenate their outputs.
        The output signals are concatenated along the feature axis (axis=1). The label from the first step is used for
        the output, and metadata from all steps is combined, with precedence given to earlier steps in case of
        conflicts.
        Args:
            data (DatasetOutputs): The input data to transform.
        Returns:
            DatasetOutputs: The transformed data after applying all feature extractors and concatenating their outputs.
        """

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
