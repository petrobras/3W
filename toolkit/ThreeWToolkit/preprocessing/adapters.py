from pydantic import Field, PrivateAttr

from ..dataset.transformed_dataset import TransformedDataset

from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class SequentialPreprocessingAdapterConfig(BasePreprocessingConfig):
    """Configuration for SequentialPreprocessingAdapter, which applies a list of preprocessing steps sequentially.
    Args:
        steps (list[BasePreprocessingConfig]): A list of preprocessing step configurations to apply sequentially.
    """

    steps: list[BasePreprocessingConfig] = Field(
        ..., description="List of preprocessing steps to apply sequentially."
    )
    _target: type = PrivateAttr(default_factory=lambda: SequentialPreprocessingAdapter)


class SequentialPreprocessingAdapter(BasePreprocessing):
    """
    Applies a list of preprocessing steps sequentially to DatasetOutputs.
    Each step should be a BasePreprocessing instance.
    """

    def __init__(self, config: SequentialPreprocessingAdapterConfig):
        """ Instantiate the adapter with the given configuration, building each\
                preprocessing step from its configuration.
        Args:
            config (SequentialPreprocessingAdapterConfig): The configuration for the adapter.
        """
        self.config: SequentialPreprocessingAdapterConfig = config
        self.steps: list[BasePreprocessing] = [
            step_config.build() for step_config in self.config.steps
        ]

    def fit(self, data: BaseDataset) -> None:
        """Fit each preprocessing step sequentially on the dataset, feeding the transformed\
           data from the previous step to the next.
           This method sequentially fits each preprocessing step on the dataset, ensuring that the output of one step is
           correctly fed into the next step's fitting process.

        Args:
            data (BaseDataset): The dataset to fit the preprocessing steps on.
        """

        fitted_steps: list[BasePreprocessing] = []
        for step in self.steps:

            def _intermediate_transform(data: DatasetOutputs) -> DatasetOutputs:
                for fitted_step in fitted_steps:
                    data = fitted_step.transform(data)
                return data

            intermediate_dataset = TransformedDataset(data, _intermediate_transform)
            step.fit(intermediate_dataset)
            fitted_steps.append(step)

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply each preprocessing step sequentially to the input data.
        Args:
            data (DatasetOutputs): The input data to transform.
        Returns:
            DatasetOutputs: The transformed data after applying all preprocessing steps sequentially.
        """
        for step in self.steps:
            data = step.transform(data)
        return data
