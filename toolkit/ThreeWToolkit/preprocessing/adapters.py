from pydantic import Field

from ..dataset.transformed_dataset import TransformedDataset

from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class SequentialPreprocessingAdapterConfig(BasePreprocessingConfig):
    """ Configuration for SequentialPreprocessingAdapter, which applies a list of preprocessing steps sequentially. """
    steps: list[BasePreprocessingConfig]
    target_: type = Field(default_factory=lambda: SequentialPreprocessingAdapter)


class SequentialPreprocessingAdapter(BasePreprocessing):
    """
    Applies a list of preprocessing steps sequentially to DatasetOutputs.
    Each step should be a BasePreprocessing instance.
    """

    def __init__(self, config: SequentialPreprocessingAdapterConfig):
        self.config: SequentialPreprocessingAdapterConfig = config
        self.steps: list[BasePreprocessing] = [step_config.build() for step_config in self.config.steps]

    def fit(self, data: BaseDataset) -> None:
        """ Fit each preprocessing step sequentially on the dataset, feeding the transformed data from the previous step
        to the next. """

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
        for step in self.steps:
            data = step.transform(data)
        return data
