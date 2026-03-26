from pydantic import Field
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig


class SequentialPreprocessingAdapterConfig(BasePreprocessingConfig):
    steps: list[BasePreprocessingConfig]
    target_: type = Field(default_factory=lambda: SequentialPreprocessingAdapter)


class SequentialPreprocessingAdapter(BasePreprocessing):
    """
    Applies a list of preprocessing steps sequentially to the input data (dict).
    Each step should be a BasePreprocessing instance.
    """

    def __init__(self, config: SequentialPreprocessingAdapterConfig):
        self.config = config
        self.preprocessing_steps = []
        for step_config in self.config.steps:
            step_instance = step_config.build()
            self.preprocessing_steps.append(step_instance)

    def fit(self, data: dict) -> None:
        for step in self.preprocessing_steps:
            step.fit(data)
            if hasattr(step, "compute"):
                step.compute()

    def compute(self) -> None:
        for step in self.preprocessing_steps:
            if hasattr(step, "compute"):
                step.compute()

    def transform(self, data: dict) -> dict:
        for step in self.preprocessing_steps:
            data = step.transform(data)
        return data
