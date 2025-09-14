from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    """Base interface for all pipeline steps."""

    def __call__(self, data):
        # Apply the standard pipeline flow: preprocess → run → postprocess
        data = self.pre_process(data)
        data = self.run(data)
        data = self.post_process(data)

        return data

    @abstractmethod
    def pre_process(self, data: Any) -> Any:
        """Standardize or adapt the input before executing the step."""
        pass

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Main logic of the step."""
        pass

    @abstractmethod
    def post_process(self, data: Any) -> Any:
        """Standardize or adapt the output after executing the step."""
        pass
