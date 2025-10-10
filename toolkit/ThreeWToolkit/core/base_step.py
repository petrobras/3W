from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    """Base interface for all pipeline steps."""

    def __call__(self, data):
        """Execute the step by running pre-processing, main logic, and post-processing.

        Args:
            data: Input data to be processed by the step.

        Returns:
            Processed data after all three phases.
        """
        data = self.pre_process(data)
        data = self.run(data)
        data = self.post_process(data)

        return data

    @abstractmethod
    def pre_process(self, data: Any) -> Any:
        """Standardize the step input.

        This method should validate and transform the input data into
        the expected format for the main processing logic.

        Args:
            data (Any): Raw input data.

        Returns:
            Any: Standardized input data.
        """
        pass

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Main step logic.

        This method contains the core functionality of the step,
        performing the actual processing or transformation.

        Args:
            data (Any): Pre-processed input data.

        Returns:
            Any: Processed output data.
        """
        pass

    @abstractmethod
    def post_process(self, data: Any) -> Any:
        """Standardize the step output.

        This method should validate and format the output data to ensure
        consistency with the expected output format.

        Args:
            data (Any): Output from the main processing logic.

        Returns:
            Any: Standardized output data.
        """
        pass
