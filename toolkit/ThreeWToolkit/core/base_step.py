from abc import ABC, abstractmethod


class BaseStep(ABC):
    """Base interface for all pipeline steps."""

    @abstractmethod
    def run(self, data: dict) -> dict:
        """Main step logic.

        This method contains the core functionality of the step,
        performing the actual processing or transformation.

        Args:
            data (dict): Pre-processed input data.

        Returns:
            dict: Processed output data.
        """
        pass
