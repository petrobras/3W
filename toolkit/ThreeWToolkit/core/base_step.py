from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TInput = TypeVar("TInput")
TPreProcessed = TypeVar("TPreProcessed")
TRunOutput = TypeVar("TRunOutput")
TOutput = TypeVar("TOutput")


class BaseStep(ABC, Generic[TInput, TPreProcessed, TRunOutput, TOutput]):
    """Base interface for all pipeline steps.

    This class allows for type transformations between methods:
    - pre_process: TInput -> TPreProcessed
    - run: TPreProcessed -> TRunOutput
    - post_process: TRunOutput -> TOutput
    """

    def __call__(self, data: TInput) -> TOutput:
        """Execute the step by running pre-processing, main logic, and post-processing.

        Args:
            data: Input data to be processed by the step.

        Returns:
            Processed data after all three phases.
        """
        preprocessed: TPreProcessed = self.pre_process(data)
        run_output: TRunOutput = self.run(preprocessed)
        output: TOutput = self.post_process(run_output)

        return output

    @abstractmethod
    def pre_process(self, data: TInput) -> TPreProcessed:
        """Standardize the step input.

        This method should validate and transform the input data into
        the expected format for the main processing logic.

        Args:
            data: Raw input data.

        Returns:
            Standardized input data.
        """
        pass

    @abstractmethod
    def run(self, data: TPreProcessed) -> TRunOutput:
        """Main step logic.

        This method contains the core functionality of the step,
        performing the actual processing or transformation.

        Args:
            data: Pre-processed input data.

        Returns:
            Processed output data.
        """
        pass

    @abstractmethod
    def post_process(self, data: TRunOutput) -> TOutput:
        """Standardize the step output.

        This method should validate and format the output data to ensure
        consistency with the expected output format.

        Args:
            data: Output from the main processing logic.

        Returns:
            Standardized output data.
        """
        pass
