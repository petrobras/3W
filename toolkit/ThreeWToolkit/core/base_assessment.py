import torch

from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from ..constants import OUTPUT_DIR
from ..core.enums import DataSplitEnum, TaskTypeEnum


class ModelAssessmentConfig(BaseModel):
    """
    Configuration for model assessment and evaluation.

    Args:
        metrics (list[str]): List of metric names to calculate.
        output_dir (Path): Directory to save assessment results.
        export_results (bool): Whether to export results to CSV files.
        generate_report (bool): Whether to generate LaTeX report using ReportGeneration.
        task_type (TaskTypeEnum): Type of task (TaskTypeEnum.CLASSIFICATION or TaskTypeEnum.REGRESSION).
        batch_size (int): Batch size for PyTorch model predictions.
        device (str): Device for PyTorch computations.
        report_title (str | None): Title for the report.
        report_author (str): Author name for the report.
        dataset_split (DataSplitEnum): Type of dataset to be evaluated.

    Example:
        >>> config = ModelAssessmentConfig(
        ...     metrics=["accuracy", "f1", "precision", "recall"],
        ...     output_dir=Path("./results"),
        ...     task_type=TaskTypeEnum.CLASSIFICATION,
        ...     generate_report=True,
        ...     report_title="Model Performance Analysis"
        ... )
    """

    metrics: list[str] = Field(default=["accuracy", "f1"])
    output_dir: Path = Field(default=Path(OUTPUT_DIR))
    export_results: bool = Field(default=True)
    generate_report: bool = Field(default=False)
    task_type: TaskTypeEnum = Field(default=TaskTypeEnum.CLASSIFICATION)
    batch_size: int = Field(default=64, gt=0)
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    report_title: str | None = Field(default=None)
    report_author: str = Field(default="3W Toolkit Report")
    dataset_split: DataSplitEnum = Field(default=DataSplitEnum.TEST)

    @field_validator("task_type")
    @classmethod
    def validate_task_type(
        cls: type["ModelAssessmentConfig"], task_type: TaskTypeEnum
    ) -> TaskTypeEnum:
        """
        Validate that the task type is supported.

        Args:
            cls (ModelAssessmentConfig): The class reference.
            task_type (TaskTypeEnum): Task type to validate.

        Returns:
            TaskTypeEnum: Validated task type.

        Raises:
            ValueError: If task_type is not supported.
        """
        valid_types = {TaskTypeEnum.CLASSIFICATION, TaskTypeEnum.REGRESSION}
        if task_type not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}")
        return task_type

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls: type["ModelAssessmentConfig"], metrics: list[str]
    ) -> list[str]:
        """
        Validate that the requested metrics are supported.

        Args:
            cls (ModelAssessmentConfig): The class reference.
            metrics (list[str]): List of metric names.

        Returns:
            list[str]: Validated list of metrics.

        Raises:
            ValueError: If any metric is not supported.
        """
        valid_metrics = {
            # Classification metrics
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            # Regression metrics
            "explained_variance",
        }
        invalid_metrics = set(metrics) - valid_metrics
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}"
            )
        return metrics

    @field_validator("device")
    @classmethod
    def validate_device(cls: type["ModelAssessmentConfig"], device: str) -> str:
        """
        Validate that the computation device is supported.

        Args:
            cls (ModelAssessmentConfig): The class reference.
            device (str): Device name ('cpu' or 'cuda').

        Returns:
            str: Validated device name.

        Raises:
            ValueError: If device is not supported.
        """
        valid_devices = {"cpu", "cuda"}
        if device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        return device
