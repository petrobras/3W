import torch

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from ThreeWToolkit.constants import OUTPUT_DIR

from ..core.enums import TaskType

from .base_step import BaseStep


class ModelAssessmentConfig(BaseModel):
    """
    Configuration for model assessment and evaluation.

    Args:
        metrics (list[str]): List of metric names to calculate.
        output_dir (Path): Directory to save assessment results.
        export_results (bool): Whether to export results to CSV.
        generate_report (bool): Whether to generate LaTeX report.
        task_type (str): Type of task (TaskType.CLASSIFICATION or TaskType.REGRESSION).
        batch_size (int): Batch size for PyTorch model predictions.
        device (str): Device for PyTorch computations.
        report_title (str | None): Title for the report.
        report_author (str): Author name for the report.

    Example:
        >>> config = ModelAssessmentConfig(
        ...     metrics=["accuracy", "f1", "precision", "recall"],
        ...     output_dir=Path("./results"),
        ...     task_type=TaskType.CLASSIFICATION,
        ...     generate_report=True,
        ...     report_title="Model Performance Analysis"
        ... )
    """

    metrics: list[str] = Field(
        default=["accuracy", "f1"], description="List of metric names to calculate"
    )
    output_dir: Path = Field(
        default=Path(OUTPUT_DIR), description="Directory to save assessment results"
    )
    export_results: bool = Field(
        default=True, description="Whether to export results to CSV files"
    )
    generate_report: bool = Field(
        default=False,
        description="Whether to generate LaTeX report using ReportGeneration",
    )
    task_type: TaskType = Field(
        default=TaskType.CLASSIFICATION,
        description="Type of task (classification or regression)",
    )
    batch_size: int = Field(
        default=64, gt=0, description="Batch size for PyTorch model predictions"
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for PyTorch computations",
    )
    report_title: Optional[str] = Field(
        default=None, description="Title for the generated report"
    )
    report_author: str = Field(
        default="3W Toolkit Report", description="Author name for the report"
    )

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v):
        valid_types = {TaskType.CLASSIFICATION, TaskType.REGRESSION}
        if v not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}")
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
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
        invalid_metrics = set(v) - valid_metrics
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Valid metrics: {valid_metrics}"
            )
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        valid_devices = {"cpu", "cuda"}
        if v not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        return v


class BaseModelAssessment(BaseStep):
    def __init__(self, config: ModelAssessmentConfig):
        self.config = config
