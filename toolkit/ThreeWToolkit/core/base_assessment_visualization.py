from abc import ABC
from pydantic import BaseModel, Field, field_validator


class AssessmentVisualizationConfig(BaseModel):
    class_names: list[str] | None = Field(
        default=None, description="List containing class names"
    )

    @field_validator("class_names")
    @classmethod
    def validate_class_names(cls, v):
        if v is None:
            return v
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("class_names must be a non-empty list if provided.")

        if not all(isinstance(name, str) and name.strip() for name in v):
            raise ValueError("All elements in class_names must be non-empty strings.")
        return v


class BaseAssessmentVisualization(ABC):
    def __init__(self, config: AssessmentVisualizationConfig):
        self.config = config
