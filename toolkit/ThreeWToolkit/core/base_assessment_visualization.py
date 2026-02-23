from abc import ABC
from pydantic import BaseModel, Field, field_validator


class AssessmentVisualizationConfig(BaseModel):
    class_names: list[str] | None = Field(
        default=None, description="List containing class names"
    )

    @field_validator("class_names")
    @classmethod
    def validate_class_names(
        cls: type["AssessmentVisualizationConfig"], class_names: list[str] | None
    ) -> list[str] | None:
        """
        Validate that the provided class names are valid.

        Args:
            cls (AssessmentVisualizationConfig): The class reference.
            class_names (list[str] | None): List of class names to validate.

        Returns:
            list[str] | None: Validated list of class names or None.

        Raises:
            ValueError: If the list is empty or contains non-string elements.
        """
        if class_names is None:
            return class_names
        if not isinstance(class_names, list) or len(class_names) == 0:
            raise ValueError("class_names must be a non-empty list if provided.")

        if not all(isinstance(name, str) and name.strip() for name in class_names):
            raise ValueError("All elements in class_names must be non-empty strings.")
        return class_names


class BaseAssessmentVisualization(ABC):
    def __init__(self, config: AssessmentVisualizationConfig):
        self.config = config
