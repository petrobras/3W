import pytest
from pydantic import BaseModel, ValidationError

from ThreeWToolkit.utils.general_utils import GeneralUtils


class SampleSchema(BaseModel):
    """Sample Pydantic schema for testing."""

    x: int
    y: float
    name: str = "default"


class TestValidateFuncArgsWithPydantic:
    def test_valid_arguments(self):
        """Test that valid arguments pass validation."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        result = sample_func(x=10, y=3.14, name="test")
        assert result == "test: x=10, y=3.14"

    def test_valid_arguments_with_defaults(self):
        """Test that default values are applied correctly."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        result = sample_func(x=5, y=2.5)
        assert result == "default: x=5, y=2.5"

    def test_invalid_type_raises_error(self):
        """Test that invalid argument types raise ValidationError."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        with pytest.raises(ValidationError):
            sample_func(x="not_an_int", y=3.14)

    def test_missing_required_argument_raises_error(self):
        """Test that missing required arguments raise TypeError."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        with pytest.raises(TypeError):
            sample_func(x=10)  # Missing required 'y'

    def test_type_coercion(self):
        """Test that Pydantic performs type coercion when possible."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> tuple:
            return (x, y, name)

        # Pydantic should coerce "10" to int and 3 to float
        result = sample_func(x="10", y=3, name="coerced")
        assert result == (10, 3.0, "coerced")
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_extra_arguments_ignored(self):
        """Test that extra arguments not in schema raise TypeError."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        # Python's signature binding rejects extra arguments
        with pytest.raises(TypeError):
            sample_func(x=10, y=3.14, name="test", extra_arg=999)

    def test_positional_arguments(self):
        """Test that positional arguments work correctly."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        result = sample_func(42, 1.5, "positional")
        assert result == "positional: x=42, y=1.5"

    def test_mixed_positional_and_keyword_arguments(self):
        """Test that mixed positional and keyword arguments work."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def sample_func(x: int, y: float, name: str = "default") -> str:
            return f"{name}: x={x}, y={y}"

        result = sample_func(100, y=2.5)
        assert result == "default: x=100, y=2.5"


class ComplexSchema(BaseModel):
    """More complex schema for advanced testing."""

    values: list[int]
    threshold: float
    enabled: bool = True


class TestValidateFuncArgsComplex:
    def test_list_validation(self):
        """Test validation with list arguments."""

        @GeneralUtils.validate_func_args_with_pydantic(ComplexSchema)
        def process_values(values: list[int], threshold: float, enabled: bool = True):
            return sum(v for v in values if v > threshold) if enabled else 0

        result = process_values(values=[1, 5, 10, 15], threshold=7.0)
        assert result == 25  # 10 + 15

    def test_boolean_validation(self):
        """Test boolean argument validation."""

        @GeneralUtils.validate_func_args_with_pydantic(ComplexSchema)
        def process_values(values: list[int], threshold: float, enabled: bool = True):
            return sum(v for v in values if v > threshold) if enabled else 0

        result = process_values(values=[1, 5, 10], threshold=3.0, enabled=False)
        assert result == 0

    def test_invalid_list_elements(self):
        """Test that invalid list elements raise ValidationError."""

        @GeneralUtils.validate_func_args_with_pydantic(ComplexSchema)
        def process_values(values: list[int], threshold: float, enabled: bool = True):
            return sum(values)

        with pytest.raises(ValidationError):
            process_values(values=[1, 2, "three"], threshold=1.0)

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @GeneralUtils.validate_func_args_with_pydantic(SampleSchema)
        def documented_function(x: int, y: float, name: str = "default") -> str:
            """This is a documented function."""
            return f"{name}: {x}, {y}"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."
