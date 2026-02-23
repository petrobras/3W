import numpy as np
import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
    ValidationInfo,
)


class BaseScoreArgsValidator(BaseModel):
    y_true: np.ndarray | pd.Series | list
    y_pred: np.ndarray | pd.Series | list
    sample_weight: np.ndarray | pd.Series | list | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("y_true", "y_pred", "sample_weight", mode="before")
    @classmethod
    def check_array_like(
        cls: type["BaseScoreArgsValidator"], value: object, info: ValidationInfo
    ) -> object:
        """
        Validate that input fields are array-like.

        Args:
            cls (BaseScoreArgsValidator): The class reference.
            value (object): The value to validate.
            info (ValidationInfo): Validation information.

        Returns:
            object: The validated value.

        Raises:
            TypeError: If the value is not a numpy array, pandas Series, or list.
        """
        if info.field_name == "sample_weight" and value is None:
            return value
        if not isinstance(value, (np.ndarray, pd.Series, list)):
            raise TypeError(
                f"'{info.field_name}' must be a np.ndarray, pd.Series or list, got {type(value)}"
            )
        return value

    @model_validator(mode="after")
    def check_shapes(self) -> "BaseScoreArgsValidator":
        """
        Validate that y_true, y_pred, and sample_weight have consistent lengths.

        Returns:
            self: The model instance if validation succeeds.

        Raises:
            ValueError: If lengths of inputs do not match.
        """
        ytrue_size = len(self.y_true)
        ypred_size = len(self.y_pred)

        if ytrue_size != ypred_size:
            raise ValueError(
                f"'y_true' and 'y_pred' must have the same number of elements "
                f"(received: {ytrue_size} and {ypred_size})"
            )
        if self.sample_weight is not None:
            sample_weight_size = len(self.sample_weight)
            if sample_weight_size != ytrue_size:
                raise ValueError(
                    f"'sample_weight' must have the same number of elements as 'y_true' "
                    f"(received: {sample_weight_size} and {ytrue_size})"
                )
        return self


class LabelsValidator(BaseModel):
    labels: list | None = None

    @field_validator("labels", mode="before")
    @classmethod
    def check_labels(cls: type["LabelsValidator"], value: list | None) -> list | None:
        """
        Validate that labels are a list or None.

        Args:
            cls (LabelsValidator): The class reference.
            value (list | None): The labels to validate.

        Returns:
            list | None: The validated labels.

        Raises:
            TypeError: If labels is not a list and not None.
        """
        if value is not None and not isinstance(value, list):
            raise TypeError(f"'labels' must be a list or None, got {type(value)}")
        return value


class PosLabelValidator(BaseModel):
    pos_label: int = 1

    @field_validator("pos_label", mode="before")
    @classmethod
    def check_pos_label(
        cls: type["PosLabelValidator"], value: int | float | None
    ) -> int | float | None:
        """
        Validate that pos_label is a number or None.

        Args:
            cls (PosLabelValidator): The class reference.
            value (int | float | None): The pos_label to validate.

        Returns:
            int | float | None: The validated pos_label.

        Raises:
            TypeError: If pos_label is not a number and not None.
        """
        if not isinstance(value, (int, float)) and value is not None:
            raise TypeError(f"'pos_label' must be a number or None, got {type(value)}")
        return value


class AverageValidator(BaseModel):
    average: str | None = "binary"

    @field_validator("average", mode="before")
    @classmethod
    def check_average(cls: type["AverageValidator"], value: str | None) -> str | None:
        """
        Validate the average parameter.

        Args:
            cls (AverageValidator): The class reference.
            value (str | None): The average method to use.

        Returns:
            str | None: The validated average method.

        Raises:
            ValueError: If average is not a valid option.
        """
        allowed = {"micro", "macro", "samples", "weighted", "binary", None}
        if value not in allowed:
            raise ValueError(f"'average' must be one of {allowed}, got '{value}'")
        return value


class ZeroDivisionValidator(BaseModel):
    zero_division: str | int = "warn"

    @field_validator("zero_division", mode="before")
    @classmethod
    def check_zero_division(
        cls: type["ZeroDivisionValidator"], value: str | int
    ) -> str | int:
        """
        Validate the zero_division parameter.

        Args:
            cls (ZeroDivisionValidator): The class reference.
            value (str | int): The zero_division handling strategy.

        Returns:
            str | int: The validated zero_division value.

        Raises:
            ValueError: If zero_division is not valid.
        """
        allowed = {"warn", 0, 1}
        if value not in allowed:
            raise ValueError(f"'zero_division' must be one of {allowed}, got '{value}'")
        return value


class AccuracyScoreConfig(BaseScoreArgsValidator):
    normalize: bool = True

    @field_validator("normalize", mode="before")
    @classmethod
    def check_bool(cls: type["AccuracyScoreConfig"], value: bool) -> bool:
        """
        Validate boolean parameters.

        Args:
            cls (AccuracyScoreConfig): The class reference.
            value (bool): The value to validate.

        Returns:
            bool: The validated boolean value.

        Raises:
            TypeError: If the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError(
                f"'normalize' must be a boolean, got {type(value)} with value '{value}'"
            )
        return value


class BalancedAccuracyScoreConfig(BaseScoreArgsValidator):
    adjusted: bool = False

    @field_validator("adjusted", mode="before")
    @classmethod
    def check_bool(cls: type["BalancedAccuracyScoreConfig"], value: bool) -> bool:
        """
        Validate boolean parameters.

        Args:
            cls (BalancedAccuracyScoreConfig): The class reference.
            value (bool): The value to validate.

        Returns:
            bool: The validated boolean value.

        Raises:
            TypeError: If the value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError(
                f"'adjusted' must be a boolean, got {type(value)} with value '{value}'"
            )
        return value


class AveragePrecisionScoreConfig(
    BaseScoreArgsValidator, AverageValidator, PosLabelValidator
):
    @field_validator("average", mode="before")
    @classmethod
    def override_average_options(
        cls: type["AveragePrecisionScoreConfig"], value: str | None
    ) -> str | None:
        """
        Override average options for AveragePrecisionScore.

        Args:
            cls (AveragePrecisionScoreConfig): The class reference.
            value (str | None): The average method to use.

        Returns:
            str | None: The validated average method.

        Raises:
            ValueError: If average is not a valid option.
        """
        allowed = {"micro", "macro", "samples", "weighted", None}
        if value not in allowed:
            raise ValueError(f"'average' must be one of {allowed}, got '{value}'")
        return value


class MultiClassValidator(BaseModel):
    multi_class: str = "raise"

    @field_validator("multi_class", mode="before")
    @classmethod
    def check_multi_class(cls: type["MultiClassValidator"], value: str) -> str:
        """
        Validate the multi_class parameter.

        Args:
            cls (MultiClassValidator): The class reference.
            value (str): The multi_class strategy.

        Returns:
            str: The validated multi_class strategy.

        Raises:
            ValueError: If multi_class is not a valid option.
        """
        allowed = {"raise", "ovr", "ovo"}
        if value not in allowed:
            raise ValueError(f"'multi_class' must be one of {allowed}, got '{value}'")
        return value


class MaxFprValidator(BaseModel):
    max_fpr: float | None = None

    @field_validator("max_fpr", mode="before")
    @classmethod
    def check_max_fpr(
        cls: type["MaxFprValidator"], value: float | None
    ) -> float | None:
        """
        Validate the max_fpr parameter.

        Args:
            cls (MaxFprValidator): The class reference.
            value (float | None): The max_fpr value.

        Returns:
            float | None: The validated max_fpr value.

        Raises:
            ValueError: If max_fpr is not in range (0, 1].
        """
        if value is not None and (
            not isinstance(value, (int, float)) or not (0 < value <= 1)
        ):
            raise ValueError(
                f"'max_fpr' must be a float in the range (0, 1], got {value}"
            )
        return value


class PrecisionScoreConfig(
    BaseScoreArgsValidator,
    LabelsValidator,
    PosLabelValidator,
    AverageValidator,
    ZeroDivisionValidator,
):
    pass


class RecallScoreConfig(
    BaseScoreArgsValidator,
    LabelsValidator,
    PosLabelValidator,
    AverageValidator,
    ZeroDivisionValidator,
):
    pass


class F1ScoreConfig(
    BaseScoreArgsValidator,
    LabelsValidator,
    PosLabelValidator,
    AverageValidator,
    ZeroDivisionValidator,
):
    pass


class RocAucScoreConfig(
    BaseScoreArgsValidator,
    AverageValidator,
    MaxFprValidator,
    MultiClassValidator,
    LabelsValidator,
):
    @field_validator("average", mode="before")
    @classmethod
    def override_average_options(
        cls: type["RocAucScoreConfig"], value: str | None
    ) -> str | None:
        """
        Override average options for RocAucScore.

        Args:
            cls (RocAucScoreConfig): The class reference.
            value (str | None): The average method to use.

        Returns:
            str | None: The validated average method.

        Raises:
            ValueError: If average is not a valid option.
        """
        allowed = {"micro", "macro", "samples", "weighted", None}
        if value not in allowed:
            raise ValueError(f"'average' must be one of {allowed}, got '{value}'")
        return value


class MultiOutputValidator(BaseModel):
    multioutput: str = "uniform_average"

    @field_validator("multioutput", mode="before")
    @classmethod
    def check_multioutput(cls: type["MultiOutputValidator"], value: str) -> str:
        """
        Validate the multioutput parameter.

        Args:
            cls (MultiOutputValidator): The class reference.
            value (str): The multioutput strategy.

        Returns:
            str: The validated multioutput strategy.

        Raises:
            ValueError: If multioutput is not a valid option.
        """
        allowed = {"raw_values", "uniform_average", "variance_weighted"}
        if value not in allowed:
            raise ValueError(f"'multioutput' must be one of {allowed}, got '{value}'")
        return value


class ForceFiniteValidator(BaseModel):
    force_finite: bool = True

    @field_validator("force_finite", mode="before")
    @classmethod
    def check_force_finite(cls: type["ForceFiniteValidator"], value: bool) -> bool:
        """
        Validate the force_finite parameter.

        Args:
            cls (ForceFiniteValidator): The class reference.
            value (bool): The force_finite flag.

        Returns:
            bool: The validated force_finite flag.

        Raises:
            TypeError: If force_finite is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError(f"'force_finite' must be a boolean, got {type(value)}")
        return value


class ExplainedVarianceScoreConfig(
    BaseScoreArgsValidator, MultiOutputValidator, ForceFiniteValidator
):
    pass
