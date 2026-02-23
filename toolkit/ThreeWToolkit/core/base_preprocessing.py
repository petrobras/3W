import pandas as pd

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ImputeMissingConfig(BaseModel):
    strategy: Literal["mean", "median", "constant"]
    fill_value: int | float | None = None
    columns: list[str] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("fill_value")
    @classmethod
    def check_fill_value_for_constant(
        cls: type["ImputeMissingConfig"],
        fill_value: int | float | None,
        info: ValidationInfo,
    ) -> int | float | None:
        """
        Validate that `fill_value` is provided when strategy is 'constant'.

        Args:
            cls (ImputeMissingConfig): The class reference.
            fill_value (int | float | None): Value to fill missing data with.
            info (ValidationInfo): Validation info containing the strategy.

        Returns:
            int | float | None: Validated fill value.

        Raises:
            ValueError: If strategy is 'constant' but fill_value is None.
        """
        strategy = info.data.get("strategy")
        if strategy == "constant" and fill_value is None:
            raise ValueError("You must provide `fill_value` when strategy='constant'")
        return fill_value


class NormalizeConfig(BaseModel):
    norm: Literal["l1", "l2", "max"] = "l2"
    axis: Literal[0, 1] = 0
    copy_values: bool = True
    return_norm_values: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WindowingConfig(BaseModel):
    window: str | tuple = "hann"
    window_size: int = Field(default=100, gt=1)
    overlap: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalize: bool = False
    fftbins: bool = True
    pad_last_window: bool = False
    pad_value: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("window")
    @classmethod
    def validate_window(
        cls: type["WindowingConfig"], window: str | tuple
    ) -> str | tuple:
        """
        Validate that the window configuration is correct.

        Args:
            cls (WindowingConfig): The class reference.
            window (str | tuple): Window name or tuple of (name, parameter).

        Returns:
            str | tuple: Validated window configuration.

        Raises:
            ValueError: If window name is invalid or required parameters are missing.
        """
        WINDOWS_WITH_REQUIRED_PARAMS = {
            "kaiser": 1,
            "kaiser_bessel_derived": 1,
            "gaussian": 1,
            "general_cosine": 1,
            "general_gaussian": 2,
            "general_hamming": 1,
            "dpss": 1,
            "chebwin": 1,
        }

        WINDOWS_WITH_OPTIONAL_OR_NO_PARAMS = {
            "boxcar",
            "triang",
            "blackman",
            "hamming",
            "hann",
            "bartlett",
            "flattop",
            "parzen",
            "bohman",
            "blackmanharris",
            "nuttall",
            "barthann",
            "cosine",
            "lanczos",
            "exponential",
            "tukey",
            "taylor",
        }

        ALL_WINDOW_NAMES = (
            set(WINDOWS_WITH_REQUIRED_PARAMS) | WINDOWS_WITH_OPTIONAL_OR_NO_PARAMS
        )

        if isinstance(window, str):
            if window not in ALL_WINDOW_NAMES:
                raise ValueError(f"Invalid window name '{window}'.")
            if window in WINDOWS_WITH_REQUIRED_PARAMS:
                raise ValueError(
                    f"Window '{window}' requires parameter(s); use a tuple like ('{window}', param)."
                )

        else:
            if len(window) == 0 or not isinstance(window[0], str):
                raise ValueError("Tuple window must start with a string window name.")

            name = window[0]
            params = window[1:]

            if name not in ALL_WINDOW_NAMES:
                raise ValueError(f"Unknown window name '{name}'.")

            if name in WINDOWS_WITH_REQUIRED_PARAMS:
                expected = WINDOWS_WITH_REQUIRED_PARAMS[name]
                if len(params) < expected:
                    raise ValueError(
                        f"Window '{name}' requires {expected} parameter(s), got {len(params)}."
                    )

        return window


class RenameColumnsConfig(BaseModel):
    columns_map: dict[str, str]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("columns_map")
    @classmethod
    def validate_columns_exist(
        cls: type["RenameColumnsConfig"],
        columns_map: dict[str, str],
        info: ValidationInfo,
    ) -> dict[str, str]:
        """
        Validate that all columns to be renamed exist in the DataFrame.

        Args:
            cls (RenameColumnsConfig): The class reference.
            columns_map (dict[str, str]): Mapping of columns to rename.
            info: Validation info, containing the data attribute.

        Raises:
            ValueError: If any column in columns_map does not exist in the DataFrame.

        Returns:
            dict[str, str]: The validated columns_map.
        """
        df: pd.DataFrame | None = info.data.get("data")
        if df is not None:
            missing = [col for col in columns_map if col not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
        return columns_map

    @field_validator("columns_map")
    @classmethod
    def validate_unique_new_column_names(
        cls: type["RenameColumnsConfig"], columns_map: dict[str, str]
    ) -> dict[str, str]:
        """
        Validate that new column names are unique.

        Args:
            cls (RenameColumnsConfig): The class reference.
            columns_map (dict[str, str]): Mapping of columns to rename.

        Raises:
            ValueError: If there are duplicate new column names.

        Returns:
            dict[str, str]: The validated columns_map.
        """
        new_names = list(columns_map.values())
        if len(new_names) != len(set(new_names)):
            raise ValueError("Duplicate new column names are not allowed.")
        return columns_map
