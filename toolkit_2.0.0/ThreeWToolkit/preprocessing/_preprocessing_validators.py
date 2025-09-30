import pandas as pd

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Literal, Optional, Union, List, Dict


class ImputeMissingArgsValidator(BaseModel):
    data: Union[pd.DataFrame, pd.Series]
    strategy: Literal["mean", "median", "constant"]
    fill_value: Optional[Union[int, float]] = None
    columns: Optional[List[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("fill_value")
    def check_fill_value_for_constant(cls, v, info):
        strategy = info.data.get("strategy")
        if strategy == "constant" and v is None:
            raise ValueError("You must provide `fill_value` when strategy='constant'")
        return v


class NormalizeArgsValidator(BaseModel):
    X: Union[pd.DataFrame, pd.Series]
    norm: Literal["l1", "l2", "max"] = "l2"
    axis: Optional[Literal[0, 1]] = 1
    copy_values: Optional[bool] = True
    return_norm_values: Optional[bool] = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("X")
    def validate_numeric(cls, v):
        if isinstance(v, pd.Series):
            if not pd.api.types.is_numeric_dtype(v):
                raise TypeError("Series must be numeric.")

        else:
            non_numeric = [
                col for col in v.columns if not pd.api.types.is_numeric_dtype(v[col])
            ]
            if non_numeric:
                raise TypeError(
                    f"All columns must be numeric. Non-numeric columns: {non_numeric}"
                )

        return v


class WindowingArgsValidator(BaseModel):
    X: Union[pd.Series, pd.DataFrame]
    window: Union[str, tuple] = "hann"
    window_size: int = Field(..., gt=1)
    overlap: float = Field(0.0, ge=0.0, lt=1.0)
    normalize: bool = False
    fftbins: bool = True
    pad_last_window: bool = False
    pad_value: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("X")
    def validate_numeric(cls, v):
        if not pd.api.types.is_numeric_dtype(v):
            raise TypeError("Series must be numeric.")
        return v

    @field_validator("window")
    def validate_window(cls, v):
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

        if isinstance(v, str):
            if v not in ALL_WINDOW_NAMES:
                raise ValueError(f"Invalid window name '{v}'.")
            if v in WINDOWS_WITH_REQUIRED_PARAMS:
                raise ValueError(
                    f"Window '{v}' requires parameter(s); use a tuple like ('{v}', param)."
                )

        else:
            if len(v) == 0 or not isinstance(v[0], str):
                raise ValueError("Tuple window must start with a string window name.")

            name = v[0]
            params = v[1:]

            if name not in ALL_WINDOW_NAMES:
                raise ValueError(f"Unknown window name '{name}'.")

            if name in WINDOWS_WITH_REQUIRED_PARAMS:
                expected = WINDOWS_WITH_REQUIRED_PARAMS[name]
                if len(params) < expected:
                    raise ValueError(
                        f"Window '{name}' requires {expected} parameter(s), got {len(params)}."
                    )

        return v

    @model_validator(mode="after")
    def check_window_size_vs_data(self):
        n_samples = len(self.X)
        if self.window_size > n_samples:
            raise ValueError(
                "`window_size` must be smaller than or equal to the length of X."
            )
        return self


class RenameColumnsArgsValidator(BaseModel):
    data: pd.DataFrame
    columns_map: Dict[str, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("columns_map")
    def validate_columns_exist(cls, columns_map, info: Any):
        """
        Validate that all columns to be renamed exist in the DataFrame.

        Args:
            cls: The class reference.
            columns_map (Dict[str, str]): Mapping of columns to rename.
            info: Validation info, containing the data attribute.

        Raises:
            ValueError: If any column in columns_map does not exist in the DataFrame.

        Returns:
            Dict[str, str]: The validated columns_map.
        """
        df: pd.DataFrame = info.data.get("data")
        if df is not None:
            missing = [col for col in columns_map if col not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
        return columns_map

    @field_validator("columns_map")
    def validate_unique_new_column_names(cls, columns_map):
        """
        Validate that new column names are unique.

        Args:
            cls: The class reference.
            columns_map (Dict[str, str]): Mapping of columns to rename.

        Raises:
            ValueError: If there are duplicate new column names.

        Returns:
            Dict[str, str]: The validated columns_map.
        """
        new_names = list(columns_map.values())
        if len(new_names) != len(set(new_names)):
            raise ValueError("Duplicate new column names are not allowed.")
        return columns_map

    @field_validator("data")
    def validate_no_duplicate_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the DataFrame does not contain duplicate column names.

        Args:
            cls: The class reference.
            df (pd.DataFrame): The input DataFrame.

        Raises:
            ValueError: If the DataFrame has duplicate column names.

        Returns:
            pd.DataFrame: The validated DataFrame.
        """
        duplicated = df.columns[df.columns.duplicated()].unique().tolist()
        if duplicated:
            raise ValueError(f"Duplicate column names found in DataFrame: {duplicated}")
        return df
