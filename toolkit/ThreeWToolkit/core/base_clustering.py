from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .enums import DistanceMetricEnum, LinkageMethodEnum


class InstanceQualityConfig(BaseModel):
    """Configuration for time series quality filtering.

    Controls detection of frozen sensors and NaN-corrupted instances,
    discarding those that exceed quality thresholds before clustering.
    """

    frozen_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Consecutive difference below this value is considered frozen.",
    )
    max_nan_ratio: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum allowed ratio of NaN samples per instance. Instances exceeding this are discarded.",
    )
    max_frozen_ratio: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum allowed ratio of frozen samples per instance. Instances exceeding this are discarded after repair attempt.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResamplingConfig(BaseModel):
    """Configuration for time series downsampling.

    Reduces series length to speed up DTW computation
    while preserving overall shape.
    """

    step_size: int = Field(
        default=100,
        gt=0,
        description="Number of original samples per output sample after resampling.",
    )
    step_method: Literal["slice", "mean"] = Field(
        default="slice",
        description="'slice' takes every n-th sample; 'mean' averages samples within each block.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TimeSeriesScalingConfig(BaseModel):
    """Configuration for per-instance Z-normalization.

    Standardizes each time series independently so
    clustering focuses on shape, not amplitude.
    """

    with_mean: bool = Field(
        default=True,
        description="Subtract the series mean if True.",
    )
    with_std: bool = Field(
        default=True,
        description="Divide by the series standard deviation if True.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DistanceMatrixConfig(BaseModel):
    """Configuration for pairwise distance computation.

    Supports DTW variants and Euclidean distance for
    comparing variable-length time series instances.
    """

    metric: DistanceMetricEnum = Field(
        default=DistanceMetricEnum.DTW,
        description="Distance metric used for comparison.",
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs. -1 uses all available CPU cores.",
    )
    return_condensed: bool = Field(
        default=False,
        description="If True, returns a 1-D condensed distance vector instead of a square matrix.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HierarchicalClusteringConfig(BaseModel):
    """Configuration for agglomerative hierarchical clustering.

    Builds a linkage tree from a distance matrix and
    supports threshold-based cluster extraction.
    """

    linkage_method: LinkageMethodEnum = Field(
        default=LinkageMethodEnum.AVERAGE,
        description="Linkage criterion used to compute the distance between clusters.",
    )
    default_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Normalized distance threshold used to cut the dendrogram when no threshold is explicitly provided.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DivisiveClusteringConfig(BaseModel):
    """Configuration for divisive outlier ranking.

    Uses recursive elimination of the most distant
    instance to produce an outlier-to-centroid ranking.
    No additional parameters are required.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MultivariateConsensusConfig(BaseModel):
    """Configuration for cross-variable consensus selection.

    Intersects per-variable cluster membership across
    a range of thresholds to find instances that are
    consistently well-behaved across all sensors.
    """

    min_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Start of the threshold sweep range.",
    )
    max_threshold: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="End of the threshold sweep range.",
    )
    threshold_step: float = Field(
        default=0.1,
        gt=0.0,
        description="Step size of the threshold sweep.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("max_threshold")
    @classmethod
    def max_threshold_greater_than_min(cls, v: float, info: ValidationInfo) -> float:
        """Validate that max_threshold exceeds min_threshold.

        Args:
            v (float): The max_threshold value.
            info (ValidationInfo): Validation context.

        Returns:
            float: The validated max_threshold value.

        Raises:
            ValueError: If max_threshold <= min_threshold.
        """
        min_t = info.data.get("min_threshold", 0.0)
        if v <= min_t:
            raise ValueError("max_threshold must be greater than min_threshold.")
        return v
