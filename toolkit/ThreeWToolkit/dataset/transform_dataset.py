from ThreeWToolkit.core.base_dataset import BaseDataset
from ThreeWToolkit.core.dataset_outputs import DatasetOutputs
from pydantic import Field
from ..core.base_preprocessing import BasePreprocessingConfig
from ..core.base_feature_extractor import BaseFeatureExtractorConfig
from ..core.base_transform import BaseTransform, BaseTransformConfig
from ..preprocessing.remap import RemapClass
from .transformed_dataset import TransformedDataset


class TransformConfig(BaseTransformConfig):
    """Configuration for transforming a dataset."""

    pre_processing: BasePreprocessingConfig | None = Field(
        default=None,
        description="List of preprocessing steps to apply to the dataset.",
    )
    feature_extraction: BaseFeatureExtractorConfig | None = Field(
        default=None,
        description="List of feature extraction steps to apply to the dataset after preprocessing.",
    )
    target_: type = Field(default_factory=lambda: TransformDataset)


class TransformDataset(BaseTransform):
    """Class for fitting preprocessing and feature extraction steps on a dataset and applying the transformations."""

    def __init__(self, config: TransformConfig):
        self.config = config

        if self.config.pre_processing is not None:
            self.pre_processing_step = self.config.pre_processing.build()

        if self.config.feature_extraction is not None:
            self.feature_extraction_step = self.config.feature_extraction.build()

    def fit(self, dataset: BaseDataset) -> None:
        """
        Fit preprocessing and feature extraction steps on the dataset, collecting necessary statistics.
        It needs to run through the entire list of events in the order given by the user,
        so it will be slow if there are many steps that require statistics collection,
        but it will ensure that the statistics are collected in the correct order.
        """
        if self.pre_processing_step is not None:
            self.pre_processing_step.fit(dataset)

    def transform(self, dataset: BaseDataset) -> TransformedDataset:
        return TransformedDataset(dataset, self.transform_event)

    def transform_event(self, data: DatasetOutputs) -> DatasetOutputs:
        """Apply the fitted preprocessing and feature extraction steps to a single event."""

        if self.pre_processing_step is not None:
            data = self.pre_processing_step.transform(data)
        if self.feature_extraction_step is not None:
            data = self.feature_extraction_step.transform(data)
        return data

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes from the RemapClass step, if present."""
        if not hasattr(self, "pre_processing_step") or self.pre_processing_step is None:
            return None

        for step in self.pre_processing_step.steps:
            if isinstance(step, RemapClass) and hasattr(step, "class_map"):
                return len(step.class_map)
        return None
