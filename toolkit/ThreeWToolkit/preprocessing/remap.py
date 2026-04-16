from pydantic import Field, PrivateAttr
from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class RemapClassConfig(BasePreprocessingConfig):
    """Configuration for remapping class labels to integers starting from 0."""

    class_map: dict | None = Field(
        default=None,
        description="Optional mapping from original class labels to new labels. Auto-generated from data if None.",
    )
    _target: type = PrivateAttr(default_factory=lambda: RemapClass)


class RemapClass(BasePreprocessing):
    """
    Preprocessing step to remap class/label values according to a provided mapping.
    If class_map is not provided, it will be generated in fit() by collecting all
    unique classes across events. Stores the original mapping for convenience.
    """

    def __init__(self, config: RemapClassConfig):
        """Initialize the RemapClass preprocessing step with the given configuration.

        Args:
            config: RemapClassConfig object containing the configuration for this preprocessing step.
        """
        self.config: RemapClassConfig = config

    def fit(self, data: BaseDataset) -> None:
        """
        Collect all unique classes from the label Series if class_map is not provided.
        If class_map is provided, this method will check that all classes in the data are present in the class_map.

        Args:
            data: BaseDataset object from which to collect unique class labels if class_map is not provided.
        """
        if self.config.class_map is not None:
            self.class_map = self.config.class_map

            for event in data:
                _ = self.transform(event)  # check that all labels can be mapped
            return

        unique_classes: set[int] = set()
        # collect unique classes across all events
        for event in data:
            if event.label is not None:
                unique_classes.update(event.label.dropna().unique())
        self.class_map = {c: i for i, c in enumerate(sorted(unique_classes))}

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """
        Remap class labels according to the class_map.

        Args:
            data: DatasetOutputs object with label field

        Returns:
            DatasetOutputs: Transformed data with remapped labels
        """
        if self.class_map is None:
            raise ValueError("RemapClass: class_map is not set. Call fit first.")

        mapped_label = data.label
        if data.label is not None:
            mapped_label = data.label.map(self.class_map)
            if mapped_label.isna().any():
                # If there are any labels that were not in the class_map, they will be NaN after mapping.
                raise ValueError("Some labels were not in the class_map.")

        return DatasetOutputs(
            signal=data.signal,
            label=mapped_label,
            metadata=data.metadata.copy(),
        )
