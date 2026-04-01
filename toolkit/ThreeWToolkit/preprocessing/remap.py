from pydantic import Field, PrivateAttr
from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class RemapClassConfig(BasePreprocessingConfig):
    class_map: dict | None = Field(
        default=None,
        description="Mapping from original class labels to new class labels. If None, it will be generated in fit() by collecting all unique classes across events.",
    )
    _target: type = PrivateAttr(default_factory=lambda: RemapClass)


class RemapClass(BasePreprocessing):
    """
    Preprocessing step to remap class/label values according to a provided mapping.
    If class_map is not provided, it will be generated in fit() by collecting all
    unique classes across events. Stores the original mapping for convenience.
    """

    def __init__(self, config: RemapClassConfig):
        self.config: RemapClassConfig = config

    def fit(self, data: BaseDataset) -> None:
        """
        Collect all unique classes from the label Series if class_map is not provided.
        """
        if self.config.class_map is not None:
            self.class_map = self.config.class_map
            return

        unique_classes = set()
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

        if data.label is not None:
            data.label = data.label.map(self.class_map)
            if data.label.isna().any():
                # If there are any labels that were not in the class_map, they will be NaN after mapping.
                raise ValueError("Some labels were not in the class_map.")

        return data
