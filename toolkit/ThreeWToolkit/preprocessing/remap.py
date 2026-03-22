from pydantic import Field
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
import pandas as pd


class RemapClassConfig(BasePreprocessingConfig):
    class_map: dict | None = None
    target_: type = Field(default_factory=lambda: RemapClass)


class RemapClass(BasePreprocessing):
    """
    Preprocessing step to remap class/label values according to a provided mapping.
    If class_map is not provided, it will be generated in fit() by collecting all unique classes across events.
    Transient classes (>=100) are mapped to the end of the normal classes.
    Stores the original mapping for convenience.
    """

    def __init__(self, config: RemapClassConfig):
        self.config = config
        self.class_map = config.class_map
        self.original_map = None
        self._fit_required = self.class_map is None
        self._seen_classes = set() if self._fit_required else None

    def fit(self, data: dict) -> None:
        """
        Collect all unique classes from the data (dict or DataFrame) if class_map is not provided.
        """
        if not self._fit_required:
            return

        label = data.get("label")
        if label is not None and "class" in label.columns:
            class_col = label["class"]
            if self._seen_classes is not None:
                self._seen_classes.update(class_col.unique())

    def compute(self) -> None:
        """
        After fit, build the class_map and original_map.
        Print debug info about initial classes and final mapping.
        """
        if not self._fit_required:
            return
        if not self._seen_classes:
            raise ValueError("No classes were seen during fit. Cannot build class_map.")

        print("[RemapClass] Initial classes seen:", self._seen_classes)

        for c in self._seen_classes:
            if pd.isna(c):
                raise ValueError(
                    "[RemapClass] NA value detected in class labels. Please handle missing values with the ImputeMissing preprocessing step before using RemapClass."
                )

        # Map normal classes to 0..N-1, transient to N..N+T-1
        class_map = {c: i for i, c in enumerate(self._seen_classes)}
        print("[RemapClass] Final class_map:", class_map)
        self.class_map = class_map
        self.original_map = {v: k for k, v in class_map.items()}
        self._fit_required = False

    def transform(self, data: dict) -> dict:
        if self.class_map is None:
            raise ValueError(
                "RemapClass: class_map is not set. Call fit and compute first."
            )
        if isinstance(data, dict):
            label = data.get("label")
            if label is not None:
                data = data.copy()
                data["label"] = label.map(self.class_map)

            return data
        else:
            raise TypeError("Input to RemapClass.transform must be a DataFrame")
