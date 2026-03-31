"""SklearnTrainer for training scikit-learn models with datasets."""

import logging
import numpy as np
from pydantic import Field, field_validator
from ..core.base_trainer import BaseTrainer, BaseTrainerConfig
from ..core.base_dataset import BaseDataset
from ..core.base_models import BaseSkLearnModels
from ..models.sklearn_models import SklearnModelsConfig, SklearnModels

logger = logging.getLogger(__name__)


class SklearnTrainerConfig(BaseTrainerConfig):
    """Configuration for scikit-learn trainer."""

    config_model: SklearnModelsConfig = Field(
        ..., description="Sklearn model configuration"
    )
    n_jobs: int | None = Field(default=None, description="Number of parallel jobs")
    verbose: int = Field(default=0, ge=0, description="Verbosity level")
    target_: type = Field(default_factory=lambda: SklearnTrainer)

    @field_validator("n_jobs")
    @classmethod
    def check_n_jobs(cls, value: int | None) -> int | None:
        if value is not None and value == 0:
            raise ValueError("n_jobs cannot be 0")
        return value


class SklearnTrainer(BaseTrainer):
    """Scikit-learn trainer for classical ML models."""

    model: SklearnModels

    def __init__(self, config: SklearnTrainerConfig):
        super().__init__(config)
        self.config: SklearnTrainerConfig = config

        self.model = config.config_model.build()

        model_params = self.model.model_class.get_params()
        if "n_jobs" in model_params and config.n_jobs is not None:
            self.model.model_class.set_params(n_jobs=config.n_jobs)

        if "verbose" in model_params:
            self.model.model_class.set_params(verbose=config.verbose)

        logger.info(
            "SklearnTrainer initialized | model=%s | n_jobs=%s",
            self.model.model_name,
            config.n_jobs,
        )

    def _prepare_data_for_training(
        self, dataset: BaseDataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert dataset to numpy arrays (X, y)."""
        logger.info("Converting dataset to arrays (size=%d)", len(dataset))

        signals_list = []
        labels_list = []

        for idx in range(len(dataset)):
            event = dataset[idx]
            signal_array = event.signal.values

            if signal_array.ndim == 2:
                signals_list.append(signal_array)
            else:
                signals_list.append(signal_array.reshape(1, -1))

            if event.label is not None:
                label_array = event.label.values
                if hasattr(label_array, "__len__"):
                    labels_list.extend(label_array)
                else:
                    labels_list.append(label_array)

        X = np.concatenate(signals_list, axis=0)
        y = np.array(labels_list)

        logger.info("Created arrays | X.shape=%s | y.shape=%s", X.shape, y.shape)
        return X, y

    def _execute_training(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, list[float] | float]:
        """Execute sklearn fit() training."""
        X_train, y_train = train_data

        logger.info(
            "Training sklearn model | samples=%d | features=%d",
            X_train.shape[0],
            X_train.shape[1],
        )

        fit_params = {}

        if self.config.use_class_weights:
            model_params = self.model.model_class.get_params()

            if self.config.class_weight_strategy == "manual":
                if self.config.manual_class_weights is None:
                    raise ValueError(
                        "manual_class_weights required when strategy='manual'"
                    )
                class_weights = self.config.manual_class_weights
            else:
                unique_classes, class_counts = np.unique(y_train, return_counts=True)
                total_samples = len(y_train)
                n_classes = len(unique_classes)
                class_weights = {
                    int(cls): total_samples / (n_classes * count)
                    for cls, count in zip(unique_classes, class_counts)
                }

            if "class_weight" in model_params:
                self.model.model_class.set_params(class_weight=class_weights)
            elif (
                hasattr(self.model.model_class.fit, "__code__")
                and "sample_weight" in self.model.model_class.fit.__code__.co_varnames
            ):
                sample_weights = np.array(
                    [class_weights.get(int(label), 1.0) for label in y_train]
                )
                fit_params["sample_weight"] = sample_weights

        self.model.model_class.fit(X_train, y_train, **fit_params)

        logger.info("Training completed")

        history: dict[str, list[float] | float] = {}

        if val_data is not None:
            X_val, y_val = val_data
            val_score = self.model.model_class.score(X_val, y_val)
            history["val_score"] = val_score
            logger.info("Validation score: %.4f", val_score)

        return history
