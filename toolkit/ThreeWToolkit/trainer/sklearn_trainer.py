import os
import logging
import numpy as np
from pydantic import Field, PrivateAttr
from ..core.base_trainer import (
    BaseTrainer,
    BaseTrainerConfig,
    PredictionResult,
    TrainingHistory,
)
from ..core.base_dataset import BaseDataset
from ..models.sklearn_models import SklearnModelsConfig, SklearnModels

logger = logging.getLogger(__name__)


class SklearnTrainerConfig(BaseTrainerConfig):
    """Configuration for scikit-learn trainer.

    Args:
        config_model: Sklearn model configuration
        n_jobs: Number of parallel jobs. Automatic to number of CPUs if not set.
        verbose: Verbosity level
        seed: Random seed for reproducibility
        use_class_weights: Whether to use class weights for imbalanced datasets
        class_weight_strategy: Strategy for calculating class weights ('balanced', 'manual', or 'none')
        manual_class_weights: Optional dictionary of class weights if strategy is 'manual'
    """

    config_model: SklearnModelsConfig = Field(
        ..., description="Sklearn model configuration"
    )
    n_jobs: int = Field(
        default=os.cpu_count() or 1,
        gt=0,
        description="Number of parallel jobs. Automatic to number of CPUs if not set.",
    )
    verbose: int = Field(default=0, ge=0, description="Verbosity level")

    _target: type = PrivateAttr(default_factory=lambda: SklearnTrainer)


class SklearnTrainer(BaseTrainer):
    """Scikit-learn trainer for classical ML models."""

    model: SklearnModels

    def __init__(self, config: SklearnTrainerConfig):
        """Initialize SklearnTrainer with given configuration. Builds the specified sklearn model and sets parameters.
        Args:
            config: SklearnTrainerConfig containing model configuration and training parameters
        """
        super().__init__(config)
        self.config: SklearnTrainerConfig = config

        self.model = config.config_model.build()  # type: ignore

        model_params = self.model.get_params()
        if "n_jobs" in model_params:
            self.model.set_params(n_jobs=config.n_jobs)

        if "verbose" in model_params:
            self.model.set_params(verbose=config.verbose)

        logger.info(
            "SklearnTrainer initialized | model=%s | n_jobs=%s",
            self.model.model_name,
            config.n_jobs,
        )

    def _prepare_data(
        self, dataset: BaseDataset, shuffle: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert dataset to numpy arrays (X, y).
        Args:
            dataset: BaseDataset containing signal and label data
            shuffle: Whether to shuffle the data (default: True). Sklearn models typically handle\
                    shuffling internally, so this parameter is not used in this implementation.
        Returns:
            Tuple of (X, y) where X is the feature array and y is the label array.
        """
        _ = shuffle  # Sklearn models typically handle shuffling internally

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
                labels_list.append(label_array)

        assert len(signals_list) == len(
            labels_list
        ), "Mismatch between signals and labels"

        X = np.concatenate(signals_list, axis=0)
        y = np.concatenate(labels_list, axis=0)

        logger.info("Created arrays | X.shape=%s | y.shape=%s", X.shape, y.shape)
        return X, y

    def _execute_training(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray] | None,
    ) -> TrainingHistory:
        """Execute sklearn fit() training.
        Args:
            train_data: Tuple of (X_train, y_train) for training
            val_data: Optional tuple of (X_val, y_val) for validation
        Returns:
            TrainingHistory containing training and validation scores
        """
        X_train, y_train = train_data

        logger.info(
            "Training sklearn model | samples=%d | features=%d",
            X_train.shape[0],
            X_train.shape[1],
        )

        fit_params = {}

        if self.config.use_class_weights:
            model_params = self.model.get_params()

            if self.config.class_weight_strategy == "manual":
                if self.config.manual_class_weights is None:
                    raise ValueError(
                        "manual_class_weights required when strategy='manual'"
                    )
                class_weights = self.config.manual_class_weights
            else:
                logger.info(
                    "Calculating class weights using strategy: %s",
                    self.config.class_weight_strategy,
                )
                unique_classes, class_counts = np.unique(y_train, return_counts=True)
                total_samples = len(y_train)
                n_classes = len(unique_classes)
                class_weights = {
                    int(cls): total_samples / (n_classes * count)
                    for cls, count in zip(unique_classes, class_counts)
                }

            if "class_weight" in model_params:
                self.model.set_params(class_weight=class_weights)
            elif (
                hasattr(self.model.model.fit, "__code__")
                and "sample_weight" in self.model.model.fit.__code__.co_varnames
            ):
                sample_weights = np.array(
                    [class_weights.get(int(label), 1.0) for label in y_train]
                )
                fit_params["sample_weight"] = sample_weights

        self.model.model.fit(X_train, y_train, **fit_params)
        train_score = self.model.model.score(X_train, y_train)
        logger.info("Training score: %.4f", train_score)

        val_score = None
        if val_data is not None:
            X_val, y_val = val_data
            val_score = self.model.model.score(X_val, y_val)
            logger.info("Validation score: %.4f", val_score)

        return TrainingHistory(
            train_loss=[train_score],
            val_loss=[val_score] if val_score is not None else None,
        )

    def predict(self, dataset: BaseDataset) -> PredictionResult:
        """Predict labels for given dataset.
        Args:
            dataset: BaseDataset containing signal data for prediction
        Returns:
            PredictionResult containing predicted labels and true labels
        """
        X, y_true = self._prepare_data(dataset)
        predictions = self.model.model.predict(X)
        return PredictionResult(y_pred=predictions, y_true=y_true)

    def _initialize_training_state(
        self, train_data, train_dataset: BaseDataset
    ) -> None:
        """Run any necessary initialization before training starts. For sklearn, there is no special state to
        initialize, so this is a no-op.
        Args:
            train_data: unused
            train_dataset: unused
        """
        return
