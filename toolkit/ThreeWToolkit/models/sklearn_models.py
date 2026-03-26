from pathlib import Path
from numpy.typing import ArrayLike

from typing import Mapping, Type
from pydantic import Field

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import GradientBoostingClassifier

from ..core.base_models import BaseModels, ModelsConfig
from ..core.enums import ModelTypeEnum
from ..utils.model_recorder import ModelRecorder
from ..assessment.strategies.sklearn_prediction_strategy import (
    SklearnPredictionStrategy,
)
from ..core.base_prediction_strategies import PredictionStrategy
from ..core.base_training_strategies import TrainingStrategy
from ..trainer.strategies.fit_once_strategy import FitOnceStrategy

# Dictionary to map the enum to the scikit-learn classes
SKLEARN_MODELS = {
    ModelTypeEnum.LOGISTIC_REGRESSION: LogisticRegression,
    ModelTypeEnum.DECISION_TREE: DecisionTreeClassifier,
    ModelTypeEnum.RANDOM_FOREST: RandomForestClassifier,
    ModelTypeEnum.SVM: SVC,
    ModelTypeEnum.KNN: KNeighborsClassifier,
    ModelTypeEnum.NAIVE_BAYES: ComplementNB,
    ModelTypeEnum.GRADIENT_BOOSTING: GradientBoostingClassifier,
}


class SklearnModelsConfig(ModelsConfig):
    """Configuration that extends the base ModelsConfig for scikit-learn models."""

    model_params: dict[str, int | float | str | bool | None] = Field(
        default_factory=dict, description="Hyperparameters for the scikit-learn model."
    )
    target_: type[BaseModels] = Field(default_factory=lambda: SklearnModels)


class SklearnModels(BaseModels):
    """
    Wrapper for scikit-learn models following the BaseModels interface.

    This class adapts sklearn estimators to the unified model interface,
    allowing them to be trained, evaluated and persisted using the same
    Strategy-based pipeline as torch models.
    """

    config: SklearnModelsConfig

    def __init__(self, config: SklearnModelsConfig):
        """
        Initialize the sklearn model wrapper.

        Args:
            config (SklearnModelsConfig): Configuration object defining
                the sklearn estimator type and its hyperparameters.
        """
        super().__init__(config)
        self.config = config

        model_class = SKLEARN_MODELS[config.model_type]
        params = config.model_params.copy()

        # Inject random_state if supported
        if "random_state" in model_class().get_params():
            params["random_state"] = config.random_seed

        self.model_class: BaseEstimator = model_class(**params)
        self._feature_names: list[str] | None = None

    @property
    def model_name(self) -> str:
        return self.model_class.__class__.__name__

    def forward(self, x: ArrayLike):
        """
        Sklearn model not implements forward function.
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save the sklearn model to disk.

        Args:
            path (Path): Destination file path (.pkl or .pickle).
        """
        if path.suffix not in {".pkl", ".pickle"}:
            raise ValueError(
                "Sklearn models must be saved with .pkl or .pickle extension"
            )

        ModelRecorder.save_best_model(
            model=self.model_class,
            filename=str(path),
        )

    def load(self, path: Path) -> "SklearnModels":
        """
        Load a sklearn model from disk into this instance.

        Args:
            path (Path): Path to the saved model file.

        Returns:
            SklearnModels: Current instance with loaded model.
        """
        self.model_class = ModelRecorder.load_model(filename=str(path))
        return self

    def get_training_strategy(self) -> Type[TrainingStrategy]:
        """
        Return the sklearn-compatible training strategy.

        Returns:
            Type[TrainingStrategy]
        """
        return FitOnceStrategy

    def get_prediction_strategy(self) -> Type[PredictionStrategy]:
        """
        Return the sklearn-compatible prediction strategy.

        Returns:
            Type[PredictionStrategy]
        """
        return SklearnPredictionStrategy

    def get_params(self) -> Mapping[str, object]:
        """Return sklearn estimator parameters."""
        return self.model_class.get_params()

    def set_params(self, **params: object) -> None:
        """Set sklearn estimator parameters."""
        self.model_class.set_params(**params)
