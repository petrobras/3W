from pathlib import Path
from typing import Mapping
from pydantic import Field

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import GradientBoostingClassifier


from ..core.base_models import BaseSkLearnModels, ModelsConfig
from ..core.enums import ModelTypeEnum
from ..utils.model_recorder import ModelRecorder

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
    """Sklearn model configuration. Use with SklearnTrainer for training."""

    model_params: dict[str, int | float | str | bool | None] = Field(
        default_factory=dict
    )
    target_: type = Field(default_factory=lambda: SklearnModels)


class SklearnModels(BaseSkLearnModels):
    """Sklearn model wrapper. Use SklearnTrainer for training."""

    def __init__(self, config: SklearnModelsConfig):
        self.config: SklearnModelsConfig = config

        model_class = SKLEARN_MODELS[config.model_type]
        params = config.model_params.copy()

        if "random_state" in model_class().get_params():
            params["random_state"] = config.random_seed

        self.model_class: BaseEstimator = model_class(**params)
        self._feature_names: list[str] | None = None

    @property
    def model_name(self) -> str:
        return self.model_class.__class__.__name__

    def save(self, filename: str | Path) -> None:
        """Save model to disk."""
        path = Path(filename)
        if path.exists() and path.suffix not in {".pkl", ".pickle"}:
            raise ValueError(
                "Sklearn models must be saved with .pkl or .pickle extension"
            )
        if not path.exists() and path.suffix == "":
            path = path.with_suffix(".pkl")
        ModelRecorder.save_model(model=self, filename=path)

    def load(self, filename: str | Path):
        """Load model from disk."""
        path = Path(filename)
        loaded = ModelRecorder.load_model(filename=path)
        if isinstance(loaded, SklearnModels):
            self.model_class = loaded.model_class
            self.config = loaded.config
        else:
            self.model_class = loaded

    def get_params(self) -> Mapping[str, object]:
        return self.model_class.get_params()

    def set_params(self, **params: object) -> None:
        self.model_class.set_params(**params)
