import joblib
from typing import Dict, Any
from pydantic import Field
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import GradientBoostingClassifier

from ..core.base_models import BaseModels, ModelsConfig
from ..core.enums import ModelTypeEnum
from ..metrics import _classification


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

    model_params: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters for the scikit-learn model."
    )


class SklearnModels(BaseModels):
    """A wrapper for scikit-learn models."""

    def __init__(self, config: SklearnModelsConfig):
        """Initializes the wrapper and the underlying scikit-learn model."""
        super().__init__(config)

        model_class = SKLEARN_MODELS[config.model_type]

        # Prepare model parameters, including the random_seed as random_state
        params = config.model_params.copy()

        # Checks if the chosen sklearn model has the random_state parameter.
        if "random_state" in model_class().get_params():
            params["random_state"] = config.random_seed

        self.model = model_class(**params)

    def train(self, X: Any, y: Any = None) -> None:
        """Trains the model on the given data."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the chosen sklearn model."""
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the model using the toolkit's standardized metric functions
        and returns a dictionary of metrics.
        """
        predictions = self.predict(X)

        metrics = {
            "accuracy": _classification.accuracy_score(y_true=y, y_pred=predictions),
            "balanced_accuracy": _classification.balanced_accuracy_score(
                y_true=y, y_pred=predictions
            ),
            "precision_weighted": _classification.precision_score(
                y_true=y, y_pred=predictions, average="weighted"
            ),
            "recall_weighted": _classification.recall_score(
                y_true=y, y_pred=predictions, average="weighted"
            ),
            "f1_weighted": _classification.f1_score(
                y_true=y, y_pred=predictions, average="weighted"
            ),
        }

        # Conditionally calculate ROC AUC based on the problem type (binary vs multiclass)
        if hasattr(self.model, "predict_proba"):
            y_scores = self.predict_proba(X)

            # Check the number of classes from the probability matrix shape
            if y_scores.shape[1] == 2:
                # Binary Case
                # Pass only the probabilities of the positive class (class 1)
                metrics["roc_auc_score"] = _classification.roc_auc_score(
                    y_true=y, y_pred=y_scores[:, 1]
                )
            else:
                # Multiclass Case
                # Pass the full probability matrix and specify the strategy
                metrics["roc_auc_score"] = _classification.roc_auc_score(
                    y_true=y, y_pred=y_scores, multi_class="ovr"
                )

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Gets the model's parameters."""
        return self.model.get_params()

    def set_params(self, **params: Any) -> None:
        """Sets the model's parameters."""
        self.model.set_params(**params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Makes probability predictions (for classification)."""
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError(
                f"{self.model.__class__.__name__} does not support probability predictions."
            )
        return self.model.predict_proba(X)

    def save(self, filepath: str):
        """Saves the trained model to a file using joblib."""
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str, config: SklearnModelsConfig):
        """Loads a trained model from a file."""
        loaded_model = joblib.load(filepath)
        model_wrapper = cls(config)
        model_wrapper.model = loaded_model
        return model_wrapper
