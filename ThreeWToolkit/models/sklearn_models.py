from typing import Dict, Any, List, Callable, Optional
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
from ..utils.model_recorder import ModelRecorder 
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
    
    SUPPORTED_METRICS = {
        _classification.accuracy_score,
        _classification.balanced_accuracy_score,
        _classification.precision_score,
        _classification.recall_score,
        _classification.f1_score,
        _classification.roc_auc_score,
        _classification.average_precision_score
    }

    def __init__(self, config: SklearnModelsConfig):
        """Initializes the wrapper and the underlying scikit-learn model."""
        super().__init__(config)
        model_class = SKLEARN_MODELS[config.model_type]
        params = config.model_params.copy()
        if "random_state" in model_class().get_params():
            params["random_state"] = config.random_seed
        self.model = model_class(**params)

    def train(self, x: Any, y: Any = None, **kwargs) -> None:
        """Trains the model on the given data."""
        self.model.fit(x, y, **kwargs)

    def predict(self, x: Any) -> Any:
        """Makes predictions using the chosen sklearn model."""
        return self.model.predict(x)
    
    def evaluate(self, x: Any, y: Any, metrics: List[Callable]):
        """
        Evaluates the model using a provided list of metric functions.
        """
        results: Dict[str, Optional[float]] = {}
        # Get standard class predictions first
        predictions = self.predict(x)
        
        y_scores = None
        if hasattr(self.model, "predict_proba"):
            y_scores = self.predict_proba(x)

        for metric_func in metrics:
            metric_name = metric_func.__name__
            
            if "roc_auc_score" in metric_name or "average_precision_score" in metric_name:
                # Check if scores were successfully calculated beforehand
                if y_scores is not None:
                    if y_scores.shape[1] == 2: # Binary case
                        results[metric_name] = metric_func(y_true=y, y_pred=y_scores[:, 1])
                    else: # Multiclass case
                        results[metric_name] = metric_func(y_true=y, y_pred=y_scores, multi_class="ovr")
                else:
                    # Model does not support predict_proba, so result is None
                    results[metric_name] = None
            else:
                # For all other metrics, pass the standard class predictions
                results[metric_name] = metric_func(y_true=y, y_pred=predictions)
        
        return results

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
        """
        Saves the trained model to a file using the toolkit's ModelRecorder.
        """
        if not (filepath.endswith(".pkl") or filepath.endswith(".pickle")):
            raise ValueError("Filename for scikit-learn models must end with .pkl or .pickle")
        
        ModelRecorder.save_best_model(model=self.model, filename=filepath)

    @classmethod
    def load(cls, filepath: str, config: SklearnModelsConfig):
        """
        Loads a trained model from a file using the toolkit's ModelRecorder.
        """
        # The recorder will load the raw scikit-learn model object
        loaded_model = ModelRecorder.load_model(filename=filepath)
        
        # Create a new wrapper instance and inject the loaded model
        model_wrapper = cls(config)
        model_wrapper.model = loaded_model
        return model_wrapper
