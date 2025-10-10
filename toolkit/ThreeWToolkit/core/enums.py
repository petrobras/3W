from enum import Enum


class ModelTypeEnum(str, Enum):
    MLP = "MLP"
    LOGISTIC_REGRESSION = "LogisticRegression"
    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"
    SVM = "SVM"
    KNN = "KNN"
    DECISION_TREE = "DecisionTree"
    NAIVE_BAYES = "NaiveBayes"


class EventPrefixEnum(str, Enum):
    REAL = "WELL"
    SIMULATED = "SIMULATED"
    DRAWN = "DRAWN"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ActivationFunctionEnum(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class OptimizersEnum(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class CriterionEnum(Enum):
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    MSE = "mse"
    MAE = "mae"
