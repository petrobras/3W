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


class TaskTypeEnum(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DataSplitEnum(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CUSTOM = "custom"


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


class AvailableWaveletsEnum(str, Enum):
    HAAR = "haar"
    DB1 = "db1"
    DB2 = "db2"
    DB3 = "db3"
    DB4 = "db4"
    DB5 = "db5"
    DB6 = "db6"
    DB7 = "db7"
    DB8 = "db8"
    DB9 = "db9"
    DB10 = "db10"
    BIOR2_2 = "bior2.2"
    BIOR4_4 = "bior4.4"
    COIF2 = "coif2"
    COIF4 = "coif4"
    DMEY = "dmey"


class AvailableEWStatisticalFeaturesEnum(str, Enum):
    EW_MEAN = "ew_mean"
    EW_STD = "ew_std"
    EW_SKEW = "ew_skew"
    EW_KURT = "ew_kurt"
    EW_MIN = "ew_min"
    EW_1QRT = "ew_1qrt"
    EW_MED = "ew_med"
    EW_3QRT = "ew_3qrt"
    EW_MAX = "ew_max"


class AvailableStatisticalFeaturesEnum(str, Enum):
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    SKEW = "skew"
    KURT = "kurt"
    Q25 = "q25"
    Q75 = "q75"
    RANGE = "range"
    IQR = "iqr"
