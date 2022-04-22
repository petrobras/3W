import toolkit
import numpy as np
from sklearn.preprocessing import LabelEncoder

experiment = toolkit.Experiment(event_name="SPURIOUS_CLOSURE_OF_DHSV")

codigos = list(experiment.codigos_classes.values())
codigos_idx = {v: i for i, v in enumerate(codigos)}

fold: toolkit.EventFold
folds: toolkit.EventFolds = experiment.folds()
for fold in folds:
    X_train, y_train = fold.extraia_amostras_treino()
    X_test = fold.extraia_amostras_teste()
    print(len(X_train), len(y_train), len(X_test))

    y_train_idx = list(map(codigos_idx.__getitem__, y_train))

    y_bins = np.bincount(y_train_idx) / len(y_train_idx)
    y_pred_mean = np.tile(y_bins, (len(X_test), 1))

    fold.calcule_metricas_parciais(y_pred_mean, codigos)

print(folds.obtenha_metricas_parciais())