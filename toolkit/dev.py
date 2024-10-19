"""This 3W toolkits' sub-module has resources related to development of
Machine Learning models.

The main tasks made possible by these features are:

- Choice of specific problem to work with;
- Data preparation;
- Experimentation of different approaches;
- Calculation and analysis of metrics.
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn import metrics
from alive_progress import alive_bar
from itertools import compress, repeat
from functools import lru_cache

from .rolling_window import rolling_window
from .base import (
    CLASS,
    EXTRA_INSTANCES_TRAINING,
    EventType,
    NORMAL_LABEL,
    PATH_DATASET,
    PATH_FOLDS,
    TRANSIENT_OFFSET,
    VARS,
    load_3w_dataset,  # For compatibility with 3W v2.0
)

# Class whose object contains all the necessary information for a
# round of K-fold of the event classifier.
class EventFold:
    def __init__(
        self,
        instancias_treino,
        instancias_teste,
        step,  # WIP
        passo_teste,  # WIP
        event_folds,
        nome_instancias_treino=None,
        nome_instancias_teste=None,
    ):
        self.event_folds: EventFolds = event_folds

        # Note: `instancias_treino` and `instancias_teste` are lists of
        # tuples (X, y)

        # Apply step to training instances
        self.instancias_treino = [(X[::step], y[::step]) for X, y in instancias_treino]

        # Apply step to test instances
        self.instancias_teste = [
            (X[::passo_teste], y[::passo_teste]) for X, y in instancias_teste
        ]

        self.nome_instancias_treino = nome_instancias_treino
        self.nome_instancias_teste = nome_instancias_teste

        # Check if any of the instances were empty after
        # applying step
        for instancia in self.instancias_treino + self.instancias_teste:
            X, y = instancia
            assert min(X.shape) > 0 and min(
                y.shape
            ), "Specified window generated instance without samples"

    # Method for extracting training samples
    @lru_cache(1)
    def extract_training_samples(self):
        # Extract training samples from self.instancias_treino
        X_train = np.concatenate([x[0] for x in self.instancias_treino])
        y_train = np.concatenate([x[1] for x in self.instancias_treino])
        return X_train, y_train

    # Method for extracting complete test samples
    @lru_cache(1)
    def extraia_amostras_teste_completo(self):
        # Extract test samples from self.instancias_teste
        X_test = np.concatenate([x[0] for x in self.instancias_teste])
        y_test = np.concatenate([x[1] for x in self.instancias_teste])
        return X_test, y_test

    # Method for extracting test samples
    def extract_test_samples(self):
        # Return only X
        return self.extraia_amostras_teste_completo()[0]

    # Method for calculating partial metrics
    def calculate_partial_metrics(
        self, y_pred_soft, idx_to_codigo, apresente=False, apresente_conf={}
    ):
        """
        Calculate partial metrics for the fold.

        Parameters
        ----------
        y_pred_soft : np.ndarray
            Soft predictions for the test set.
        idx_to_codigo : list or dict
            Mapping from prediction index to class code.
        apresente : bool, optional
            Whether to display the results, by default False
        apresente_conf : dict, optional
            Configuration for displaying the results, by default {}
        """

        X_test, y_test = self.extraia_amostras_teste_completo()

        assert len(y_pred_soft) == len(
            y_test
        ), f"Incorrect number of predictions: expected {len(y_test)}, found {len(y_pred_soft)}"

        # Class codes for the task that this fold is part of
        event_labels = self.event_folds.experiment.event_labels
        n_codigos = len(event_labels)
        lista_codigos = list(event_labels.values())
        codigo_regime = event_labels["regime"]
        codigo_transiente = (
            event_labels["transiente"] if "transiente" in event_labels else None
        )
        coluna_regime = next(
            i for i, j in enumerate(idx_to_codigo) if j == codigo_regime
        )
        coluna_transiente = None
        if codigo_transiente is not None:
            coluna_transiente = next(
                i for i, j in enumerate(idx_to_codigo) if j == codigo_transiente
            )

        # Soft predictions with correct shape
        shape_ok = (len(y_test), n_codigos)
        assert (
            y_pred_soft.shape == shape_ok
        ), f"Prediction must have shape (n_samples, n_classes) = ({shape_ok[0]},{shape_ok[1]})"

        # All codes must appear in ordem_codigos_evento
        codigos_faltando = set(lista_codigos) - set(
            [idx_to_codigo[i] for i in range(n_codigos)]
        )
        assert (
            len(codigos_faltando) == 0
        ), f"Missing codes in 'idx_to_codigo': {codigos_faltando}"

        # Calculating class prediction
        y_pred_idx = y_pred_soft.argmax(1)
        y_pred = list(map(idx_to_codigo.__getitem__, y_pred_idx))

        # Calculating predicted probability of regime + transient to
        # plot
        y_prob_nao_normal = y_pred_soft[:, coluna_regime].copy()
        if coluna_transiente is not None:
            y_prob_nao_normal += y_pred_soft[:, coluna_transiente]

        # Main metric
        f_beta = metrics.fbeta_score(
            y_test, y_pred, beta=1.2, average="micro", labels=lista_codigos
        )
        f_beta *= 100.0

        # MEAN_LOG_LOSS
        log_loss_medio = metrics.log_loss(
            y_test, y_pred_soft, labels=lista_codigos, normalize=True
        )

        metricas = {"F_BETA [%]": f_beta, "MEAN_LOG_LOSS": log_loss_medio}

        self.event_folds.salve_metricas_parciais(self, metricas)
        if apresente:
            # Loading default presentation configuration
            def set_config(name, value, overwrite=False):
                if overwrite or (name not in apresente_conf):
                    apresente_conf[name] = value

            set_config("sharex", False)
            set_config("mostrar_nome_instancia", True)
            set_config("largura_fig", 1.8)
            set_config("hspace", 0.7)
            set_config("mostra_prob", True)
            # ===============================================

            # Chart values: normal=0, transient=0.5, in regime=1
            plot_values = {
                event_labels["normal"]: 0,
                event_labels["regime"]: 1,
            }
            if codigo_transiente is not None:
                plot_values[event_labels["transiente"]] = 0.5

            y_pred_plot = list(map(plot_values.__getitem__, y_pred))
            y_teste_plot = list(map(plot_values.__getitem__, y_test))

            # Create a plot for each group
            grupos_count = np.bincount(grupos_teste)
            n_grupos = len(grupos_count)

            fig, axes = plt.subplots(
                n_grupos,
                1,
                figsize=(11, apresente_conf["largura_fig"] * n_grupos),
                sharex=apresente_conf["sharex"],
            )
            plt.subplots_adjust(hspace=apresente_conf["hspace"])
