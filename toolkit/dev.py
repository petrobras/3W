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
import seaborn as sns
import os

matplotlib.use("agg")
from matplotlib import pyplot as plt
from pathlib import Path, PurePosixPath
from sklearn import metrics
from alive_progress import alive_bar
from itertools import chain, compress, repeat
from functools import lru_cache
from zipfile import ZipFile
from typing import Union

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
)

# Transforma lista de instâncias (lista de tuplas (X, y)) em lista de
# exemplos (X, y).
# Adicionalmente, também filtra alvos inválidos
def extraia_exemplos(instancias, retornar_grupos=False):
    if len(instancias) == 0:
        return ([], [])

    # Busca alvos nulos
    y_finite = map(np.isfinite, chain(*(instancia[1] for instancia in instancias)))
    X_iter = chain(*(instancia[0] for instancia in instancias))
    y_iter = chain(*(instancia[1] for instancia in instancias))

    # Adiciona iterador de grupos
    if retornar_grupos:
        grupos = list(
            chain(*(repeat(grupo, len(X)) for grupo, (X, y) in enumerate(instancias)))
        )

        iter_zip = zip(X_iter, y_iter, grupos)
    else:
        iter_zip = zip(X_iter, y_iter)

    # Executa iteradores e retorna X, y, [grupos] como listas
    result = list(map(list, zip(*compress(iter_zip, y_finite))))

    # Converte y (segunda lista do resultado) para int
    result[1] = list(map(int, result[1]))

    return tuple(result)


# Classe cujo objeto contém todas as informações necessárias para uma
# rodada do K-fold do classificador de evento.
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

        # Nota: `instancias_treino` e `instancias_teste` são listas de
        # tuplas (X, y)

        # Aplica passo em instâncias de treino
        self.instancias_treino = [(X[::step], y[::step]) for X, y in instancias_treino]

        # Aplica passo em instâncias de teste
        self.instancias_teste = [
            (X[::passo_teste], y[::passo_teste]) for X, y in instancias_teste
        ]

        self.nome_instancias_treino = nome_instancias_treino
        self.nome_instancias_teste = nome_instancias_teste

        # Verfica se alguma das instâncias ficou vazia depois de
        # aplicado passo
        for instancia in self.instancias_treino + self.instancias_teste:
            X, y = instancia
            assert min(X.shape) > 0 and min(
                y.shape
            ), "Janela especificada gerou instância sem amostras"

    # Método para extração de amostras para treino
    @lru_cache(1)
    def extract_training_samples(self, retornar_grupos=False):
        return extraia_exemplos(self.instancias_treino, retornar_grupos)

    # Método para extração de amostras para teste
    @lru_cache(1)
    def extraia_amostras_teste_completo(self, retornar_grupos=False):
        return extraia_exemplos(self.instancias_teste, retornar_grupos)

    # Método para extração de amostras para teste
    def extract_test_samples(self):
        # Retorna apenas X
        return self.extraia_amostras_teste_completo()[0]

    # Método para cálculo de métricas parciais
    def calculate_partial_metrics(
        self, y_prev_soft, idx_to_codigo, apresente=False, apresente_conf={}
    ):
        """
        idx_to_codigo (list or dict):
            idx_to_codigo[i] = j indica que a i-ésima coluna de
            y_prev_soft corresponde ao código da classe j da tarefa
            corrente.
        """
        _, y_teste, grupos_teste = self.extraia_amostras_teste_completo(
            retornar_grupos=True
        )

        assert len(y_prev_soft) == len(
            y_teste
        ), f"Número incorreto de previsões: esperado {len(y_teste)}, encontrado {len(y_prev_soft)}"

        # códigos das classes para a tarefa que esse fold faz parte
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

        # Predições soft com shape correto
        shape_ok = (len(y_teste), n_codigos)
        assert (
            y_prev_soft.shape == shape_ok
        ), f"Predição deve ter shape (n_samples, n_classes) = ({shape_ok[0]},{shape_ok[1]})"

        # Todos os códigos devem aparecer em ordem_codigos_evento
        codigos_faltando = set(lista_codigos) - set(
            [idx_to_codigo[i] for i in range(n_codigos)]
        )
        assert (
            len(codigos_faltando) == 0
        ), f"Códigos faltando em 'idx_to_codigo': {codigos_faltando}"

        # Calculando predição da classe
        y_prev_idx = y_prev_soft.argmax(1)
        y_prev = list(map(idx_to_codigo.__getitem__, y_prev_idx))

        # Calculando probabilidade predita de regime + transiente para
        # plotar
        y_prob_nao_normal = y_prev_soft[:, coluna_regime].copy()
        if coluna_transiente is not None:
            y_prob_nao_normal += y_prev_soft[:, coluna_transiente]

        # Métrica principal
        f_beta = metrics.fbeta_score(
            y_teste, y_prev, beta=1.2, average="micro", labels=lista_codigos
        )
        f_beta *= 100.0

        # MEAN_LOG_LOSS
        log_loss_medio = metrics.log_loss(
            y_teste, y_prev_soft, labels=lista_codigos, normalize=True
        )

        metricas = {"F_BETA [%]": f_beta, "MEAN_LOG_LOSS": log_loss_medio}

        self.event_folds.salve_metricas_parciais(self, metricas)
        if apresente:

            # Carregando configuração padrão de apresentação
            def set_config(name, value, overwrite=False):
                if overwrite or (name not in apresente_conf):
                    apresente_conf[name] = value

            set_config("sharex", False)
            set_config("mostrar_nome_instancia", True)
            set_config("largura_fig", 1.8)
            set_config("hspace", 0.7)
            set_config("mostra_prob", True)
            # ===============================================

            # Valores do gráfico: normal=0, transiente=0.5, em regime=1
            plot_values = {
                event_labels["normal"]: 0,
                event_labels["regime"]: 1,
            }
            if codigo_transiente is not None:
                plot_values[event_labels["transiente"]] = 0.5

            y_prev_plot = list(map(plot_values.__getitem__, y_prev))
            y_teste_plot = list(map(plot_values.__getitem__, y_teste))

            # Cria um plot para cada grupo
            grupos_count = np.bincount(grupos_teste)
            n_grupos = len(grupos_count)

            fig, axes = plt.subplots(
                n_grupos,
                1,
                figsize=(11, apresente_conf["largura_fig"] * n_grupos),
                sharex=apresente_conf["sharex"],
            )
            plt.subplots_adjust(hspace=apresente_conf["hspace"])
            if n_grupos == 1:
                axes = [axes]

            axes[0].set_title(
                f"F_BETA [%]: {f_beta:.3f}   MEAN_LOG_LOSS: {log_loss_medio:.5f}"
            )

            grupo_idx_inicio = 0
            for grupo, ax in enumerate(axes):
                grupo_count = grupos_count[grupo]
                y_prev_grupo = y_prev_plot[
                    grupo_idx_inicio : grupo_idx_inicio + grupo_count
                ]
                y_teste_grupo = y_teste_plot[
                    grupo_idx_inicio : grupo_idx_inicio + grupo_count
                ]
                y_prob_nao_normal_grupo = y_prob_nao_normal[
                    grupo_idx_inicio : grupo_idx_inicio + grupo_count
                ]

                ax.plot(y_prev_grupo, marker=11, color="orange", linestyle="")
                ax.plot(y_teste_grupo, marker=10, color="green", linestyle="")
                ax.set_ylim([-0.2, 1.2])
                yticks, yticklabels = [0, 1], ["normal", "em regime"]
                if codigo_transiente is not None:
                    yticks.insert(1, 0.5)
                    yticklabels.insert(1, "transiente")
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                if apresente_conf["mostrar_nome_instancia"] and (
                    self.nome_instancias_teste is not None
                ):
                    title = ax.get_title()
                    if title != "":
                        title += "\n"
                    title += f"{self.nome_instancias_teste[grupo]}"
                    ax.set_title(title)

                # Segundo eixo com probabilidade de regiem+transiente
                if apresente_conf["mostra_prob"]:
                    ax2 = ax.twinx()
                    ax2.plot(
                        100.0 * (y_prob_nao_normal_grupo),
                        color="orange",
                        linestyle="-",
                        alpha=0.6,
                        label="prob. não normal",
                    )
                    ax2.set_ylim(0, 100)

                grupo_idx_inicio += grupo_count

            axes[0].legend(["predita", "verdadeira"])
            axes[-1].set_xlabel("amostra")
            plt.show()

        return metricas


# Classe que encapsula vários objetos da classe EventFold
class EventFolds:
    def __init__(
        self,
        experiment,
        nomes_instancias,
        folds_instancias,
    ):

        self.experiment: Experiment = experiment
        self.event_type = experiment.event_type
        self.use_instancias_extras = experiment.use_instancias_extras
        self.pad_mode = experiment.pad_mode
        self.pbar = experiment.pbar
        self.warnings = experiment.warnings
        self.forca_binario = experiment.forca_binario

        self.LABEL = experiment.LABEL
        self.OBSERVATION_LABELS = experiment.OBSERVATION_LABELS
        self.TRANSIENT = experiment.TRANSIENT
        self.window = experiment.window
        self.step = experiment.step

        assert not self.use_instancias_extras, "Funcionalidade não implementada"

        # Filtro de nomes de eventos
        self.filtre_nomes_instancias = lambda filtro: list(
            compress(nomes_instancias, map(filtro, folds_instancias))
        )

        # Guarda nome das instâncias extras
        self.nomes_instancias_extras = self.filtre_nomes_instancias(
            lambda fold: fold == EXTRA_INSTANCES_TRAINING
        )

        # Obtém código de todos os folds, ignorando o fold negativo (utilizado
        # sempre para treino)
        self.folds_nums = sorted(set(folds_instancias) - {EXTRA_INSTANCES_TRAINING})

        # Carrega instâncias do evento
        nomes_instancias_evento = self.filtre_nomes_instancias(
            lambda fold: fold != EXTRA_INSTANCES_TRAINING
        )
        self.instancias = {}
        with alive_bar(
            len(nomes_instancias_evento),
            disable=not (self.pbar),
            force_tty=True,
            title=f"Loading instances",
            bar="bubbles",
            spinner=None,
        ) as bar:
            for nome_instancia in nomes_instancias_evento:
                self.instancias[nome_instancia] = self.carregue_instancia(
                    nome_instancia
                )
                bar()

        # Cria folds, agrupado por fold_num
        self.folds = []
        for fold_num in self.folds_nums:
            # Treino
            nome_instancias_treino = self.filtre_nomes_instancias(
                lambda fold: fold not in {fold_num, EXTRA_INSTANCES_TRAINING}
            )
            instancias_treino = [
                self.instancias[nome_instancia]
                for nome_instancia in nome_instancias_treino
            ]

            # Teste
            nome_instancias_teste = self.filtre_nomes_instancias(
                lambda fold: fold == fold_num
            )
            instancias_teste = [
                self.instancias[nome_instancia]
                for nome_instancia in nome_instancias_teste
            ]

            # Cria Fold
            event_fold = EventFold(
                instancias_treino,
                instancias_teste,
                self.step,  # WIP
                self.step,  # WIP
                self,
                nome_instancias_treino,
                nome_instancias_teste,
            )
            self.folds.append(event_fold)

        self.folds_metricas = {fold: None for fold in self.folds}

    def extrai_arrays(self, instancia_abs, pad_mode="na"):
        """
        Extrai np.arrays X e y a partir do csv em instancia_abs.
        Na extração os valore de referência são calculados e incluídos
        como colunas em X. X tem ses dados completados segundo pad_mode
        para formar primeiras janelas.

        pad_mode:
            'na'    : completa X com NA alinhando com primeiro dado
            anotado em y
            'valid' : descarta os dados que não cabem na primeira janela
            de detecção
        """
        # Leitura do arquivo CSV que contém a instância
        with instancia_abs.open() as f:
            df = pd.read_csv(f, usecols=VARS + [CLASS])

        # Extração dos conjuntos de amostras para treino
        X_treino = df[VARS].values.astype(np.float32)

        # Verifica primeiro índice da variável target
        first_class = df[CLASS].first_valid_index()
        inicio_X = first_class - self.window + 1
        inicio_y = first_class

        # Verifica o tamanho da jenala solicitada e aplica pad se
        # necessário
        if inicio_X < 0:
            if self.warnings:
                warnings.warn(
                    f'Arquivo "{instancia_abs}" não possui amostras suficientes para janela de detecção solicitada ({self.window}s.\
                        Aplicando pad {pad_mode})',
                    RuntimeWarning,
                )
            if pad_mode == "na":
                # Completando os dados em X_treino para com NA
                X_treino = np.vstack(
                    [
                        np.full(
                            (-inicio_X, X_treino.shape[1]),
                            np.nan,
                            dtype=np.float32,
                        ),
                        X_treino,
                    ]
                )
                inicio_X = 0
            elif pad_mode == "valid":
                # Descartando (-inicio_X) instantes do df para ter 1a
                # janela válida
                inicio_y += -inicio_X
                inicio_X = 0

                # Validando se janela solicitada é maior do que dados
                # disponíveis
                if inicio_y >= df.shape[0]:
                    raise (
                        Exception(
                            f"Arquivo '{instancia_abs}' não possui amostras suficientes para pad: {pad_mode}."
                        )
                    )

                # Validando se mais de 50% dos dados normais foram
                # descartados (ou algum outro controle de qualidade?)
                # TODO

            else:
                raise (Exception(f"Opção de pad não reconhecida: {pad_mode}."))

        X_treino_pad = X_treino[inicio_X:]
        y_treino = df.iloc[inicio_y:][CLASS].values

        return X_treino_pad, y_treino

    def carregue_instancia(self, instancia):
        instancia_abs = Path(os.path.join(PATH_DATASET, instancia))
        X_treino_extra, y_treino = self.extrai_arrays(
            instancia_abs, pad_mode=self.pad_mode
        )

        # Aplicação de janela deslizante
        Xw_treino = rolling_window(X_treino_extra, self.window, axes=0, toend=False)

        # Check de sanidade
        assert len(y_treino) == len(
            Xw_treino
        ), f'[BUG] X e y de treino não estão sincronizados para o arquivo "{instancia_abs}"'

        assert (
            min(Xw_treino.shape) > 0
        ), f'Janela especificada gerou instância sem amostras para o arquivo "{instancia_abs}"'

        # Ao usar instâncias de outros eventos para o treinamento do
        # evento corrente (self.event_type)
        # códigos de outros eventos podem surgir em y_treino.
        # y_treino deve ter somente os códigos do evento corrente.
        # Os códigos novos (derivados de outros eventos) são convertidos
        # para código do evento Normal (0).
        y_finite_mask = np.isfinite(y_treino)
        outro_codigo_mask = y_finite_mask & np.isin(
            y_treino, list(self.OBSERVATION_LABELS), invert=True
        )
        if self.warnings and outro_codigo_mask.sum() > 0:
            novos_codigos = set(y_treino[outro_codigo_mask])
            warnings.warn(
                f'Códigos de outros eventos ("{novos_codigos}") sendo convertidos para 0.',
                RuntimeWarning,
            )
        y_treino[outro_codigo_mask] = 0

        # Tratamento para classificação binária : codigo_transitente ->
        # codigo_regime
        if self.TRANSIENT and self.forca_binario:
            codigo_regime = self.LABEL
            codigo_transiente = self.LABEL + TRANSIENT_OFFSET
            y_treino[y_treino == codigo_transiente] = codigo_regime

        return Xw_treino, y_treino

    def __iter__(self):
        for fold in self.folds:
            yield fold

    def __len__(self):
        return len(self.folds)

    # Método para retenção de métricas
    def salve_metricas_parciais(self, fold, metricas):
        assert fold in self.folds_metricas, "Fold não encontrado"
        if self.folds_metricas[fold] is not None:
            warnings.warn(
                "Fold com métricas já computadas. Recarregue os folds "
                + "para evitar esta mensagem.",
                RuntimeWarning,
            )
        self.folds_metricas[fold] = metricas

    @lru_cache(1)
    def extraia_amostras_simuladas_e_desenhadas(self):
        # Obtém instâncias extras (simuladas e desenhadas, representadas
        # pelo fold==EXTRA_INSTANCES_TRAINING)
        instancias_extras = []
        with alive_bar(len(self.nomes_instancias_extras)) as bar:
            for nome_instancia in self.pbar(self.nomes_instancias_extras):
                instancias_extras.append(self.carregue_instancia(nome_instancia))
                bar()

        instancias_extras_passo = [
            (X[:: self.step], y[:: self.step]) for X, y in instancias_extras
        ]  # Aplica passo de treino
        return extraia_exemplos(instancias_extras_passo)

    # Método para consulta/cálculo de métricas parciais e globais (média
    # e std)
    def get_metrics(self, boxplot=False):
        folds_metrics = {
            i: metrics for i, (_, metrics) in enumerate(self.folds_metricas.items())
        }
        df_metricas = pd.DataFrame.from_dict(folds_metrics, orient="index")
        if boxplot:
            for metrica in ["F_BETA [%]", "MEAN_LOG_LOSS"]:
                plt.figure(figsize=(11, 1))
                sns.boxplot(x=df_metricas[metrica], width=0.4, palette="colorblind")
                sns.stripplot(
                    x=df_metricas[metrica],
                    jitter=True,
                    marker="o",
                    alpha=0.5,
                    color="black",
                )
                plt.show()
        df_metricas.index.name = "FOLD"
        df_metricas.loc["MEAN"] = df_metricas.mean()
        df_metricas.loc["STANDARD DEVIATION"] = df_metricas.std()
        self.experiment.metrics = df_metricas.loc["MEAN"].to_dict()
        return df_metricas


class Experiment:
    """This class contains objects related to machine learning approach
    experiments"""

    def __init__(
        self,
        event_name,
        ova=True,
        use_instancias_extras=False,  # WIP
        pad_mode="valid",
        pbar=True,
        warnings=False,
        forca_binario=False,  # WIP
    ):
        """_summary_"""
        self.event_type = EventType(event_name)
        self.ova = ova
        self.use_instancias_extras = use_instancias_extras  # WIP
        self.pad_mode = pad_mode
        self.pbar = pbar
        self.warnings = warnings
        self.forca_binario = forca_binario  # WIP

        self.LABEL = self.event_type.LABEL
        self.OBSERVATION_LABELS = self.event_type.OBSERVATION_LABELS
        self.DESCRIPTION = self.event_type.DESCRIPTION
        self.TRANSIENT = self.event_type.TRANSIENT
        self.window = self.event_type.window
        self.step = self.event_type.step

    @property
    def event_labels(self):  # WIP
        """
        Dicionário com os códigos das classes que envolvem essa tarefa
        de classificação. As classes podem ser 'normal', 'regime' e
        'transiente'. A classe transiente não existe para tarefas de
        classificação binária.
        """
        codigos = {"normal": 0, "regime": self.LABEL}
        if self.TRANSIENT and (not self.forca_binario):
            codigos["transiente"] = self.LABEL + TRANSIENT_OFFSET
        return codigos

    def folds(self):
        folds = os.path.join(PATH_FOLDS, f"folds_clf_{self.LABEL:02d}.csv")
        with Path(folds).open() as f:
            df_event = pd.read_csv(f)

        if not self.ova:
            df_event = df_event.query("~is_ova")

        nomes_instancias = df_event["instancia"].tolist()
        folds_instancias = df_event["fold"].tolist()

        return EventFolds(
            self,
            nomes_instancias,
            folds_instancias,
        )
