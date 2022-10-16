#! python3
# -*- coding: utf-8 -*-
import os
import importlib
import warnings
import pickle
import tempfile
import json
import datetime
import logging

import numpy as np
import lightgbm as lgb
import optuna
import mlflow

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix

import click

import joblib

# import dask.distributed

import seaborn as sns
import matplotlib.pyplot as plt

from mais.data.dataset import MAEDataset


def warn(*args, **kwargs):
    pass


warnings.warn = warn

np.seterr(divide="ignore", invalid="ignore")

logger = logging.getLogger("tune_lgbm")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

formatter = logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s] %(message)s")
ch.setFormatter(formatter)

lgb.register_logger(logger)

####################################################
# some helper functions


def metric_suite(y_true, y_score):
    """compute many metrics using y_pred (predict_proba)"""
    y_pred = np.argmax(y_score, -1)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc_ovr_macro": roc_auc_score(
            y_true, y_score, average="macro", multi_class="ovr"
        ),
        "roc_auc_ovr_weighted": roc_auc_score(
            y_true, y_score, average="weighted", multi_class="ovr"
        ),
        "roc_auc_ovo_macro": roc_auc_score(
            y_true, y_score, average="macro", multi_class="ovo"
        ),
        "roc_auc_ovo_weighted": roc_auc_score(
            y_true, y_score, average="weighted", multi_class="ovo"
        ),
    }

    cm = {
        "raw": confusion_matrix(y_true, y_pred, normalize=None),
        "normalized": confusion_matrix(y_true, y_pred, normalize="true"),
    }

    return {"metrics": metrics, "confusion_matrix": cm}


def gather_results(results):
    """combine list of dicts of several runs into dict of lists"""
    return {
        "scores": [r["score"] for r in results],
        "metrics": {
            m: [r["metrics"][m] for r in results] for m in results[0]["metrics"].keys()
        },
        "confusion_matrices": {
            m: [r["confusion_matrix"][m] for r in results]
            for m in results[0]["confusion_matrix"].keys()
        },
    }


def log_result(result):
    """log result dictionary for a single run to mlflow"""

    # log target metric
    mlflow.log_metric("score", result["score"])

    # log aux metrics
    mlflow.log_metrics(result["metrics"])

    # plot and log confusion matrix
    fig, ax = plot_confusion_matrix(result["confusion_matrix"]["normalized"])
    ax.set_title("Confusion matrix")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # store model and experiment
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "model.pkl"), "wb") as f:
            pickle.dump(result["model"], f)
        mlflow.log_artifact(f.name)

        with open(os.path.join(tmp_dir, "experiment.pkl"), "wb") as f:
            pickle.dump(result["experiment"], f)
        mlflow.log_artifact(f.name)


def log_results(results):
    """log gathered results dictionary for a cross-validated run to mlflow"""
    # log target metric
    mlflow.log_metric("score-avg", np.mean(results["scores"]))
    mlflow.log_metric("score-std", np.std(results["scores"]))

    # log aux metrics
    for k, v in results["metrics"].items():
        mlflow.log_metric(f"{k}-avg", np.mean(v))
        mlflow.log_metric(f"{k}-std", np.std(v))

    # plot and log confusion matrices
    fig, ax = plot_confusion_matrix(
        np.sum(results["confusion_matrices"]["raw"], axis=0), normalize=True
    )
    ax.set_title("Overall confusion matrix")
    mlflow.log_figure(fig, "overall_confusion_matrix.png")
    plt.close(fig)

    fig, ax = plot_confusion_matrix(
        np.mean(results["confusion_matrices"]["normalized"], axis=0),
        np.std(results["confusion_matrices"]["normalized"], axis=0),
    )
    ax.set_title("Average confusion matrix")
    mlflow.log_figure(fig, "averaged_confusion_matrix.png")
    plt.close(fig)


def plot_confusion_matrix(cm, std=None, normalize=False):
    """generate confusion matrix visualization"""
    if normalize:
        cm = cm / cm.sum(1, keepdims=True)  # normalize over rows

    if std is not None:
        annot = [
            [rf"${v:.2f}\pm{s:.2f}$" for v, s in zip(vr, sr)] for vr, sr in zip(cm, std)
        ]
    else:
        annot = [[f"${v:.2f}$" for v in vr] for vr in cm]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, cmap="viridis", annot=annot, fmt="", square=True, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    return fig, ax


###########################
# main functions


def score_model(train_set, test_set, experiment, model, n_jobs=-1):
    """train and evaluate an experiment/model pair"""
    # split 20% for early stopping set
    Xt, Xs, yt, ys, gt, gs = train_test_split(
        train_set.X,
        train_set.y,
        train_set.g,
        test_size=0.2,
        random_state=0,
        stratify=train_set.y,
    )
    # rebalance train set
    Xt, yt, _ = experiment.balance(Xt, yt, gt, train_set.g_class)

    # preprocess and fit
    logger.info("experiment.fit")
    Xt, yt = experiment.fit_transform(Xt, yt)
    Xs, ys = experiment.transform(Xs, ys)

    logger.info(f"model.fit -- X.shape={Xt.shape}")
    fit_cb = [
        lgb.early_stopping(
            50,
            verbose=False,
        ),
        lgb.log_evaluation(10),
    ]
    model.fit(Xt, yt, eval_set=[(Xs, ys)], callbacks=fit_cb)

    XT, yT = experiment.transform(test_set.X, test_set.y)

    # compute target metric
    score = experiment.metric_scorer()(model, XT, yT)

    # collect auxiliary metrics
    logger.info("model.predict")
    y_score = model.predict_proba(XT)

    return {
        "model": model,
        "experiment": experiment,
        "score": score,
        **metric_suite(yT, y_score),
    }


def cross_val_score(events, experiment, model, num_splits, n_jobs=-1):
    """train and evaluate an experiment/model pair across several splits"""
    # preprocess all events
    transformed_events = MAEDataset.transform_events(
        events,
        experiment.raw_transform,
        instance_types=experiment.instance_types,
        tgt_events=experiment.tgt_events,
        n_jobs=n_jobs,
    )

    # gather event types for stratification
    event_types = [e[2] for e in transformed_events]

    results = []
    for train, test in StratifiedKFold(num_splits).split(
        transformed_events, event_types
    ):
        train_set = MAEDataset.gather([transformed_events[i] for i in train])
        test_set = MAEDataset.gather([transformed_events[i] for i in test])
        results.append(score_model(train_set, test_set, experiment, model, n_jobs))

    # gather result
    return gather_results(results)


def hyperparameter_search(events, experiment_sampler, model_sampler, config):

    # single trial is a k-fold
    def objective(trial):
        experiment = experiment_sampler(trial)
        model = model_sampler(trial)

        with mlflow.start_run(nested=True, run_name=f"trial - {trial.number} - cv"):
            mlflow.log_params(trial.params)
            results = cross_val_score(
                events, experiment, model, config["num_splits"], config["n_jobs"]
            )
            log_results(results)

        # optimize mean of target metric
        return np.mean(results["scores"])

    # create study with optional fixed hyperparameters
    study = optuna.create_study(
        sampler=optuna.samplers.PartialFixedSampler(
            config["fixed_params"],
            optuna.samplers.TPESampler(
                multivariate=True, warn_independent_sampling=False
            ),
        ),
        direction="maximize",
    )

    # load additional options
    for k, v in config["user_attrs"].items():
        study.set_user_attr(k, v)

    with mlflow.start_run(nested=True, run_name="tuning"):

        def mlflow_callback(study, trial):
            mlflow.log_metric("score-avg", trial.value, trial.number)

        study.optimize(objective, config["num_trials"], callbacks=[mlflow_callback])
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best", study.best_value)

        # log study as artifact for later analysis
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "study.pkl"), "wb") as f:
                pickle.dump(study, f)
            mlflow.log_artifact(f.name)
    return study


def nested_cv_score(events, experiment_sampler, model_sampler, config):
    # adjust num splits for config
    config["num_splits"] = config["inner_splits"]

    # extract the source of each event
    event_type = [e["event_type"] for e in events]
    results = []
    cv = StratifiedKFold(config["outer_splits"])  # split events, stratified
    for train, test in cv.split(events, event_type):
        train_events = [events[i] for i in train]
        test_events = [events[i] for i in test]

        # find best parameters for current split
        study = hyperparameter_search(
            train_events, experiment_sampler, model_sampler, config
        )
        best_trial = study.best_trial

        best_experiment = experiment_sampler(best_trial)
        best_model = model_sampler(best_trial)

        # evaluate best params

        # map and gather tranining set
        transformed_train_events = MAEDataset.transform_events(
            train_events,
            best_experiment.raw_transform,
            instance_types=best_experiment.instance_types,
            tgt_events=best_experiment.tgt_events,
            n_jobs=config["n_jobs"],
        )
        train_set = MAEDataset.gather(transformed_train_events)

        # map and gather test set
        transformed_test_events = MAEDataset.transform_events(
            test_events,
            best_experiment.raw_transform,
            instance_types=best_experiment.instance_types,
            tgt_events=best_experiment.tgt_events,
            n_jobs=config["n_jobs"],
        )
        test_set = MAEDataset.gather(transformed_test_events)

        result = score_model(
            train_set, test_set, best_experiment, best_model, config["n_jobs"]
        )
        results.append(result)

    # aggregate results
    results = gather_results(results)
    return results


def lightgbm_sampler(trial):
    return lgb.LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.1,
        is_unbalance=True,
        subsample_freq=1,
        verbose=-1,
        metrics=["multi_error"],
        subsample=trial.suggest_float("subsample", 0.1, 1.0, step=0.05),
        colsample_bytree=trial.suggest_float("feature_fraction", 0.1, 1.0, step=0.05),
        reg_alpha=trial.suggest_float("lambda_l1", 1e-5, 10, log=True),
        reg_lambda=trial.suggest_float("lambda_l2", 1e-5, 10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 128, step=1),
    )


def parse_json(ctx, self, value):
    return json.loads(value)


@click.group()
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj.update(kwargs)


@cli.command()
@click.option("-r", "--data-root", type=click.Path(exists=True))
@click.option("-e", "--experiment-name", type=click.STRING)
@click.option("-n", "--num-trials", type=click.INT, default=100)
@click.option("-i", "--inner-splits", type=click.INT, default=5)
@click.option("-o", "--outer-splits", type=click.INT, default=5)
@click.option("-j", "--n-jobs", type=click.INT, default=-1)
@click.option("--fixed-params", type=click.STRING, default="{}", callback=parse_json)
@click.option("--user-attrs", type=click.STRING, default="{}", callback=parse_json)
@click.pass_context
def nested_cv(ctx, **kwargs):
    # gather configuration
    config = {**ctx.obj, **kwargs}

    with joblib.parallel_backend("loky", n_jobs=config["n_jobs"]):

        # preload events
        events = MAEDataset.load_events(config["data_root"], -1)

        model_sampler = lightgbm_sampler
        experiment_sampler = importlib.import_module(config["experiment_name"]).sample
        mlflow.set_experiment(datetime.datetime.now().strftime("%Y%m%d_%H%M_nested_cv"))

        with mlflow.start_run(run_name="nested_cv"):
            mlflow.log_params(config)
            results = nested_cv_score(events, experiment_sampler, model_sampler, config)
            log_results(results)


@cli.command()
@click.option("-t", "--train-root", type=click.Path(exists=True))
@click.option("-T", "--test-root", type=click.Path(exists=True))
@click.option("-e", "--experiment-name", type=click.STRING)
@click.option("-n", "--num-trials", type=click.INT, default=100)
@click.option("-s", "--num-splits", type=click.INT, default=5)
@click.option("-j", "--n-jobs", type=click.INT, default=-1)
@click.option("--fixed-params", type=click.STRING, default="{}", callback=parse_json)
@click.option("--user-attrs", type=click.STRING, default="{}", callback=parse_json)
@click.pass_context
def tune(ctx, **kwargs):
    # grab all cli params
    config = {**ctx.obj, **kwargs}

    # _ = dask.distributed.Client(n_workers=config["n_jobs"], processes=True)
    with joblib.parallel_backend("loky", n_jobs=config["n_jobs"]):
        # preload events
        train_events = MAEDataset.load_events(config["train_root"], config["n_jobs"])

        # select samplers
        model_sampler = lightgbm_sampler
        experiment_sampler = importlib.import_module(config["experiment_name"]).sample

        # create experiment
        mlflow.set_experiment(datetime.datetime.now().strftime("%Y%m%d_%H%M_tuning"))

        with mlflow.start_run(run_name="testing"):
            mlflow.log_params(config)

            # find best hyper-params
            study = hyperparameter_search(
                train_events, experiment_sampler, model_sampler, config
            )

            # train model with best params
            best_trial = study.best_trial
            best_experiment = experiment_sampler(best_trial)
            best_model = model_sampler(best_trial)

            mlflow.log_params(best_trial.params)

            # map and gather tranining set
            transformed_train_events = MAEDataset.transform_events(
                train_events,
                best_experiment.raw_transform,
                instance_types=best_experiment.instance_types,
                tgt_events=best_experiment.tgt_events,
                n_jobs=config["n_jobs"],
            )
            train_set = MAEDataset.gather(transformed_train_events)

            # map and gather test set
            test_events = MAEDataset.load_events(config["test_root"], config["n_jobs"])
            transformed_test_events = MAEDataset.transform_events(
                test_events,
                best_experiment.raw_transform,
                instance_types=best_experiment.instance_types,
                tgt_events=best_experiment.tgt_events,
                n_jobs=config["n_jobs"],
            )
            test_set = MAEDataset.gather(transformed_test_events)
            result = score_model(
                train_set, test_set, best_experiment, best_model, config["n_jobs"]
            )
            log_result(result)


if __name__ == "__main__":
    os.nice(19)
    cli(obj={})
