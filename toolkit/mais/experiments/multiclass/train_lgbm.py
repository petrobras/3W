# -*- coding: utf-8 -*-
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import numpy as np

np.seterr(divide="ignore", invalid="ignore")

import click
import os
import importlib
import yaml

import lightgbm as lgb
from sklearn.model_selection import train_test_split

import mlflow
import optuna

from mlpbr.data.dataset import MAEDataset
from .tune_lgbm import prepare_data


@click.group()
@click.option("--train-path", type=click.Path(exists=True))
@click.option("--test-path", type=click.Path(exists=True))
@click.option("-e", "--experiment-name", type=click.STRING)
@click.option("-t", "--tgt-class", type=click.INT)
@click.option("--max-epochs", type=click.INT, default=300)
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj.update(kwargs)


@cli.command()
@click.option("-p", "--param-file", type=click.Path(exists=True))
@click.option("-j", "--num-jobs", type=click.INT, default=8)
@click.pass_context
def eval_lgbm(ctx, **kwargs):
    """evaluate model"""
    # grab all cli params
    config = {**ctx.obj, **kwargs}

    # preload events
    training_events = MAEDataset.load_events(config["train_path"])
    test_events = MAEDataset.load_events(config["test_path"])

    # fake trial sampler with fixed parameters
    with open(config["param_file"]) as f:
        trial = optuna.trial.FixedTrial(yaml.safe_load_all(f))

    with mlflow.start_run(run_name=f"Eval LGBM -- Target {config['tgt_class']}"):
        mlflow.log_params(config)
        mlflow.log_params(trial.params)

        # fake sample experiment
        experiment = importlib.import_module(config["experiment_name"]).sample(
            trial, config["tgt_class"]
        )
        metric_name = experiment.metric_name()

        # process events
        data = prepare_data(experiment, events, config["num_jobs"])

        Xt = data["train_X"]
        yt = data["train_y"]

        Xv = data["val_X"]
        yv = data["val_y"]

        base_params = {
            "seed": 0,
            "verbosity": -2,
            "task": "train",
            "boosting": "gbdt",
            "objective": "binary",
            "boost_from_average": True,
            "is_unbalance": True,
            "learning_rate": 0.1,
            "bagging_freq": 1,
            "metrics": ["custom", "auc", "average_precision"],
            "num_threads": config["num_jobs"],
            "first_metric_only": True,
        }
        opt_params = {
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.1, 1.0, step=0.05
            ),
            "pos_bagging_fraction": trial.suggest_float(
                "pos_bagging_fraction", 0.1, 1.0, step=0.05
            ),
            "neg_bagging_fraction": trial.suggest_float(
                "neg_bagging_fraction", 0.1, 1.0, step=0.05
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-5, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-5, 10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 128, step=1),
        }

        # preprocess
        experiment.fit(Xt)
        Xt, yt = experiment.transform(Xt, yt)
        Xv, yv = experiment.transform(Xv, yv)

        # construct lgbm data structures
        train_set = lgb.Dataset(Xt, yt)
        val_set = train_set.create_valid(Xv, yv)

        metrics = {}
        booster = lgb.train(
            num_boost_round=config["max_epochs"],
            params={
                **base_params,
                **opt_params,
            },
            train_set=train_set,
            valid_sets=[val_set, train_set],
            valid_names=["validation", "training"],
            feval=experiment.metric_lgbm(),
            callbacks=[report_callback],
            verbose_eval=False,
            evals_result=metrics,
            early_stopping_rounds=100,
        )


@cli.command()
@click.option("-p", "--param-file", type=click.Path(exists=True))
@click.pass_context
def cv(ctx, param_file):
    """Standalone cross-validation of single parameter set"""

    config = ctx.obj
    with open(param_file) as f:
        params = yaml.full_load(f)

    trial = optuna.trial.FixedTrial(params)
    base_objective(trial, config)


if __name__ == "__main__":
    os.nice(19)
    cli(obj={})
