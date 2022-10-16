# -*- coding: utf-8 -*-
""" Generate report from evaluation results """

import os
import pickle

import click
from dotenv import find_dotenv, load_dotenv

import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save
from sklearn import metrics


def plot_confusion_matrix(cm):
    """Plots normalized confusion matrix. The x-axis refers to the True class.

    * Parameters:

            - **cm**: np.ndarray - confusion matrix

    """

    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    cm_fix = cm_norm.copy()
    cm_fix[np.isnan(cm_fix)] = 0

    plt.matshow(cm_fix.T)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            cn = "--" if np.isnan(cm_norm[i, j]) else f"{100*cm_norm[i,j]:.1f}"
            s = f"{cm[i,j]} ({cn}\%)"
            plt.text(
                i,
                j,
                s,
                color="black" if cm_norm[i, j] > 0.5 else "white",
                horizontalalignment="center",
                verticalalignment="center",
                multialignment="center",
            )

    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    plt.xlabel("True class")
    plt.ylabel("Predicted class")


@click.command()
@click.option("--report-path", type=click.Path(exists=True))
@click.option("--output-location", type=click.STRING, default="tex")
def main(report_path, output_location):
    """Creates a report containg the test results from a model, inclusing the ROC curve, AUC result the confusion matrix for each class.

    * Parameters:

            - **report_path**: STRING - Reports location

            - **output_location**: STRING - Folder to save the report

    """
    out_path = f"{report_path}/{output_location}/"
    os.makedirs(out_path, exist_ok=True)

    with open(f"{report_path}/test_score.pkl", "rb") as f:
        test_score = pickle.load(f)

    # plot global metrics
    fpr, tpr, t = metrics.roc_curve(
        test_score["label"], test_score["predicted"], drop_intermediate=True
    )
    auc = metrics.roc_auc_score(test_score["label"], test_score["predicted"])

    plt.plot(fpr[1:], tpr[1:], label=f"AUC = {auc:.5f}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid()
    plt.legend()
    tikz_save(
        out_path + "roc.tikz.tex",
        externalize_tables=True,
        tex_relative_path_to_data="figs",
    )
    plt.close()

    # find false alarm shift
    false_alarm_threshold_idx = np.argwhere(fpr == 0)[-1][0]
    false_alarm_threshold = t[false_alarm_threshold_idx]

    precision, recall, t = metrics.precision_recall_curve(
        test_score["label"], test_score["predicted"]
    )
    ap = metrics.average_precision_score(test_score["label"], test_score["predicted"])

    plt.plot(recall, precision, label=f"AP = {ap:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.grid()
    plt.legend()
    tikz_save(out_path + "prc.tikz.tex", tex_relative_path_to_data="figs")
    plt.close()

    cm = metrics.confusion_matrix(
        test_score["label"], test_score["predicted"] > 0, labels=[0, 1]
    )
    plot_confusion_matrix(cm)
    plt.colorbar()
    tikz_save(
        out_path + "global_confusion_matrix.tikz.tex",
        externalize_tables=True,
        tex_relative_path_to_data="figs",
    )
    plt.close()

    cm = metrics.confusion_matrix(
        test_score["label"],
        test_score["predicted"] > false_alarm_threshold,
        labels=[0, 1],
    )

    plot_confusion_matrix(cm)
    plt.colorbar()
    tikz_save(
        out_path + "global_confusion_matrix_shifted.tikz.tex",
        tex_relative_path_to_data="figs",
    )
    plt.close()

    for c in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
        class_score = test_score[test_score["class_group"] == c]

        # import ipdb; ipdb.set_trace()
        cm = metrics.confusion_matrix(
            class_score["label"], class_score["predicted"] > 0, labels=[0, 1]
        )
        plot_confusion_matrix(cm)
        plt.colorbar()
        tikz_save(
            out_path + f"class{c}_confusion_matrix.tikz.tex",
            tex_relative_path_to_data="figs",
        )
        plt.close()

        cm = metrics.confusion_matrix(
            class_score["label"],
            class_score["predicted"] > false_alarm_threshold,
            labels=[0, 1],
        )
        plot_confusion_matrix(cm)
        plt.colorbar()
        tikz_save(
            out_path + f"class{c}_confusion_matrix_shifted.tikz.tex",
            tex_relative_path_to_data="figs",
        )
        plt.close()


if __name__ == "__main__":
    os.nice(19)
    load_dotenv(find_dotenv())
    main()
