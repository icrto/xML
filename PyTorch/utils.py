import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# plot configs
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def norm(img):
    """ Scales image into [0, 1]

    Arguments:
        img {tensor} -- input image

    Returns:
        [tensor] -- scaled image
    """
    return img / 255.0


def freeze(model):
    """ Freezes model layers

    Arguments:
        model {PyTorch model} -- model
    """
    for layer in model.parameters():
        layer.requires_grad = False


def unfreeze(model):
    """ Unfreezes model layers

    Arguments:
        model {PyTorch model} -- model
    """
    for layer in model.parameters():
        layer.requires_grad = True


def create_folder(folder):
    """ Creates folder (with timestamped filename) to store results and models

    Arguments:
        folder {str} -- folder

    Returns:
        [timestamp, results_path] -- timestamp and folder name
    """
    if not os.path.exists(folder):
        os.mkdir(folder)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(folder, timestamp)

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    return timestamp, results_path


def plot_metric_train_val(nr_epochs, history_csv, metric, path, filename, plot_title):
    """ Plots (saves) training and validation evolution of specified metric

    Arguments:
        nr_epochs {int} -- total number of epochs
        history_csv {str} -- csv which contains logged training history
        metric {str} -- quantity to plot
        path {str} -- destination folder
        filename {str} -- destination filename
        plot_title {str} -- title of the plot
    """

    hist = pd.read_csv(history_csv)
    x_values = np.linspace(1, nr_epochs, nr_epochs)
    plt.plot(x_values, hist[str("train_" + metric)], "--")
    plt.plot(x_values, hist[str("val_" + metric)], ":")
    plt.title(plot_title)
    if "loss" in metric:
        plt.ylabel("Loss")
    elif "acc" in metric:
        plt.ylabel("Accuracy")
        plt.ylim([0.0, 1.1])
    plt.xlabel("Epoch")
    plt.xlim([0, nr_epochs])
    plt.legend(["Train", "Validation"], loc="best")
    plt.savefig(os.path.join(path, filename))
    plt.close()


def plot_roc_curve(filename, scores, labels):
    """ Plots (saves) roc curve for a binary classification scenario

    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
    """
    lw = 1

    fpr, tpr, _ = roc_curve(labels, scores[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(1, figsize=(10, 10))
    plt.plot(
        fpr,
        tpr,
        label="ROC curve (area = {0:0.2f})" "".format(roc_auc),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    lgd = plt.legend(loc="best")
    plt.savefig(filename + "_roc.png", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(filename, scores, labels):
    """ Plots (saves) precision vs recall curve for a binary classification scenario

    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
    """

    precision, recall, _ = precision_recall_curve(labels, scores[:, 1])
    auc_prec_recall = auc(recall, precision)
    average_precision = average_precision_score(labels, scores[:, 1])

    plt.figure(1, figsize=(10, 10))
    plt.plot(
        recall,
        precision,
        label="Precision Recall Curve (AP = {0:0.2f}; area = {0:0.2f})"
        "".format(average_precision, auc_prec_recall),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    lgd = plt.legend(loc="best")
    plt.savefig(
        filename + "_prec_recall.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
    )
    plt.close()


def plot_roc_curve_multiclass(filename, scores, labels, classes):
    """ Plots (saves) roc curve for a multiclass classification scenario

    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
        classes {list} -- list of class names
    """

    lw = 1
    nr_classes = len(classes)
    labels = label_binarize(labels, classes=list(range(nr_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nr_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nr_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1, figsize=(10, 10))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    colors = cycle(
        [
            "coral",
            "mediumorchid",
            "aqua",
            "darkolivegreen",
            "cornflowerblue",
            "gold",
            "pink",
            "chocolate",
            "brown",
            "darkslategrey",
            "tab:cyan",
            "slateblue",
            "yellow",
            "palegreen",
            "tan",
            "silver",
        ]
    )
    for i, color in zip(range(nr_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="class {0} (area = {1:0.2f})" "".format(classes[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    lgd = plt.legend(loc="best")
    plt.savefig(
        filename + "_roc_all.png", bbox_extra_artists=(lgd,), bbox_inches="tight"
    )
    plt.close()

    plt.figure(2, figsize=(10, 10))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    lgd = plt.legend(loc="best")
    plt.savefig(filename + "_roc.png", bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve_multiclass(filename, scores, labels, classes):
    """ Plots (saves) precision vs recall curve for a multiclass classification scenario

    Arguments:
        filename {str} -- destination filename
        scores {list} -- list of predicted probabilities for all images
        labels {list} -- list of target labels for all images
        classes {list} -- list of class names
    """
    lw = 1
    nr_classes = len(classes)
    labels = label_binarize(labels, classes=list(range(nr_classes)))

    precision = dict()
    recall = dict()
    auc_prec_recall = dict()
    average_precision = dict()
    for i in range(nr_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], scores[:, i])
        auc_prec_recall[i] = auc(recall[i], precision[i])
        average_precision[i] = average_precision_score(labels[:, i], scores[:, i])

    # Compute micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        labels.ravel(), scores.ravel()
    )
    auc_prec_recall["micro"] = auc(recall["micro"], precision["micro"])
    average_precision["micro"] = average_precision_score(
        labels, scores, average="micro"
    )

    # Compute macro-average
    # First aggregate all recall
    all_recall = np.unique(np.concatenate([recall[i] for i in range(nr_classes)]))

    # Then interpolate all ROC curves at these points
    mean_precision = np.zeros_like(all_recall)
    for i in range(nr_classes):
        mean_precision += np.interp(all_recall, recall[i], precision[i])

    # Finally average it and compute AUC
    mean_precision /= nr_classes

    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    auc_prec_recall["macro"] = auc(recall["macro"], precision["macro"])
    average_precision["macro"] = average_precision_score(
        labels, scores, average="macro"
    )

    # Plot all ROC curves
    plt.figure(1, figsize=(10, 10))
    plt.plot(
        recall["micro"],
        precision["micro"],
        label="micro-average (AP = {0:0.2f}; area = {0:0.2f})"
        "".format(average_precision["micro"], auc_prec_recall["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        recall["macro"],
        precision["macro"],
        label="macro-average (AP = {0:0.2f}; area = {0:0.2f})"
        "".format(average_precision["macro"], auc_prec_recall["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    colors = cycle(
        [
            "coral",
            "mediumorchid",
            "aqua",
            "darkolivegreen",
            "cornflowerblue",
            "gold",
            "pink",
            "chocolate",
            "brown",
            "darkslategrey",
            "tab:cyan",
            "slateblue",
            "yellow",
            "palegreen",
            "tan",
            "silver",
        ]
    )
    for i, color in zip(range(nr_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=lw,
            label="class {0} (AP = {1:0.2f}; area = {1:0.2f})"
            "".format(classes[i], average_precision[i], auc_prec_recall[i]),
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    lgd = plt.legend(loc="best")
    plt.savefig(
        filename + "_prec_recall_all.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(2, figsize=(10, 10))
    plt.plot(
        recall["micro"],
        precision["micro"],
        label="micro-average (AP = {0:0.2f}; area = {0:0.2f})"
        "".format(average_precision["micro"], auc_prec_recall["micro"]),
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        recall["macro"],
        precision["macro"],
        label="macro-average (AP = {0:0.2f}; area = {0:0.2f})"
        "".format(average_precision["macro"], auc_prec_recall["macro"]),
        color="red",
        linestyle=":",
        linewidth=2,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    lgd = plt.legend(loc="best")
    plt.savefig(filename + "_prec_recall.png", bbox_inches="tight")
    plt.close()

