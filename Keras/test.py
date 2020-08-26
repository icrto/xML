from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from ExplainerClassifierCNN import ExplainerClassifierCNN
import utils
import argparse
from dataset import load_data, DataGenerator
import numpy as np
from losses import unsupervised_explanation_loss, hybrid_explanation_loss
from tensorflow.keras.optimizers import SGD

np.random.seed(0)

parser = argparse.ArgumentParser(description="Configurable parameters.")

# Processing parameters
parser.add_argument(
    "--gpu", type=str, default="1", help="Which gpus to use in CUDA_VISIBLE_DEVICES."
)

# Model
parser.add_argument("model_ckpt", type=str, help="Model to load.")

# Directories and paths
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/media/TOSHIBA6T/ICRTO",
    help="Folder where dataset is located.",
)

# Data parameters
parser.add_argument(
    "--dataset",
    type=str,
    default="imagenetHVZ",
    choices=["synthetic", "NIH-NCI", "imagenetHVZ"],
    help="Dataset to load.",
)
parser.add_argument(
    "--nr_classes", type=int, default=2, help="Number of target classes."
)
parser.add_argument(
    "--img_size", nargs="+", type=int, default=[224, 224], help="Input image size."
)

# Testing parameters
parser.add_argument(
    "-bs", "--batch_size", type=int, default=32, help="Training batch size."
)
parser.add_argument(
    "-clf",
    "--classifier",
    type=str,
    default="resnet50",
    choices=["vgg", "resnet50"],
    help="Classifier.",
)
parser.add_argument(
    "--init_bias",
    type=float,
    default=3.0,
    help="Initial bias for the batch norm layer of the Explainer. For more details see the paper.",
)

# Loss parameters
parser.add_argument(
    "--loss",
    type=str,
    default="unsupervised",
    choices=["hybrid", "unsupervised"],
    help="Specifiy which loss to use. Either hybrid or unsupervised.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.9,
    help="Alfa of the last training phase. Loss = alpha * Lclassif + (1-alpha) * Lexplic",
)
parser.add_argument(
    "--beta", type=float, help="Lexplic_unsup = beta * L1 + (1-beta) * Total Variation"
)
parser.add_argument(
    "--gamma",
    type=float,
    help="Lexplic_hybrid = beta * L1 + (1-beta) * Total Variation + gamma* Weakly Loss",
)


# Other (misc)
parser.add_argument(
    "--cmap",
    type=str,
    default="viridis",
    help="Colourmap to use when saving the produced explanations.",
)

args = parser.parse_args()

# select defined gpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# get folder name & timestamp of stored model
path = os.path.dirname(args.model_ckpt)
timestamp = path.split("/")[-1]

# verify loss parameters
if args.beta is None:
    print("Please define a value for beta.")
    sys.exit(-1)

masks = False
if args.loss == "hybrid":
    masks = True  # ensure that the data generator returns object detection masks
    if args.gamma is None:
        print("Please define a value for gamma.")
        sys.exit(-1)
    loss_fn = hybrid_explanation_loss(beta=args.beta, gamma=args.gamma)
elif args.loss == "unsupervised":
    loss_fn = unsupervised_explanation_loss(beta=args.beta)

img_size = tuple(args.img_size)

# instantiate model class and load model to test
model = ExplainerClassifierCNN(
    num_classes=args.nr_classes,
    img_size=img_size,
    clf=args.classifier,
    init_bias=args.init_bias,
)
model.e2e_model.load_weights(args.model_ckpt)
model.e2e_model.compile(
    optimizer=SGD(),
    loss_weights={"classifier": args.alpha, "explainer": 1.0 - args.alpha,},
    loss={"explainer": loss_fn, "classifier": "categorical_crossentropy"},
    weighted_metrics={"classifier": ["accuracy"]},
)


# load test data and create test data generators
_, _, test_df, _, classes = load_data(
    folder=args.dataset_path, dataset=args.dataset, masks=masks, class_weights=None,
)

test_datagen = DataGenerator(
    test_df,
    batch_size=args.batch_size,
    img_size=img_size,
    num_classes=args.nr_classes,
    masks=masks,
    aug_prob=0,
    shuffle=True,
)

# test & save the results
return_dict = model.e2e_model.evaluate(test_datagen, return_dict=True)
f = open(os.path.join(path, timestamp + "_test_stats_best_loss.txt"), "w")
f.write(str(return_dict))
f.close()

all_labels = []
all_probs = []
for batch_imgs, input_dict in test_datagen:
    all_labels.extend(test_datagen.batch_labels)
    _, batch_probs = model.e2e_model.predict((batch_imgs, input_dict), verbose=0)
    all_probs.extend(batch_probs)


all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# plot roc & precision_recall curves
if args.nr_classes > 2:  # multiclass
    utils.plot_roc_curve_multiclass(
        os.path.join(path, timestamp), all_probs, all_labels, classes
    )
    utils.plot_precision_recall_curve_multiclass(
        os.path.join(path, timestamp), all_probs, all_labels, classes
    )
else:  # binary classification
    utils.plot_roc_curve(os.path.join(path, timestamp), all_probs, all_labels)
    utils.plot_precision_recall_curve(
        os.path.join(path, timestamp), all_probs, all_labels
    )

# save generated explanations
model.save_explanations(
    test_datagen, 2, path, test=True, classes=classes, cmap=args.cmap
)

