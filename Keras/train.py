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
from tensorflow.keras.optimizers import Adadelta, SGD

np.random.seed(0)

parser = argparse.ArgumentParser(description="Configurable parameters.")

# Processing parameters
parser.add_argument(
    "--gpu", type=str, default="1", help="Which gpus to use in CUDA_VISIBLE_DEVICES."
)
parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers for dataloader."
)

# Directories and paths
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/media/TOSHIBA6T/ICRTO",
    help="Folder where dataset is located.",
)
parser.add_argument(
    "--folder",
    type=str,
    default="/media/TOSHIBA6T/ICRTO/results",
    help="Directory where images and models are to be stored.",
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
parser.add_argument(
    "--aug_prob",
    type=float,
    default=0,
    help="Probability of applying data augmentation to each image.",
)

# Training parameters
parser.add_argument(
    "--nr_epochs",
    type=str,
    default="10,10,50",
    help="Number of epochs for each of the 3 training phases as an array, for example: 50,100,50.",
)
parser.add_argument(
    "-bs", "--batch_size", type=int, default=32, help="Training batch size."
)
parser.add_argument(
    "--pretrained",
    default=False,
    action="store_true",
    help="True if one wants to load pre trained models in training phase 1, False otherwise.",
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
    default=2.0,
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
    type=str,
    default="1.0,0.0,0.9",
    help="Loss = alpha * Lclassif + (1-alpha) * Lexplic for each phase, for example: 1.0,0.0,0.9.",
)
parser.add_argument(
    "--beta", type=float, help="Lexplic_unsup = beta * L1 + (1-beta) * Total Variation"
)
parser.add_argument(
    "--gamma",
    type=float,
    help="Lexplic_hybrid = beta * L1 + (1-beta) * Total Variation + gamma* Weakly Loss",
)
parser.add_argument(
    "--class_weights",
    action="store_true",
    default=False,
    help="Use class weighting in loss function.",
)

# Learning parameters
parser.add_argument(
    "--opt",
    type=str,
    default="sgd",
    choices=["sgd", "adadelta"],
    help="Optimiser to use. Either adadelta or sgd.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=str,
    default="1e-3,0,1e-4",
    help="Learning rate for each training phase, for example: 1e-3,0,1e-4.",
)
parser.add_argument(
    "--decay", type=float, default=0.0001, help="Learning rate decay to use with sgd."
)
parser.add_argument(
    "-mom",
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum to use with sgd optimiser.",
)
parser.add_argument(
    "-min_lr",
    "--min_learning_rate",
    type=float,
    default=1e-5,
    help="Minimum learning rate to use with sgd and with ReduceLearningRateonPlateau.",
)
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="Patience (number of epochs for a model to be considered as converged) to use with sgd and with ReduceLearningRateonPlateau.",
)
parser.add_argument(
    "--factor",
    type=float,
    default=0.2,
    help="Learning rate changing factor to use with sgd and with ReduceLearningRateonPlateau.",
)

# Early Stopping Parameters
parser.add_argument(
    "--early_patience",
    type=str,
    default="200,200,200",
    help="Number of epochs (for each phase) to consider before Early Stopping, for example: 10,20,5.",
)
parser.add_argument(
    "--early_delta",
    type=int,
    default=1e-4,
    help="Minimum change in the monitored quantity to qualify as an improvement for Early Stopping.",
)

args = parser.parse_args()

# select defined gpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# verify loss parameters
if args.beta is None:
    print("Please define a value for beta.")
    sys.exit(-1)

masks = False
if args.loss == "hybrid":
    masks = True  # ensure that the dataloader returns object detection masks
    if args.gamma is None:
        print("Please define a value for gamma.")
        sys.exit(-1)
    loss_fn = hybrid_explanation_loss(beta=args.beta, gamma=args.gamma)
elif args.loss == "unsupervised":
    loss_fn = unsupervised_explanation_loss(beta=args.beta)

# split epochs' string into an array of 3 integer values, one for each training phase
eps = args.nr_epochs.split(",")
nr_epochs = np.array([int(x) for x in eps])

# split learning rates' string into an array of 3 integer values, one for each training phase
lrs = args.learning_rate.split(",")
lr = np.array([float(x) for x in lrs])

# split alphas' string into an array of 3 integer values, one for each training phase
alphas = args.alpha.split(",")
alpha = np.array([float(x) for x in alphas])

# split patience for early stopping string into an array of 3 integer values, one for each training phase
early_patience = args.early_patience.split(",")
early_patience = np.array([int(x) for x in early_patience])

img_size = tuple(args.img_size)

# create folder to store the results and models
folder = args.folder
timestamp, path = utils.create_folder(folder)

# save training config
with open(os.path.join(path, timestamp + "_train_parameters_summary.txt"), "w") as f:
    f.write(str(args))

# instantiate model class
model = ExplainerClassifierCNN(
    num_classes=args.nr_classes,
    img_size=img_size,
    clf=args.classifier,
    init_bias=args.init_bias,
    pretrained=args.pretrained,
)

# save a summary of the model used
model.save_architecture(timestamp, path)


# define class weights for imbalanced data
if args.class_weights:
    class_weights = "balanced"
else:
    class_weights = None

# load data and create training and validation data generators
tr_df, val_df, _, weights, classes = load_data(
    folder=args.dataset_path,
    dataset=args.dataset,
    masks=masks,
    class_weights=class_weights,
)

train_datagen = DataGenerator(
    tr_df,
    batch_size=args.batch_size,
    img_size=img_size,
    num_classes=args.nr_classes,
    masks=masks,
    aug_prob=args.aug_prob,
    shuffle=True,
)
val_datagen = DataGenerator(
    val_df,
    batch_size=args.batch_size,
    img_size=img_size,
    num_classes=args.nr_classes,
    masks=masks,
    aug_prob=0,
    shuffle=True,
)

# Start training (3 phases)
for phase in range(3):
    print("PHASE ", str(phase))
    if nr_epochs[phase] == 0:
        continue

    if phase == 0:
        # freeze explainer
        utils.freeze(model.explainer)
    elif phase == 1:
        # unfreeze explainer and freeze classifier
        utils.unfreeze(model.explainer)
        utils.freeze(model.classifier)
    elif phase == 2:
        # unfreeze classifier
        utils.unfreeze(model.classifier)

    if args.opt == "sgd":
        opt = SGD(lr=lr[phase], decay=args.decay, momentum=args.momentum,)
    elif args.opt == "adadelta":
        opt = Adadelta()

    model.e2e_model.compile(
        optimizer=opt,
        loss_weights={
            "classifier": float(alpha[phase]),
            "explainer": 1.0 - float(alpha[phase]),
        },
        loss={"explainer": loss_fn, "classifier": "categorical_crossentropy"},
        metrics={"classifier": ["accuracy"]},
    )

    model_filename = timestamp + "_phase" + str(phase) + "_model.h5"
    model_path = os.path.join(path, model_filename)

    callbacks = utils.config_callbacks(
        model,
        args.factor,
        args.patience,
        args.min_learning_rate,
        early_patience[phase],
        args.early_delta,
        model_path,
    )

    history = model.e2e_model.fit(
        train_datagen,
        validation_data=val_datagen,
        epochs=nr_epochs[phase],
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=False,
        # class_weight={"classifier": weights, "explainer": None}, --> waiting for tensorflow to fix the bug and make this possible
    )
    hist_df = utils.save_history(
        history,
        os.path.join(path, str(timestamp + "_phase" + str(phase) + "_history.csv")),
    )
    utils.plot_metric_train_val(
        nr_epochs[phase],
        hist_df,
        "classifier_loss",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_classifier_loss.png"),
        "Classifier Loss",
    )
    utils.plot_metric_train_val(
        nr_epochs[phase],
        hist_df,
        "explainer_loss",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_explainer_loss.png"),
        "Explainer Loss",
    )
    utils.plot_metric_train_val(
        nr_epochs[phase],
        hist_df,
        "loss",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_global_loss.png"),
        "Global Loss",
    )
    utils.plot_metric_train_val(
        nr_epochs[phase],
        hist_df,
        "classifier_accuracy",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_classifier_acc.png"),
        "Accuracy",
    )

    model.save_explanations(val_datagen, phase, path, classes=classes, cmap="viridis")
