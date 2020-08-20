import os
from ExplainerClassifierCNN import ExplainerClassifierCNN
from Dataset import Dataset, load_data
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import argparse
import utils
import csv
from EarlyStopping import EarlyStopping
from summary import summary
import sys

torch.manual_seed(0)
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
<<<<<<< HEAD
parser.add_argument(
    "dataset",
    type=str,
    choices=["simplified_no_colour", "NIH-NCI", "imagenet16", "imagenetHVZ"],
    help="Dataset to load.",
)
parser.add_argument(
    "--nr_classes", type=int, default=2, help="Number of target classes."
)
parser.add_argument(
    "--img_size", type=tuple, default=(224, 224), help="Input image size."
)
parser.add_argument(
    "--aug_prob",
    type=float,
    default=0,
    help="Probability of applying data augmentation to each image.",
)
=======
parser.add_argument('dataset', type=str, choices=['simplified_no_colour', 'NIH-NCI', 'imagenetHVZ'],
                    help='Dataset to load.')
parser.add_argument('--nr_classes', type=int, default=2,
                    help='Number of target classes.')
parser.add_argument('--img_size', type=tuple,
                    default=(224, 224), help='Input image size.')
parser.add_argument('--aug_prob', type=float, default=0,
                    help='Probability of applying data augmentation to each image.')
>>>>>>> 2c6582f40429c23fb8a3a81dff42b1196b1c1cf0

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
    choices=["vgg", "resnet152", "resnet101", "resnet50", "resnet34", "resnet18"],
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
    "--beta", type=str, help="Lexplic_unsup = beta * L1 + (1-beta) * Total Variation"
)
parser.add_argument(
    "--gamma",
    type=str,
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
    "-lr_clf",
    "--learning_rate_classifier",
    type=str,
    default="1e-3,0,1e-4",
    help="Learning rate for each training phase of the classifier, for example: 1e-3,0,1e-4.",
)
parser.add_argument(
    "-lr_expl",
    "--learning_rate_explainer",
    type=str,
    default="0,1e-4,1e-4",
    help="Learning rate for each training phase of the explainer, for example: 0,1e-4,1e-4.",
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# verify loss parameters
loss = args.loss
masks = False
if loss == "hybrid":
    masks = True  # ensure that the dataloader returns object detection masks

if loss == "unsupervised":
    if args.beta is None:
        print("Please define a value for beta.")
        sys.exit(-1)
elif loss == "hybrid":
    if args.beta is None:
        print("Please define a value for beta.")
        sys.exit(-1)
    if args.gamma is None:
        print("Please define a value for gamma.")
        sys.exit(-1)
else:
    print("Invalid loss function. Try again with <unsup> or <hybrid>.")
    sys.exit(-1)

# split epochs' string into an array of 3 integer values, one for each training phase
eps = args.nr_epochs.split(",")
nr_epochs = np.array([int(x) for x in eps])

# split classifier learning rates' string into an array of 3 integer values, one for each training phase
lrs_clf = args.learning_rate_classifier.split(",")
lr_clf = np.array([float(x) for x in lrs_clf])

# split explainer learning rates' string into an array of 3 integer values, one for each training phase
lrs_expl = args.learning_rate_explainer.split(",")
lr_expl = np.array([float(x) for x in lrs_expl])

# split alphas' string into an array of 3 integer values, one for each training phase
alphas = args.alpha.split(",")
alpha = np.array([float(x) for x in alphas])

# split patience for early stopping string into an array of 3 integer values, one for each training phase
early_patience = args.early_patience.split(",")
early_patience = np.array([int(x) for x in early_patience])

# create folder to store the results and models
folder = args.folder
timestamp, path = utils.create_folder(folder)

# save training config
with open(os.path.join(path, timestamp + "_train_parameters_summary.txt"), "w") as f:
    f.write(str(args))

# instantiate model class and put model in GPU (if GPU available)
model = ExplainerClassifierCNN(
    num_classes=args.nr_classes,
    img_size=args.img_size,
    clf=args.classifier,
    init_bias=args.init_bias,
    pretrained=args.pretrained,
)
model.to(device)

# save a summary of the model used
summary(
    model.decmaker,
    [(3, 224, 224), (1, 224, 224)],
    filename=os.path.join(path, timestamp + "_model_dec_info.txt"),
)
summary(
    model.explainer,
    (3, 224, 224),
    filename=os.path.join(path, timestamp + "_model_exp_info.txt"),
)

# define class weights for imbalanced data
if args.class_weights:
    class_weights = "balanced"
else:
    class_weights = None

# load data and create training and validation loaders
tr_df, val_df, _, weights, classes = load_data(
    folder=args.dataset_path,
    dataset=args.dataset,
    masks=masks,
    class_weights=class_weights,
)
weights = torch.FloatTensor(weights).to(device)

train_dataset = Dataset(
    tr_df, masks=masks, img_size=args.img_size, aug_prob=args.aug_prob
)
val_dataset = Dataset(val_df, masks=masks, img_size=args.img_size, aug_prob=0)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
)


# Start training (3 phases)
for phase in range(3):
    print("PHASE ", str(phase))
    if nr_epochs[phase] == 0:
        continue

    early_stopping = EarlyStopping(
        patience=early_patience[phase],
        delta=args.early_delta,
        verbose=True,
        folder=path,
        timestamp=timestamp,
    )

    # define history and checkpoint files for each phase
    history_file = os.path.join(
        path, timestamp + "_phase" + str(phase) + "_history.csv"
    )
    checkpoint_filename = os.path.join(path, timestamp + "_phase" + str(phase))

    with open(history_file, "a+") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(
            [
                "epoch",
                "train_global_loss",
                "train_decmaker_loss",
                "train_explainer_loss",
                "train_decmaker_acc",
                "val_global_loss",
                "val_decmaker_loss",
                "val_explainer_loss",
                "val_decmaker_acc",
            ]
        )

    # select optimizer
    if args.opt == "adadelta":
        opt = optim.Adadelta(
            [
                {"params": model.decmaker.parameters(), "lr": lr_clf[phase]},
                {"params": model.explainer.parameters(), "lr": lr_expl[phase]},
            ],
            weight_decay=args.decay,
        )
    else:
        opt = optim.SGD(
            [
                {"params": model.decmaker.parameters(), "lr": lr_clf[phase]},
                {"params": model.explainer.parameters(), "lr": lr_expl[phase]},
            ],
            weight_decay=args.decay,
            momentum=args.momentum,
        )

    # use scheduler when the classifier is a resnet; ignore it when classifier is vgg
    if "resnet" in args.classifier.lower():
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.factor,
            patience=args.patience,
            verbose=True,
            min_lr=args.min_learning_rate,
        )
    else:
        scheduler = None

    for epoch in range(nr_epochs[phase]):  # training loop for each phase
        print("Epoch %d / %d" % (epoch + 1, nr_epochs[phase]))

        print("Train")
        model.train(train_loader, opt, device, args, phase, weights, alpha[phase])

        print("Val")
        # pass entire training set through validation function to obtain metrics at the end of the training epoch
        (
            train_global_loss,
            train_explainer_loss,
            train_decmaker_loss,
            train_decmaker_acc,
        ) = model.validation(train_loader, device, args, alpha[phase])
        print(
            "Train Loss %f \tTrain Exp Loss %f \tTrain Dec Loss %f \tTrain Acc %f"
            % (
                train_global_loss,
                train_explainer_loss,
                train_decmaker_loss,
                train_decmaker_acc,
            )
        )

        (
            val_global_loss,
            val_explainer_loss,
            val_decmaker_loss,
            val_decmaker_acc,
        ) = model.validation(val_loader, device, args, alpha[phase])
        print(
            "Val Loss %f \tVal Exp Loss %f \tVal Dec Loss %f \tVal Acc %f"
            % (val_global_loss, val_explainer_loss, val_decmaker_loss, val_decmaker_acc)
        )

        print()

        # save model and epoch metrics
        model.checkpoint(
            checkpoint_filename, epoch, val_global_loss, val_decmaker_acc, opt
        )

        with open(history_file, "a+") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(
                [
                    epoch,
                    train_global_loss,
                    train_decmaker_loss,
                    train_explainer_loss,
                    train_decmaker_acc,
                    val_global_loss,
                    val_decmaker_loss,
                    val_explainer_loss,
                    val_decmaker_acc,
                ]
            )

        # check if early stopping should be activated according to the patience given for each training phase
        early_stopping(val_global_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # apply scheduling if scheduler is defined (only for resnet)
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_global_loss)

    # at the end of each training phase, plot the evolution of several metrics and plot the resulting explanations
    print()
    utils.plot_metric_train_val(
        epoch + 1,
        history_file,
        "decmaker_loss",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_decmaker_loss.png"),
        "Classifier Loss",
    )
    utils.plot_metric_train_val(
        epoch + 1,
        history_file,
        "explainer_loss",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_explainer_loss.png"),
        "Explainer Loss",
    )
    utils.plot_metric_train_val(
        epoch + 1,
        history_file,
        "global_loss",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_global_loss.png"),
        "Global Loss",
    )
    utils.plot_metric_train_val(
        epoch + 1,
        history_file,
        "decmaker_acc",
        path,
        os.path.join(path, timestamp + "_phase" + str(phase) + "_decmaker_acc.png"),
        "Accuracy",
    )
    model.save_explanations(
        val_loader, phase, device, path, classes=classes, cmap="viridis"
    )

