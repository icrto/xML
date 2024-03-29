import os
from ExplainerClassifierCNN import ExplainerClassifierCNN
from Dataset import Dataset, load_data
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import utils
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
parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size.")
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
    default=1.0,
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get folder name & timestamp of stored model
path = os.path.dirname(args.model_ckpt)
timestamp = path.split("/")[-1]

# verify loss parameters
loss = args.loss
if args.beta is None:
    print("Please define a value for beta.")
    sys.exit(-1)

masks = False
if loss == "hybrid":
    masks = True  # ensure that the dataloader returns object detection masks
    if args.gamma is None:
        print("Please define a value for gamma.")
        sys.exit(-1)

img_size = tuple(args.img_size)

# instantiate model class and load model to test
model = ExplainerClassifierCNN(
    num_classes=args.nr_classes,
    img_size=img_size,
    clf=args.classifier,
    init_bias=args.init_bias,
)

ckpt = torch.load(args.model_ckpt, map_location=device)
ckpt_epoch = ckpt["epoch"]
ckpt_loss = ckpt["best_loss"]
ckpt_acc = ckpt["best_acc"]

model.classifier.load_state_dict(ckpt["classifier"])
model.explainer.load_state_dict(ckpt["explainer"])
model.to(device)


# load test data and create test loaders
_, _, test_df, _, classes = load_data(
    folder=args.dataset_path, dataset=args.dataset, masks=masks, class_weights=None,
)

test_dataset = Dataset(test_df, masks=masks, img_size=img_size, aug_prob=0)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
)

# test
(
    test_global_loss,
    test_explainer_loss,
    test_classifier_loss,
    test_classifier_acc,
    whole_probs,
    _,
    whole_labels,
) = model.test(test_loader, device, args, args.alpha)

if args.nr_classes > 2:  # multiclass
    utils.plot_roc_curve_multiclass(
        os.path.join(path, timestamp), whole_probs, whole_labels, classes
    )
    utils.plot_precision_recall_curve_multiclass(
        os.path.join(path, timestamp), whole_probs, whole_labels, classes
    )
else:  # binary classification
    utils.plot_roc_curve(os.path.join(path, timestamp), whole_probs, whole_labels)
    utils.plot_precision_recall_curve(
        os.path.join(path, timestamp), whole_probs, whole_labels
    )

print(
    "Test Loss %f\tTest Exp Loss %f\tTest Dec Loss %f\tTest Acc %f"
    % (test_global_loss, test_explainer_loss, test_classifier_loss, test_classifier_acc)
)
print()

# save results
with open(os.path.join(path, timestamp + "test_stats_best_loss.txt"), "w") as f:
    print(
        "Epoch %f\tCkpt Loss %f\tCkpt Acc %f\tTest Loss %f\tTest Exp Loss %f\tTest Dec Loss %f\tTest Acc %f"
        % (
            ckpt_epoch,
            ckpt_loss,
            ckpt_acc,
            test_global_loss,
            test_explainer_loss,
            test_classifier_loss,
            test_classifier_acc,
        ),
        file=f,
    )

# save generated explanations
model.save_explanations(
    test_loader, 2, device, path, test=True, classes=classes, cmap=args.cmap
)

