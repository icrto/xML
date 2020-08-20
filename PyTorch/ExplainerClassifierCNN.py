from VGG import VGGClf
from ResNetMod import resnet152, resnet101, resnet50, resnet34, resnet18
from Explainer import Explainer
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
import utils
import Losses
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


class ExplainerClassifierCNN(nn.Module):
    """ Class ExplainerClassifierCNN
    """

    def __init__(
        self,
        img_size=(224, 224),
        num_classes=2,
        clf="resnet50",
        init_bias=3.0,
        pretrained=False,
    ):
        super(ExplainerClassifierCNN, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        self.clf = clf
        self.init_bias = init_bias
        self.pretrained = pretrained

        # classifier
        if "resnet152" in self.clf.lower():
            self.decmaker = resnet152(
                pretrained=self.pretrained,
                progress=True,
                img_size=img_size,
                num_classes=num_classes,
            )
        elif "resnet101" in self.clf.lower():
            self.decmaker = resnet101(
                pretrained=self.pretrained,
                progress=True,
                img_size=img_size,
                num_classes=num_classes,
            )
        elif "resnet50" in self.clf.lower():
            self.decmaker = resnet50(
                pretrained=self.pretrained,
                progress=True,
                img_size=img_size,
                num_classes=num_classes,
            )
        elif "resnet34" in self.clf.lower():
            self.decmaker = resnet34(
                pretrained=self.pretrained,
                progress=True,
                img_size=img_size,
                num_classes=num_classes,
            )
        elif "resnet18" in self.clf.lower():
            self.decmaker = resnet18(
                pretrained=self.pretrained,
                progress=True,
                img_size=img_size,
                num_classes=num_classes,
            )
        else:
            self.decmaker = VGGClf(img_size=self.img_size, num_classes=self.num_classes)

        # explainer
        self.explainer = Explainer(img_size=self.img_size, init_bias=self.init_bias)

    def train(
        self, dataloader, optimiser, device, args, phase, weights, alpha, disable=False
    ):

        self.decmaker.train()
        self.explainer.train()

        dec_criterion = torch.nn.CrossEntropyLoss(reduction="mean", weight=weights)

        if phase == 0:
            # freeze explainer
            utils.freeze(self.explainer)
            self.explainer.eval()
        elif phase == 1:
            # unfreeze explainer and freeze classifier
            utils.unfreeze(self.explainer)
            utils.freeze(self.decmaker)
        elif phase == 2:
            # unfreeze classifier
            utils.unfreeze(self.decmaker)

        for batch_imgs, batch_labels, _, batch_masks in tqdm(
            dataloader, disable=disable
        ):
            optimiser.zero_grad()
            if len(batch_masks) > 0:  # hybrid loss
                batch_imgs, batch_labels, batch_masks = (
                    batch_imgs.to(device),
                    batch_labels.to(device),
                    batch_masks.to(device),
                )
            else:
                batch_imgs, batch_labels = (
                    batch_imgs.to(device),
                    batch_labels.to(device),
                )

            # forward pass
            batch_expls = self.explainer(batch_imgs)
            batch_probs = self.decmaker(batch_imgs, batch_expls)

            # losses computation
            dec_loss = dec_criterion(batch_probs, batch_labels)

            if args.loss == "unsupervised":
                exp_loss = Losses.batch_unsupervised_explanation_loss(
                    batch_expls, float(args.beta), reduction="mean"
                )
            elif args.loss == "hybrid":
                exp_loss = Losses.batch_hybrid_explanation_loss(
                    batch_expls,
                    batch_masks,
                    float(args.beta),
                    float(args.gamma),
                    reduction="mean",
                )

            loss = float(alpha) * dec_loss + (1 - float(alpha)) * exp_loss

            loss.backward()
            optimiser.step()

    def validation(self, dataloader, device, args, alpha, disable=False):

        self.decmaker.eval()
        self.explainer.eval()

        val_loss = 0
        val_exp_loss = 0
        val_dec_loss = 0
        val_acc = 0

        whole_preds = []
        whole_labels = []
        dec_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():

            for batch_imgs, batch_labels, _, batch_masks in tqdm(
                dataloader, disable=disable
            ):

                if len(batch_masks) > 0:  # hybrid loss
                    batch_imgs, batch_labels, batch_masks = (
                        batch_imgs.to(device),
                        batch_labels.to(device),
                        batch_masks.to(device),
                    )
                else:
                    batch_imgs, batch_labels = (
                        batch_imgs.to(device),
                        batch_labels.to(device),
                    )

                # forward pass
                batch_expls = self.explainer(batch_imgs)
                batch_probs = self.decmaker(batch_imgs, batch_expls)

                # losses computation
                batch_dec_loss = dec_criterion(batch_probs, batch_labels)

                if args.loss == "unsupervised":
                    batch_exp_loss = Losses.batch_unsupervised_explanation_loss(
                        batch_expls, float(args.beta), reduction="sum"
                    )
                elif args.loss == "hybrid":
                    batch_exp_loss = Losses.batch_hybrid_explanation_loss(
                        batch_expls,
                        batch_masks,
                        float(args.beta),
                        float(args.gamma),
                        reduction="sum",
                    )

                batch_loss = (
                    float(alpha) * batch_dec_loss + (1 - float(alpha)) * batch_exp_loss
                )

                val_exp_loss += batch_exp_loss.item()
                val_dec_loss += batch_dec_loss.item()
                val_loss += batch_loss.item()

                batch_probs = F.softmax(batch_probs, dim=1)
                _, batch_preds = torch.max(batch_probs, 1)

                whole_labels.extend(batch_labels.data.cpu().numpy())
                whole_preds.extend(batch_preds.data.cpu().numpy())

            val_loss /= len(dataloader.dataset)
            val_exp_loss /= len(dataloader.dataset)
            val_dec_loss /= len(dataloader.dataset)
            val_acc += accuracy_score(whole_labels, whole_preds)

        return val_loss, val_exp_loss, val_dec_loss, val_acc

    def test(self, dataloader, device, args, alpha, disable=False):

        self.decmaker.eval()
        self.explainer.eval()

        val_loss = 0
        val_exp_loss = 0
        val_dec_loss = 0
        val_acc = 0

        whole_preds = []
        whole_probs = []
        whole_labels = []
        dec_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():

            for batch_imgs, batch_labels, _, batch_masks in tqdm(
                dataloader, disable=disable
            ):

                if len(batch_masks) > 0:  # hybrid loss
                    batch_imgs, batch_labels, batch_masks = (
                        batch_imgs.to(device),
                        batch_labels.to(device),
                        batch_masks.to(device),
                    )
                else:
                    batch_imgs, batch_labels = (
                        batch_imgs.to(device),
                        batch_labels.to(device),
                    )

                # forward pass
                batch_expls = self.explainer(batch_imgs)
                batch_probs = self.decmaker(batch_imgs, batch_expls)

                # losses computation
                batch_dec_loss = dec_criterion(batch_probs, batch_labels)

                if args.loss == "unsupervised":
                    batch_exp_loss = Losses.batch_unsupervised_explanation_loss(
                        batch_expls, float(args.beta), reduction="sum"
                    )
                elif args.loss == "hybrid":
                    batch_exp_loss = Losses.batch_hybrid_explanation_loss(
                        batch_expls,
                        batch_masks,
                        float(args.beta),
                        float(args.gamma),
                        reduction="sum",
                    )

                batch_loss = (
                    float(alpha) * batch_dec_loss + (1 - float(alpha)) * batch_exp_loss
                )

                val_exp_loss += batch_exp_loss.item()
                val_dec_loss += batch_dec_loss.item()
                val_loss += batch_loss.item()

                batch_probs = F.softmax(batch_probs, dim=1)
                _, batch_preds = torch.max(batch_probs, 1)

                whole_labels.extend(batch_labels.data.cpu().numpy())
                whole_preds.extend(batch_preds.data.cpu().numpy())
                whole_probs.extend(batch_probs.data.cpu().numpy())

            val_loss /= len(dataloader.dataset)
            val_exp_loss /= len(dataloader.dataset)
            val_dec_loss /= len(dataloader.dataset)
            val_acc += accuracy_score(whole_labels, whole_preds)

        return (
            val_loss,
            val_exp_loss,
            val_dec_loss,
            val_acc,
            np.array(whole_probs),
            np.array(whole_preds),
            np.array(whole_labels),
        )

    def save_explanations(
        self,
        dataloader,
        phase,
        device,
        path,
        disable=False,
        test=False,
        classes=None,
        cmap=None,
    ):
        """ Generates and saves explanations for a set of images given by dataloader

        Arguments:
            dataloader {torch.utils.data.Dataloader} -- dataloader
            phase {int} -- training phase
            device {torch.device} -- device (CPU or GPU) to run inference
            path {str} -- directory to store the generated explanations

        Keyword Arguments:
            disable {bool} -- enable/disable progress bar (default: {False})
            test {bool} -- whether we are running inference on the test set or not (default: {False})
            classes {list} -- list of class names (default: {None})
            cmap {str} -- matplotlib colourmap for the produced explanations (default: {None})
        """
        # defines colourmap and pixel value range
        if cmap == "None":
            cmap = "seismic"
            mode = "captum"  # for better comparison with captum methods
            vmin, vmax = -1, 1
        else:
            vmin, vmax = 0, 1
            mode = "default"

        # puts model in eval mode
        self.decmaker.eval()
        self.explainer.eval()

        print("\nSAVING EXPLANATIONS")
        timestamp = path.split("/")[-1]
        with torch.no_grad():

            for batch_imgs, batch_labels, batch_names, _ in tqdm(
                dataloader, disable=disable
            ):

                batch_imgs, batch_labels = (
                    batch_imgs.to(device),
                    batch_labels.to(device),
                )

                # forward pass to predict classification output and the corresponding explanation
                batch_expls = self.explainer(batch_imgs)
                batch_probs = self.decmaker(batch_imgs, batch_expls)
                batch_probs = F.softmax(batch_probs, dim=1)

                for idx, img in enumerate(batch_imgs):  # transverses the batch

                    img = np.transpose(
                        batch_imgs[idx].squeeze().cpu().detach().numpy(), (2, 1, 0)
                    )
                    expl = np.swapaxes(batch_expls[idx].squeeze().cpu().numpy(), 0, 1)

                    string = "Label: " + classes[batch_labels[idx]] + "\n"
                    for i, c in enumerate(classes):
                        string += (
                            c
                            + " "
                            + str(
                                round(
                                    batch_probs[idx][i].squeeze().cpu().numpy().item(),
                                    6,
                                )
                            )
                            + "\n"
                        )

                    # saves figure comparing the original image with the generated explanation side-by-side
                    plt.figure(1, figsize=(12, 5))
                    plt.subplot(121)
                    ax = plt.gca()
                    ax.relim()
                    ax.autoscale()
                    ax.axes.get_yaxis().set_visible(False)
                    ax.axes.get_xaxis().set_visible(False)
                    plt.title("Original Image", fontsize=14)
                    plt.imshow(img)

                    plt.subplot(122)
                    ax = plt.gca()
                    ax.axes.get_yaxis().set_visible(False)
                    ax.axes.get_xaxis().set_visible(False)
                    plt.title("Explanation", fontsize=14)
                    plt.imshow(expl, vmin=vmin, vmax=vmax, cmap=cmap)

                    plt.text(
                        0.91, 0.5, string, fontsize=12, transform=plt.gcf().transFigure
                    )
                    if test:
                        plt.savefig(
                            os.path.join(
                                path,
                                "{}_phase{}_ex_img_test_{}_{}_{}".format(
                                    timestamp,
                                    str(phase),
                                    mode,
                                    cmap,
                                    batch_names[idx].split("/")[-1],
                                ),
                            ),
                            bbox_inches="tight",
                            transparent=True,
                            pad_inches=0.1,
                        )
                    else:
                        plt.savefig(
                            os.path.join(
                                path,
                                "{}_phase{}_ex_img_{}".format(
                                    timestamp,
                                    str(phase),
                                    batch_names[idx].split("/")[-1],
                                ),
                            ),
                            bbox_inches="tight",
                            transparent=True,
                            pad_inches=0.1,
                        )
                    plt.close()
        print()

    def checkpoint(self, filename, epoch, batch_loss, batch_acc, optimiser):
        """ Saves model when validation loss/accuracy improves and at every epoch

        Arguments:
            filename {str} -- checkpoint filename
            epoch {int} -- current epoch
            batch_loss {float} -- current loss
            batch_acc {float} -- current accuracy
            optimiser {torch.optim} -- current optimiser state
        """

        # Update best loss
        if not hasattr(self, "best_loss") or epoch == 0:
            self.best_loss = float("inf")
        if not hasattr(self, "best_acc") or epoch == 0:
            self.best_acc = float(0)

        if self.best_loss > batch_loss:
            self.best_loss = batch_loss

        if self.best_acc < batch_acc:
            self.best_acc = batch_acc

        # Create checkpoint
        chkpt = {
            "epoch": epoch,
            "best_loss": batch_loss,
            "best_acc": batch_acc,
            "decmaker": self.decmaker.module.state_dict()
            if type(self.decmaker) is nn.parallel.DistributedDataParallel
            else self.decmaker.state_dict(),
            "explainer": self.explainer.module.state_dict()
            if type(self.explainer) is nn.parallel.DistributedDataParallel
            else self.explainer.state_dict(),
            "optimiser": optimiser.state_dict(),
        }

        # Save best checkpoint
        if self.best_loss == batch_loss:
            best_loss = filename + "_best_loss.pt"
            torch.save(chkpt, best_loss)

        if self.best_acc == batch_acc:
            best_acc = filename + "_best_acc.pt"
            torch.save(chkpt, best_acc)

        torch.save(chkpt, filename + "_latest.pt")

        # Delete checkpoint
        del chkpt
