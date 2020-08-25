from Explainer import explainer
from VGG import vgg
from ResNet50Mod import resnet50
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers as KL
import numpy as np
from tensorflow.keras.models import Model
import os
import utils
import sys
import matplotlib.pyplot as plt


class ExplainerClassifierCNN:
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
        if self.clf == "resnet50":
            self.classifier = resnet50(
                img_size=self.img_size,
                num_classes=self.num_classes,
                pretrained=self.pretrained,
            )
        else:
            self.classifier = vgg(img_size=self.img_size, num_classes=self.num_classes)

        # explainer
        self.explainer = explainer(img_size=self.img_size, init_bias=self.init_bias)

        # build the model
        self.build_model()

    def build_model(self):
        input_image = KL.Input(tuple(list(self.img_size) + [3]), name="input_img")
        explanation = self.explainer(input_image)
        decision = self.classifier([explanation, input_image])
        self.e2e_model = Model(inputs=[input_image], outputs=[explanation, decision])

    def save_architecture(self, timestamp, path):
        self.exp_model_filename = timestamp + "_model_exp.png"
        self.dec_model_filename = timestamp + "_model_clf.png"
        self.e2e_model_filename = timestamp + "_model_e2e.png"

        plot_model(self.explainer, to_file=os.path.join(path, self.exp_model_filename))
        print("Model printed to " + os.path.join(path, self.exp_model_filename))

        plot_model(self.classifier, to_file=os.path.join(path, self.dec_model_filename))
        print("Model printed to " + os.path.join(path, self.dec_model_filename))

        plot_model(self.e2e_model, to_file=os.path.join(path, self.e2e_model_filename))
        print("Model printed to " + os.path.join(path, self.e2e_model_filename))

    def save_explanations(
        self, datagen, phase, path, test=False, classes=None, cmap=None,
    ):
        """ Generates and saves explanations for a set of images given by dataloader

            Arguments:
                datagen {tf.keras.utils.Sequence} -- data generator
                phase {int} -- training phase
                path {str} -- directory to store the generated explanations

            Keyword Arguments:
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

        print("\nSAVING EXPLANATIONS")
        timestamp = path.split("/")[-1]

        for batch_imgs, input_dict in datagen:
            batch_labels = datagen.batch_labels
            batch_names = datagen.batch_names

            batch_expls, batch_probs = self.e2e_model.predict(
                (batch_imgs, input_dict), verbose=0
            )

            for idx, img in enumerate(batch_imgs):
                img = batch_imgs[idx]
                expl = batch_expls[idx]

                string = "Label: " + classes[batch_labels[idx]] + "\n"
                for i, c in enumerate(classes):
                    string += c + " " + str(round(batch_probs[idx][i], 6,)) + "\n"

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
                                timestamp, str(phase), batch_names[idx].split("/")[-1],
                            ),
                        ),
                        bbox_inches="tight",
                        transparent=True,
                        pad_inches=0.1,
                    )
                plt.close()
        print()
