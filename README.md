# Understanding the decisions of CNNs: an in-model approach
This is the official implementation of the paper [Understanding the decisions of CNNs: an in-model approach](https://doi.org/10.1016/j.patrec.2020.04.004). 

<p align="justify">
It also includes my master thesis "Producing Decisions and Explanations: A Joint Approach Towards Explainable CNNs" (full document and final presentation), in which the paper was based and further extended.

If you use this repository, please cite:
</p>

```

@article{
  author = {Rio-Torto, Isabel and Fernandes, Kelwin and Teixeira, Luís F}
  doi = {10.1016/j.patrec.2020.04.004},
  issn = {01678655},
  journal = {Pattern Recognition Letters},
  month = {may},
  number = {C},
  pages = {373--380},
  title = {{Understanding the decisions of CNNs: An in-model approach}},
  url = {http://www.sciencedirect.com/science/article/pii/S0167865520301240}
  volume = {133},
  year = {2020}
}
```

Some preliminary work was published in:

```
@InProceedings{
  author = {Rio-Torto, Isabel and Fernandes, Kelwin and Teixeira, Luís F.},
  editor = {Morales, Aythami and Fierrez, Julian and Sánchez, José Salvador and Ribeiro, Bernardete},
  title = {Towards a Joint Approach to Produce Decisions and Explanations Using CNNs},
  booktitle = {Pattern Recognition and Image Analysis},
  year = {2019},
  publisher = {Springer International Publishing},
  pages = {"3--15"},
  url = {https://link.springer.com/chapter/10.1007/978-3-030-31332-6_1},
 }
```

**IMPORTANT: the Keras version is currently not up-to-date with the most recent PyTorch version!**

## Contents

[***Architecture***](https://github.com/icrto/xML#Architecture)

[***Loss***](https://github.com/icrto/xML#Loss)

[***Metrics***](https://github.com/icrto/xML#Metrics)

[***Implementation***](https://github.com/icrto/xML#Implementation)

[***Training***](https://github.com/icrto/xML#Training)

[***Results***](https://github.com/icrto/xML#Results)

[***Requirements***](https://github.com/icrto/xML#Requirements)

[***Usage***](https://github.com/icrto/xML#Usage)

[***Credits***](https://github.com/icrto/xML#Credits)


[***TODO***](https://github.com/icrto/xML#TODO)



## Architecture
<p align="justify">
With the advent of Deep Learning, in particular Convolutional Neural Networks, there is also a growing demand for Interpretability/Explainability of these highly complex and abstract models, mainly in highly regulated areas such as medicine. Several interpretability methods have already been proposed, and these are for the most part post-model methods, i.e. methods applied after the main (classification) model is trained.
</p>

<p align="justify">
However, we argue that interpretability should be taken into account from the start, as a design requirement and a desirable property of the system. As such, we propose an <b>in-model approach</b>, i.e. an interpretability method applied during training of the main model. The proposed joint architecture, as can be seen below, is composed by and Explainer and a Classifier.
</p>

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/architecture.png">

<p align="justify">
  The <b>explainer</b>, as the name implies, is responsible for taking an input image and producing (in an <b>unsupervised manner</b>) the corresponding visual explanation for why the classification module classified that image in a certain way. This visual explanation takes the form of a <b>heatmap</b>, highlighting the most relevant regions that lead to the final decision. This module, an encoder-decoder, is based on the widely known UNET, originally proposed for medical image segmentation.
</p>

<p align="justify">
  The <b>classifier</b> (the "main" module) not only takes as input the same image, but also the output of the explainer, i.e., it is trained using the explanation. The main idea is that the classifier should focus only on the relevant image regions, which is aligned with the intuition that, when explaining a decision (whether or not an image contains a dog) humans tend to first separate what is the object of interest and what is "background", and then proceed to look for patterns in the region where the object is in order to classify it correctly. Conversely, sometimes humans cannot tell if an object belongs to some class, but can tell which regions of the image do not contain said class (when classifying cervical cancer lesions, one is sure that the area outside the cervix is irrelevant for this problem).
</p>

<p align="justify">
  Both modules are connected by the purple arrows represented in the picture. These arrows represent one of the two inputs to a custom multiplication layer responsible for performing the element-wise multiplication of the classifier layer with the the explainer’s output. In order for this operation to be possible, the explainer's output is downsampled by average pooling (see the paper for further details on why it is essential that this is done by average pooling instead of max pooling). These connections allow the classifier to focus only on the important image regions highlighted by the explainer.
</p>

<p align="justify">
  Although in the picture we represented a VGG16-based classifier, <b>any classification model can be used</b>, provided that the correct connections are introduced. In fact, the results we show were obtained using a ResNet18-based classifier. Since this joint architecture aims at explaining the classifier’s decisions, the classifier should be chosen first, depending on the classification problem and the available data; <b>the explainer must adapt to the classifier and not the other way around</b>.
</p>

## Loss
<p align="justify">
  Our main loss function is simply defined as a <b>weighted sum of a classification loss and an explanation loss</b>. Briefly, the hyperparameter α allows the user to control how much importance to give to each of the modules. During training (see section <a href="https://github.com/icrto/xML#Training">training</a>), we use different α values according to the module being trained at each stage.
</p>

![JointLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D%20%3D%20%5Calpha%20%5Cmathcal%7BL%7D_%7Bclass%7D%20&plus;%20%281%20-%20%5Calpha%29%20%5Cmathcal%7BL%7D_%7Bexpl%7D)

<p align="justify">
  The <b>classification</b> loss we used was <b>categorical cross entropy</b>, similarly to what is done in conventional multiclass classification problems. For the explanation loss we propose two alternatives, an <b>unsupervised approach</b> and a <b>hybrid approach</b> (unsupervised + weakly-supervised terms).
</p>

### Unsupervised Explanation Loss
<p align="justify">
  An explanation should be <b>sparse</b> (contain the most important information, eliminating redundancy) and <b>spatially contiguous</b> (semantically related parts of the images should be connected). Once more, we introduce the β hyperparameter as a way to control the influence of these properties in the produced explanations of the mini-batch.
</p>

![UnsupervisedLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20%3D%20%5Cbeta%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D_i%29%20&plus;%20%281%20-%20%5Cbeta%29%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D_i%29)

<p align="justify">
  Through the <b>penalised l1 norm</b> we ensure sparsity, by minimising the pixel-wise content of the produced heatmaps, performing feature selection. As is further detailed in the paper, this penalty works as an explanation budget, limiting the percentage of the input image that can be considered an explanation.
</p>

![Sparsity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%20%7C%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20%5Clabel%7Beq%3Asparsity%7D)

<p align="justify">
  Spatial contiguity is promoted through the <b>total variation</b> loss term, which encourages minimised local spatial transitions, both horizontally and vertically (we want the absolute differences between each row/column and the next to be small). 
</p>

![Contiguity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%7C%5Chat%7Bz%7D_%7Bi&plus;1%2Cj%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20&plus;%20%7C%5Chat%7Bz%7D_%7Bi%2Cj&plus;1%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C)

### Hybrid Explanation Loss
<p align="justify">
  In order to impose stronger constraints on the explanations, but still being able to do it without needing to annotate our data for this purpose, we proposed a hybrid approach, which involves the previously mentioned unsupervised loss and an extra weakly-supervised loss term (introduced with the γ hyperparameter). This extra term is introduced to drive the explanations not to focus on regions outside the interest regions. We do this resorting to object detection annotations (aka bounding boxes), and in this case the regions of interest are the areas inside the bounding boxes. The main idea here (further details can be found in section 3.5.2 of the paper) is to <b>punish explanations outside this region of interest</b>, by computing the product between the inverted mask (this binary mask is filled with 1s in the region of interest - the object - and 0s everywhere else) and the explanation.
</p>

![HybridLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_hybrid%7D%20%3D%20%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20&plus;%20%5Cgamma%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%7D%5Cright%7C)

## Metrics

<a href="https://www.sciencedirect.com/science/article/pii/S1051200417302385">Methods for interpreting and understanding deep neural networks</a>

![AOPC](https://latex.codecogs.com/gif.latex?AOPC%20%3D%20%5Cfrac%7B1%7D%7BL%20&plus;%201%7D%5Cbigg%5Clangle%5Csum%5Climits_%7Bk%3D0%7D%5E%7BL%7D%20f%28x_%7BMoRF%7D%5E%7B%280%29%7D%29%20-%20f%28x_%7BMoRF%7D%5E%7B%28k%29%7D%29%5Cbigg%5Crangle_%7Bp%28x%29%7D)

![POMPOM](https://latex.codecogs.com/gif.latex?POMPOM%20%3D%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%20%7D%5B%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%5D%20%3E%20%5Cepsilon%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%20%7D%5B%281-z_%7Bi%2Cj%7D%29%5D%20%3E%20%5Cepsilon%20&plus;%20%5Cepsilon%7D%5Cright%7C)


## Implementation

- ConvMod.py
- Dataset.py
- Explainer.py
- ExplainerClassifierCNN.py
- Losses.py
- ResNetMod.py
- test.py
- train.py
- utils.py
- VGG.py

## Training

### 3 Phase Process
### Hyperparameters


## Results
### ImagenetHVZ

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_explanations.png" width=1420 height=400>

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_avgfunction.png">

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_aopc.png">

### NIH-NCI Cervical Cancer
<img align="center" width=1420 height=400 src="https://github.com/icrto/xML/blob/master/example_images/cervix_grid_explanations.png">

## Requirements

For the PyTorch version please consult [requirements_pytorch.txt](https://github.com/icrto/xML/blob/master/requirements_pytorch.txt). 

I'm using `PyTorch 1.3.1` and `Keras 2.2.4` in `Python 3.6.9`.

## Usage

Train

```
python3 train.py imagenetHVZ --nr_epochs 10,10,60 -bs 8 --init_bias 3.0 --loss hybrid --alpha 1.0,0.25,0.9 --beta 0.9 --gamma 1.0 -lr_clf 0.01,0,0.01 -lr_expl 0,0.01,0.01 --aug_prob 0.2 --opt sgd -clf resnet50 --early_patience 100,100,10 --folder <path_to_destination_folder>
```

Test

```
python3 test.py <path_to_model> imagenetHVZ 0.9 -bs 8 -clf resnet50 --init_bias 3.0 --loss hybrid --beta 0.9 --gamma 1.0
```


## Credits

Neural network artwork → [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

***PyTorch Version***

Data augmentation → [albumentations](https://github.com/albumentations-team/albumentations)

Early stopping → [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch.git)

Network summary → [pytorch-summary](https://github.com/sksq96/pytorch-summary.git)

Other interpretability methods → [captum](https://captum.ai)

Deep Taylor Decomposition → [Deep-Taylor-Decomposition](https://github.com/myc159/Deep-Taylor-Decomposition)

***Keras Version***

Other interpretability methods → [innvestigate](https://github.com/albermax/innvestigate)



## TODO
- [ ] update keras version
- [ ] results on imagenet16
- [ ] update synthetic dataset generation script

## Contact Info

If you wish to know more about this project, do not hesitate to contact me at icrtto@gmail.com or to open an issue.
