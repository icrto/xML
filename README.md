# Understanding the decisions of CNNs: an in-model approach
This is the official implementation of the paper [Understanding the decisions of CNNs: an in-model approach](https://doi.org/10.1016/j.patrec.2020.04.004). 

<p align="justify">
  It also includes my master thesis "Producing Decisions and Explanations: A Joint Approach Towards Explainable CNNs" (full document and final presentation), in which the paper was based on and further extended.

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

[***Contact Info***](https://github.com/icrto/xML#contact-info)


## Architecture
<p align="justify">
  With the advent of Deep Learning, in particular Convolutional Neural Networks, there is also a growing demand for Interpretability/Explainability of these highly complex and abstract models, mainly in highly regulated areas such as medicine. Several interpretability methods have already been proposed, and these are for the most part post-model methods, i.e. methods applied after the main (classification) model is trained.
</p>

<p align="justify">
  However, we argue that interpretability should be taken into account from the start, as a design requirement and a desirable property of the system. As such, we propose an <b>in-model approach</b>, i.e. an interpretability method applied during training of the main model. The proposed joint architecture, as can be seen below, is composed by an Explainer and a Classifier, and is capable of producing visual explanations for classification decisions in an unsupervised way.
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/architecture.png">
</p>

<p align="justify">
  The <b>Explainer</b>, as the name implies, is responsible for taking an input image and producing (in an <b>unsupervised manner</b>) the corresponding visual explanation for why the classification module classified that image in a certain way. This visual explanation takes the form of a <b>heatmap</b>, highlighting the most relevant regions that lead to the final decision. This module, an encoder-decoder network, is based on the widely known UNET, originally proposed for medical image segmentation.
</p>

<p align="justify">
  The <b>Classifier</b> (the "main" module) not only takes as input the same image, but also the output of the Explainer, i.e., it is trained using the explanation. The main idea is that the classifier should focus only on the relevant image regions, aligned with the intuition that, when explaining a decision (whether or not an image contains a dog) humans tend to first separate what is the object of interest and what is "background", and then proceed to look for patterns in the region where the object is in order to classify it correctly. Conversely, sometimes humans cannot tell if an object belongs to some class, but can tell which regions of the image do not contain that class (when classifying cervical cancer lesions, one is sure that the area outside the cervix is irrelevant for the problem at hand).
</p>

<p align="justify">
  Both modules are connected by the purple arrows represented in the picture. These arrows represent one of the two inputs to a custom multiplication layer responsible for performing the element-wise multiplication of the Classifier layer with the the Explainer’s output. In order for this operation to be possible, the Explainer's output is downsampled by average pooling (see the paper for further details on why it is essential that this is done by average pooling instead of max pooling). These connections allow the Classifier to focus only on the important image regions highlighted by the Explainer.
</p>

<p align="justify">
  Although in the picture we represented a VGG16-based classifier, <b>any classification model can be used</b>, provided that the correct connections are introduced. In fact, the results we show were obtained using a ResNet18-based classifier. Since this joint architecture aims at explaining the classifier’s decisions, the classifier should be chosen first, depending on the classification problem and the available data; <b>the explainer must adapt to the classifier and not the other way around</b>.
</p>

## Loss
<p align="justify">
  Our main loss function is simply defined as a <b>weighted sum of a classification loss and an explanation loss</b>. Briefly, the hyperparameter α allows the user to control how much importance to give to each of the modules. During training (see section <a href="https://github.com/icrto/xML#Training">Training</a>), we use different α values according to the module being trained at each stage.
</p>

![JointLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D%20%3D%20%5Calpha%20%5Cmathcal%7BL%7D_%7Bclass%7D%20&plus;%20%281%20-%20%5Calpha%29%20%5Cmathcal%7BL%7D_%7Bexpl%7D)

<p align="justify">
  The <b>classification</b> loss we used was <a href="https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss">categorical cross entropy</a>, similarly to what is done in conventional multiclass classification problems. For the explanation loss we propose two alternatives, an <b>unsupervised approach</b> and a <b>hybrid approach</b> (unsupervised + weakly-supervised terms).
</p>

### Unsupervised Explanation Loss
<p align="justify">
  An explanation should be <b>sparse</b> (contain the most important information, eliminating redundancy) and <b>spatially contiguous</b> (semantically related parts of the images should be connected). Once more, we introduce the β hyperparameter as a way to control the influence of these properties in the produced explanations of the mini-batch.
</p>

![UnsupervisedLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20%3D%20%5Cbeta%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D_i%29%20&plus;%20%281%20-%20%5Cbeta%29%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D_i%29)

<p align="justify">
  Through the <b>penalised l1 norm</b> we ensure sparsity, by minimising the pixel-wise content of the produced heatmaps, performing feature selection. As is further detailed in the paper, this penalty works as an explanation budget, limiting the percentage of the input image pixels that can be considered as part of the explanation. The greater the penalty, the smaller the available budget and less pixels are highlighted.
</p>

![Sparsity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%20%7C%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20%5Clabel%7Beq%3Asparsity%7D)

<p align="justify">
  Spatial contiguity is promoted through the <b>total variation</b> loss term, which encourages minimised local spatial transitions, both horizontally and vertically (we want the absolute differences between each row/column and the next to be as small as possible). 
</p>

![Contiguity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%7C%5Chat%7Bz%7D_%7Bi&plus;1%2Cj%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20&plus;%20%7C%5Chat%7Bz%7D_%7Bi%2Cj&plus;1%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C)

### Hybrid Explanation Loss
<p align="justify">
  In order to impose stronger constraints on the explanations, but still being able to do it without needing to annotate our data for this purpose, we proposed a hybrid approach, which involves the previously mentioned unsupervised loss and an extra weakly-supervised loss term (introduced with the γ hyperparameter). This extra term is introduced to drive the explanations not to focus on regions outside the interest regions. We do this resorting to object detection annotations (aka bounding boxes), and in this case the regions of interest are the areas inside the bounding boxes. The main idea here (further details can be found in section 3.5.2 of the paper) is to <b>punish explanations outside this region of interest</b>, by computing the product between the inverted mask (this binary mask is filled with 1s in the region of interest - the object - and 0s everywhere else) and the explanation.
</p>

![HybridLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_hybrid%7D%20%3D%20%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20&plus;%20%5Cgamma%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%7D%5Cright%7C)

## Metrics

<p align="justify">
  Defining what a "good" explanation is and how to <b>measure</b> and compare <b>explanations</b> is still an <b>open problem</b> in this research field. A valid explanation for a certain person, might not be acceptable or understandable to another individual. In fact, explanations are context-, audience/user- and domain-dependent, making it difficult to define quantitative metrics.
</p>

<p align="justify">
  Montavon et. al proposed a <b>perturbation/occlusion/feature removal process to assess explanation quality</b>. This process is explained in <a href="https://www.sciencedirect.com/science/article/pii/S1051200417302385">Methods for interpreting and understanding deep neural networks</a> and a representative picture taken from that paper is found below.
<p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/perturbation_process.png" width=600 height=400>
</p> 

<p align="justify">
  The perturbation/feature/occlusion removal process can be described as follows: we start by first dividing the heatmaps (produced by some interpretability method under assessment) into a predefined <b>grid</b>. Afterwards, for each patch/tile of this grid we compute the average of its pixel values, so that patches with higher relevance (as indicated by the heatmap) give higher values. The next step is <b>sorting</b> these patches in <b>descending order</b> according to these previously computed values. Then, starting from the patch with higher relevance, we perturb that area in the original image and forward the now perturbed image through our classification network, obtaining the output value f(x) (softmax probability for the positive class, for example). Finally, we repeat this process, but this time adding to the initial perturbation the next most relevant patch, and so on until the whole image is perturbed or a certain number of perturbation steps is reached.
</p>

<p align="justify">
  The intuition is that, <b>the more relevant the patch, the more it will affect (decrease) the classification output</b>, so we expect a steeper decrease in f(x) in the initial stages of the perturbation process and a lower slope of the curve from there onwards. This curve is called the <b>MoRF (Most Relevant First) curve</b>. 
</p>

<p align="justify">
  We then measure explanation quality, by computing <b>AOPC</b>, i.e. the <b>Area Over the MoRF Curve</b>, averaged over the entire test set. Conversely to what is expected with the MoRF curve, the AOPC curve should increase, since we are accumulating bigger and bigger areas (the steepest the decrease in the MoRF curve, the greater the area over it).
</p>

![AOPC](https://latex.codecogs.com/gif.latex?AOPC%20%3D%20%5Cfrac%7B1%7D%7BL%20&plus;%201%7D%5Cbigg%5Clangle%5Csum%5Climits_%7Bk%3D0%7D%5E%7BL%7D%20f%28x_%7BMoRF%7D%5E%7B%280%29%7D%29%20-%20f%28x_%7BMoRF%7D%5E%7B%28k%29%7D%29%5Cbigg%5Crangle_%7Bp%28x%29%7D)

<p align="justify">
  To compare between our unsupervised and hybrid approaches, we propose another evaluation metric, <b>POMPOM (Percentage of Meaningful Pixels Outside the Mask)</b>. For a single instance, it is defined as the number of meaningful pixels outside the region of interest (as given by a "ground-truth" weak annotation mask) in relation to its total number of pixels. We only consider meaningful pixels, i.e pixel values higher than ε. ε is a small factor (∼ 1e−6) to ensure numerical stability (avoid divisions by 0). Therefore, POMPOM admits values in [0, 1], going from black to white images, respectively. <b>The lower the POMPOM value</b> for the entire test set, <b>the better</b> the produced explanations, since they have less pixels outside our regions of interest.
</p>

![POMPOM](https://latex.codecogs.com/gif.latex?POMPOM%20%3D%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%20%7D%5B%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%5D%20%3E%20%5Cepsilon%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%20%7D%5B%281-z_%7Bi%2Cj%7D%29%5D%20%3E%20%5Cepsilon%20&plus;%20%5Cepsilon%7D%5Cright%7C)


## Implementation

### PyTorch
<ul>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/ConvMod.py">ConvMod.py</a> - contains the ConvMod class (used in both <a href="https://github.com/icrto/xML/blob/master/PyTorch/Explainer.py">Explainer.py</a> and <a href="https://github.com/icrto/xML/blob/master/PyTorch/VGG.py">VGG.py</a>), that implements a conv-relu-conv-relu module.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/Dataset.py">Dataset.py</a> - provides the Dataset class, necessary to create a PyTorch <code>dataloader</code>. The <code>__getitem__</code> function is responsible for randomly sampling an image from the corresponding pandas dataframe (previously obtained with the <code>load_data</code> function), with its respective label and mask. The mask is only provided if the variable <code>masks</code> is set to <code>True</code>, which might only be needed in case one wants to apply the hybrid explanation loss.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/Explainer.py">Explainer.py</a> - defines the Explainer's layers (see paper and/or source code for a more detailed description of the module).</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/ExplainerClassifierCNN.py">ExplainerClassifierCNN.py</a> - contains the implementation of the joint architecture. It instantiates the explainer and a classifier (chosen from a list of modified resnet models or our VGG-based implementation). In this file the methods <code>train</code>, <code>validation</code>, <code>test</code> and <code>save_explanations</code> are implemented. The <code>train</code> method is responsible for one epoch of training: it starts by freezing/unfreezing each module according to the current training phase (as described in the <a href="https://github.com/icrto/xML#Training">Training</a> section), then performs the forward and backward passes, computing the loss and updating the network's parameters accordingly. The <code>validation</code> and <code>test</code> methods are similar, performing a forward pass in eval mode and computing several evaluation metrics for both classification and explanation parts. Finally, the <code>save_explanations</code> method plots and saves the produced explanations alongside their respective original input images. This file also includes the method <code>checkpoint</code>, responsible for saving checkpoints of the model during training.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/losses.py">losses.py</a> - contains the implementations for the unsupervised and hybrid explanation losses, both for a single instance and for a mini-batch with different <code>reduction</code> modes.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/ResNetMod.py">ResNetMod.py</a> - defines one alternative for the Classifier architecture as a modified version of the ResNet network (see paper and/or source code for a more detailed description of the module). The original ResNet source code (from <code>torchvision</code>) is modified to include the multiplication layers and connections between explainer and classifier. These layers are introduced after the first layers and after each super-block.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/test.py">test.py</a> - allows the user to test a trained model given by a previously saved PyTorch checkpoint (<code>.pt</code> file). The script starts by validating the parameters, loading the model and creating the <code>test dataloader</code>. Having done this, we simply need to call the function <code>test</code> and save our results to a <code>.txt</code> file, as well as save the produced explanations and the roc and precision-recall curves.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/train.py">train.py</a> - this is where the magic happens. As usual, we start by validating and processing our input arguments, creating our model and loading our data into PyTorch <code>dataloaders</code>. Then, we define a <code>for</code> loop for our 3 training phases. For each phase we define different optimisers and learning rate schedulers, as well as different <code>csv</code> files to save our training history. Each phase has a certain number of epochs, so for each epoch we call our <code>train</code> method to actually train our networks and then we call <code>validation</code> for both training and validation sets. Validation is done for our training set as well, although we could just save the training statistics computed in the <code>train</code> method, because we want to average the loss and metrics over the entire training set having our network stable during the process. If we used the statistics from the training step either we would only record these values for our last training mini-batch (i.e. using the most recent "version" of our network) or we would average the results for all the mini-batches (which is not 100% correct since the first mini-batch results would have been obtained from a less trained network than the last mini-batch results obtained from the most recently updated version of our network). After this validation process, we just checkpoint our model and save the results, while also evaluating if it's time to stop training according to an Early Stopping policy. At the end of each phase we also plot the loss and some metrics obtained during our training epochs, as well as the produced explanations.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/utils.py">utils.py</a> - contains auxiliary functions, such as image normalisation, freezing and unfreezing of model layers and plotting functions (for plotting metrics and losses' values during training/validation and roc/precision-recall curves).</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/VGG.py">VGG.py</a> - defines one alternative for the Classifier architecture as a version of the VGG-16 network (see paper and/or source code for a more detailed description of the module). The original VGG-16 is implemented and modified to include the multiplication layers and connections between explainer and classifier. These layers are introduced after each <code>conv-relu-conv-relu</code> stage and before <code>pooling</code>, as shown in the <a href="https://github.com/icrto/xML#Architecture">Architecture</a> section.</p></li>
</ul>

### Keras
<ul>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/dataset.py">dataset.py</a> - provides the DataGenerator class, necessary to create a Keras <code>sequence</code>. The <code>__getitem__</code> function is responsible for randomly sampling an image from the corresponding pandas dataframe (previously obtained with the <code>load_data</code> function), with its respective label and mask. The mask is only provided if the variable <code>masks</code> is set to <code>True</code>, which might only be needed in case one wants to apply the hybrid explanation loss.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/Explainer.py">Explainer.py</a> - defines the Explainer's layers (see paper and/or source code for a more detailed description of the module).</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/PyTorch/ExplainerClassifierCNN.py">ExplainerClassifierCNN.py</a> - contains the implementation of the joint architecture. It instantiates the explainer and a classifier (either a modified ResNet50 model or our VGG-based implementation). In this file the methods <code>build_model</code>, <code>save_architecture</code> and <code>save_explanations</code> are implemented. The <code>build_model</code> method defines the <code>forward pass</code> and the model's inputs and outputs, while the <code>save_architecture</code> method plots the architecture of each module. Finally, the <code>save_explanations</code> method plots and saves the produced explanations alongside their respective original input images.
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/losses.py">losses.py</a> - contains the implementations for the unsupervised and hybrid explanation losses.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/ResNet50Mod.py">ResNet50Mod.py</a> - defines one alternative for the Classifier architecture as a modified version of the ResNet50 network (see paper and/or source code for a more detailed description of the module). The original ResNet50 source code (from <code>tensorflow.keras</code>) is modified to include the multiplication layers and connections between explainer and classifier. These layers are introduced after the first layers and after each super-block.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/test.py">test.py</a> - allows the user to test a trained model given by a previously saved Keras checkpoint (<code>.h5</code> file). The script starts by validating the parameters, loading the model and creating the <code>test data generator</code>. Having done this, we simply need to call the functions <code>evaluate</code> and <code>predict</code>  and save our results to a <code>.txt</code> file, as well as save the produced explanations and the roc and precision-recall curves.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/train.py">train.py</a> - this is where the magic happens. As usual, we start by validating and processing our input arguments, creating our model and loading our data into Keras <code>data generator</code>. Then, we define a <code>for</code> loop for our 3 training phases. For each phase we define different optimisers and learning rate schedulers, as well as different <code>csv</code> files to save our training history. Then, we just call the method <code>fit</code> to perform training and validation for a defined number of epochs. During this process, we checkpoint our model and save the results, while also evaluating if it's time to stop training according to an Early Stopping callback. At the end of each phase we also plot the loss and some metrics obtained during our training epochs, as well as the produced explanations.</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/utils.py">utils.py</a> - contains auxiliary functions, such as image normalisation, freezing and unfreezing of model layers and plotting functions (for plotting metrics and losses' values during training/validation and roc/precision-recall curves).</p></li>
  <li><p align="justify"><a href="https://github.com/icrto/xML/blob/master/Keras/VGG.py">VGG.py</a> - defines one alternative for the Classifier architecture as a version of the VGG-16 network (see paper and/or source code for a more detailed description of the module). The original VGG-16 is implemented and modified to include the multiplication layers and connections between explainer and classifier. These layers are introduced after each <code>conv-relu-conv-relu</code> stage and before <code>pooling</code>, as shown in the <a href="https://github.com/icrto/xML#Architecture">Architecture</a> section.</p></li>
</ul>
  
## Training

<p align="justify">
  Training this architecture involves 3 phases:
</p>
<ol>
  <li><p align="justify">Only the <b>Classifier is trained</b>, while the <b>Explainer remains frozen</b>. Note that the Explainer is initialised so that the <b>initial explanations consist of white images</b> (matrices filled with 1s). We achieve this by imposing a large bias ( > 1) in the Explainer’s batch normalisation layer. This bias is controlled by the hyperparameter <code>init_bias</code>. Doing this ensures that the connections between Classifier and Explainer are bypassed. So, the Classifier is not taking the Explainer’s output into account yet, because the Explainer has not been trained at this point. The intuition is that we start by considering the whole image as an explanation (at the beginning every pixel/region is considered as relevant for the decision) and gradually eliminate irrelevant regions as redundancy is decreased (see the GIFs below).</p></li>
  <li><p align="justify">The process is reversed: the <b>Classifier remains frozen</b>, but the <b>Explainer learns</b> by altering its explanations and assessing the corresponding impact on the classification. This happens because the joint classification and explanation loss affects both modules (but only the Explainer is updated accordingly in this phase). In practice, the <b>Explainer is indirectly trained with the supervision of the classification component</b>.</p></li>
  <li><p align="justify">Finally the <b>whole architecture is fine-tuned end-to-end</b>. The Classifier learns with the new information provided by the Explainer, refining its classification outputs. At the same time, the Explainer continues adapting its explanations to the concepts the Classifier is learning and considers important for classifying the images.</p></li>
</ol>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/unsupervised_phase1.gif">
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/unsupervised_phase2.gif">
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/hybrid_phase1.gif">
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/hybrid_phase2.gif">
</p>

<b>IMPORTANT REMARKS</b>
<p align="justify">
  It is imperative that <b>at the end of phase 1 the Classifier remains somewhat unstable</b>, i.e. that its loss does not plateau, so that in phase 3 both modules can learn from each other. Otherwise, in phase 3 the Classifier would not update its parameters with the new information provided by the now trained Explainer and vice-versa. Therefore, in the end, both Explainer and Classifier improve, fruit of this dynamic interaction between the two and their respective loss functions.
</p>

<b>USEFUL TIPS & TRICKS</b>

<p align="justify">
  As previously mentioned, we use different values for the <b>α hyperparameter</b> during each of the training phases: during <b>phases 1 and 3, α > 0.5</b> (closer to 1.0), because the correct classification is an indispensable part of the system; it directly affects the Classifier and indirectly affects the Explainer. Conversely, during <b>phase 2, α < 0.5</b> (closer to 0), so as to give more strength to the Explainer, since it is the only module being trained.
</p>
  
<p align="justify">
  The <b>β</b> and <b>γ hyperparameters</b> are <b>kept constant</b> during all training phases.
</p>

<p align="justify">
  For the last training phase, we started by tuning α, keeping β constant at 1. We first performed a coarse grained search in the range {0.5, 0.75, 0.9} and found out that a value around 0.9 produced the most reasonable explanations, while maintaining the predictive performance. The β hyperparameter was tuned by searching within a broader range, [0.0, 1.0], and afterwards within a smaller set, {0.75, 0.8, 0.9, 0.99}. Finally, in the hybrid scenario, γ was tuned by searching in {10.0, 1.0, 1e − 2, 1e − 4} while keeping the best β value.
 </p>

<p align="justify">
  Fine-tuning the number of epochs for each phase can be a tricky process, involving several experiments until one gains the necessary intuition for the specific data one is working with. To aliviate this problem, we present here the intuition we gained during the development of this work (however, keep in mind that this applied to the datasets we used, but <b>might not hold true for every dataset out there</b>). When starting exploring a new dataset, the <b>first thing</b> we did was <b>train only the classifier</b> to see which performance it could reach and if the classification network was adequate to our problem. We usually did this for many epochs (let's say around 100) and tried out different batch sizes, learning rates, optimisers and schedulers. After finding the best configuration that's usually the one we sticked with for every training phase.
</p>

<p align="justify">
  Afterwards, and following the important aforementioned remark, as a rule of thumb, we would obtain the number of epochs for our first training phase by "cutting in half" the number of epochs needed to converge our classifier. For example, if it took us 80 epochs to reach a good classification performance, we would define 20 as the number of epochs for our first training phase. This can be a first estimate, but we can actually exaggerate a bit further and cut this number even more (in the paper we used 10 epochs). As another rule of thumb, what we did was look at our <b>classification loss</b> and see where its value had <b>decreased to around 25% of its initial value</b>. So, imagine the loss started at 0.9, and at epoch 12 it reached 0.2, then we would define 10 as the number of epochs for our first training phase.
</p>

<p align="justify">
  Having chosen the number of epochs for our first training phase, we can move on to the second. Here we did something similar, and trained our network in this phase (always after training phase 1 for the number of epochs defined and explained before) for a great number of epochs to ensure that the <b>Explainer</b> was able to <b>reach a loss of 0</b>. After achieving this, we would choose the number of epochs as the one where the <b>loss had decreased considerably</b> (more than 80% of the initial value - 0.2 when the loss started at about 1.0), <b>but was not 0</b> (we found out that letting the Explainer reach a loss of 0 would usually render it incapable of leaving this local minima and learning anything in the last training phase).
</p>

<p align="justify">
  Finally, we would train the last phase for a considerable number of epochs (usually 100) and let <b>Early Stopping</b> tell us where to stop training. Here we found a bit of a <b>trade-off between accuracy and explanation quality</b>; lowering the classification loss usually meant increasing the explanation loss a bit. However, since our main focus is still the classification component (remember we use α > 0.5 in this phase, usually closer to 1.0) we found this aspect to be negligible, especially considering that in the second training phase the explanation loss was close to 0, so it naturally needs to increase a bit to ensure proper classification of the images (we don't want every feature map to be multiplied by zeros).
</p>

## Results
<p align="justify">
For results on synthetic data check <a href="https://github.com/icrto/xML/tree/master/Synthetic%20Dataset">this</a>.
</p>

### ImagenetHVZ

<p align="justify">
  As the name implies, this dataset is a sub-set of <a href="http://www.image-net.org">ImageNet</a>, composed only by horses and zebras (synsets n02389026 and n02391049, respectively). Of the 2600 images, we kept only the ones for which bounding boxes were available, totalling 324 images of horses and 345 images of zebras. Data was split into 85%-15% for training and testing, giving a total of 100 images for the latter. The training set was further split into 80%-20% for training and validation. You can find this dataset <a href="">here</a>.
 </p>

<p align="justify">
  Taking into account the tips & tricks previously mentioned, here is the summary of the hyperparameter values used when conducting experiments on the imagenetHVZ dataset.
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/tableimagenetHVZ.png" width=500>
</p>

<p align="justify">
  Below is shown a comparison between the proposed approaches and some state-of-the-art post-model interpretability methods, as well as a Canny edge detector and an Otsu segmentation. Each explainability method (column) is accompanied by the respective POMPOM value for entire the test set, and, for each image (row), we also compute the class confidence score (we show two images of zebras and two images of horses, respectively). The colourmap ranges from purple to yellow (between [0, 1]) and absolute pixel values were considered for all methods for a fair comparison.
</p>

<p align="justify">
 It can be seen that the explanations produced by our network are image- and class-dependent, and highlight in a clearer way the relevant image regions. Keeping in mind that the dataset is only composed by images of zebras and horses, one of the main properties that distinguishes both animals is the texture of their fur, which is clearly highlighted by the proposed approaches. It is also worth noting that horses are identifiable via their riding equipment (5th row), and their legs and hooves (6th row). As expected, explanations produced by the hybrid approach present less active pixels outside the region of interest when compared to the ones produced by the unsupervised approach.
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_explanations.png" width=1420 height=400>
</p>

<p align="justify">
  To quantitatively evaluate and compare our explanations, we used the method briefly explained in the <a href="https://github.com/icrto/xML#Metrics">Metrics</a> section, and present our results both in terms of average function value and AOPC. We tested different grid configurations, i.e. several patch sizes (shown in each row of the figures presented below) to assess the impact of the perturbation in relation to kernel size, as well as several perturbation strategies (different columns of the figures) to guarantee independence from the model and from the average image background. As stated in the original paper by Samek at al. (<a href="https://arxiv.org/pdf/1509.06321.pdf">Evaluating the visualization of what a
  Deep Neural Network has learned</a>), "We consider a region highly relevant if replacing the information in this region in arbitrary ways reduces the prediction score of the classifier; we do not want to restrict the analysis to highly specialized information removal schemes.". These perturbation/occlusion/removal strategies consisted in replacing every pixel of each patch (from the most to the least relevant) by either a black, white or gray pixel/patch, by applying a gaussian blur to the original patch, or by replacing each pixel by an RGB value sampled from a uniform distribution.
 </p>
 
 <p align="justify">
  As explained by Montavon et al. in <a href="https://www.sciencedirect.com/science/article/pii/S1051200417302385">Methods for interpreting and understanding deep neural networks</a>, we usually choose to remove/occlude a patch instead of a single pixel so that we remove actual content of the image, while also avoiding introducing pixel artifacts (spurious structures). However, it is still important to keep in mind that the result of the analysis depends to some extent on the perturbation/occlusion/feature removal process. Various perturbation strategies can be used, but it should keep as much as possible the region being modified on the data manifold; this guarantees that the classification model continues to work reliably through the whole perturbation process.
</p>
 
 <p align="justify">
  As expected, the average function value decreases and AOPC increases with the number of patches removed. Our proposed approaches consistently present higher AOPC values throughout the experiments, which means that they better identify the relevant pixels in the images. Furthermore, the hybrid approach slightly outperforms the unsupervised approach, and both clearly outperform state-of-the-art methods.
</p>

<p align="justify">
  Relatively to patch size, we can observe that with a patch size much larger than the kernel size (16 × 16 and 32 × 32), AOPC scores are higher, as we are obtaining a region score closer to the filter score, which means that perturbing that region largely impacts the filter response and, consequently, lowers the classification performance. Conversely, perturbing smaller regions (4 × 4 and 8 × 8) has less impact on the filter response, leading to smaller AOPC values. Furthermore, independently of the patch size, our methods present a consistent behaviour, which cannot be said of the other interpretability methods, suggesting that our approach in fact produces relevant explanations, while some of the state-of-the-art methods sometimes produce counter intuitive results (for example confront the Deconvolution method with a patch size of 4x4 with black, gray, uniform and white perturbation strategies).
</p>

<p align="justify">
  Regarding the perturbation/occlusion/feature removal strategy, gaussian blurring is the one that produces the most similar results throughout the experiments, because it is the only approach that retains local information and keeps each modified data point closer to the original. White and gray perturbation strategies perform the worst, as they are the ones that alter the most each pixel value (moving the corrupted image outside the data manifold). Finally, between these two sets we have the uniform and black perturbation strategies.
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_avgfunction.png">
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_aopc.png">
</p>

### NIH-NCI Cervical Cancer

<p align="justify">
  We also used a medical dataset, the NIH-NCI Cervical Cancer Dataset. We used 2120 images (all with bounding boxes of the cervix region available). We divided the data as follows: instances with no histology done, normal or less than CIN2 (Cervical Intraepithelial Neoplasia) are considered as not cancerous; the rest are considered cancerous. This dataset was split into 95%-5% for training and testing (106 images for testing), and the training set was further divided into 80%-20% for training and validation.
</p>

<p align="justify">
  The following table presents the hyperparameters used in the experiments done with this dataset.
</p>

<p align="center">
<img src="https://github.com/icrto/xML/blob/master/example_images/tableNIH.png" width=500>
</p>

<p align="justify">
  We also present some of the produced explanations for this dataset. Once more, the proposed approaches highlight relevant regions in a clearer way. However, sometimes non-interest regions are highlighted as well, such as specular reflections. This, together with the fact that every other interpretability method does not perform that well, leads us to believe that this particular classification model is not the best for this task, since we used a standard ResNet-18 for such a specific medical imaging problem without any custom pre-processing steps, such as reflection and/or hand/glove removal, from which we believe the classification model would certainly benefit a lot and, consequently, improve the quality of the produced explanations.
</p>
  
<p align="center">
<img width=1420 height=400 src="https://github.com/icrto/xML/blob/master/example_images/cervix_grid_explanations.png">
</p>

## Requirements

For the PyTorch version please consult [requirements_pytorch.txt](https://github.com/icrto/xML/blob/master/requirements_pytorch.txt). 
For the Keras version please consult [requirements_keras.txt](https://github.com/icrto/xML/blob/master/requirements_keras.txt). 

We used `PyTorch 1.3.1` and `tensorflow.keras 2.4.0` (`tensorflow 2.3.0`) in `Python 3.6.9`.

## Usage

<p align="justify">
  Want to use this code? You are in the right place! Welcome!
</p>

<p align="justify">
  In the <a href="https://github.com/icrto/xML/blob/master/PyTorch/Dataset.py">Dataset.py</a> (PyTorch version) or  <a href="https://github.com/icrto/xML/blob/master/Keras/dataset.py">dataset.py</a> (Keras version) files you can find the function <code>load_data</code> implemented, as well as individual methods to process and load our synthetic dataset, the imagenetHVZ dataset and the NIH-NCI Cervical Cancer Dataset.
</p>

<p align="justify">
  For the <b>synthetic dataset</b>, you just need to download the <a href="https://github.com/icrto/xML/blob/master/Synthetic%20Dataset/data.zip">data</a> and point the variable <code>dataset_path</code> in <a href="https://github.com/icrto/xML/blob/master/PyTorch/train.py">train.py</a> (PyTorch version) or <a href="https://github.com/icrto/xML/blob/master/Keras/train.py">train.py</a> (Keras version) to the directory where you stored it.
</p>

<p align="justify">
  The same applies for the <b>imagenetHVZ</b> dataset. Download it <a href="https://drive.google.com/file/d/1K8tZPP5uYNwHw6-DIHXBdD8aIq234yaK/view?usp=sharing">here</a> and point the variable <code>dataset_path</code> in <a href="https://github.com/icrto/xML/blob/master/PyTorch/train.py">train.py</a> (PyTorch version) or <a href="https://github.com/icrto/xML/blob/master/Keras/train.py">train.py</a> (Keras version) to the directory where you stored it.
</p>

<p align="justify">
  For the <b>NIH-NCI Cervical Cancer dataset</b>, after downloading the data, your directory structure should look like this:
</p>

```
data
└───ALTS
│   │   covariate_data.xls
│   │   imageID1.jpg
|   |   imageID1_mask.jpg
│   │   imageID2.jpg
|   |   imageID2_mask.jpg
│   │   ... 
└───Biopsy
│   │   covariate_data.xls
│   │   imageID1.jpg
|   |   imageID1_mask.jpg
│   │   imageID2.jpg
|   |   imageID2_mask.jpg
│   │   ... 
└───CVT
│   │   covariate_data.xls
│   │   imageID1.jpg
|   |   imageID1_mask.jpg
│   │   imageID2.jpg
|   |   imageID2_mask.jpg
│   │   ... 
└───NHS
│   │   covariate_data.xls
│   │   imageID1.jpg
|   |   imageID1_mask.jpg
│   │   imageID2.jpg
|   |   imageID2_mask.jpg
│   │   ... 
```

<p align="justify">
  Note that for every one of these datasets or other ones you wish to explore, you can always reimplement the corresponding <code>load_data_dataset_name</code> method and add it to the <code>load_data</code> function. However, <b>keep in mind that the <code>__getitem__</code> function assumes that you have a <code>dataframe</code> containing the absolute path to every image and corresponding mask (only needed for the hybrid explanation loss), as well as its label</b>. Also, the <code>train.py</code> and <code>test.py</code> files expect you to first call <code>load_data</code> to generate your training, validation and test dataframes, and only then create your <code>PyTorch datasets</code> and <code>dataloaders</code> (PyTorch version)/<code>data generators</code> (Keras version) with these dataframes.
</p>

<p align="justify">
  After having done this, be sure to include the name of your dataset in the <code>dataset</code> argument of the <code>train.py</code> and <code>test.py</code> files.
</p>

<p align="justify">
  You finally have everyhting ready to <b>train</b> the network! Do this by calling <code>train.py</code> or by giving other arguments if you do not wish to use their default values, such as in the example below. 
</p>

<p align="justify">
  The rest is already taken care of! The scripts will save the produced explanations at the end of each phase, save plots of the evaluation metrics and save the model that achieved the best validation loss.
</p>

```
python3 train.py --dataset imagenetHVZ --dataset_path <path_to_dataset> --nr_epochs 10,10,60 -bs 8 --init_bias 3.0 --loss hybrid --alpha 1.0,0.25,0.9 --beta 0.9 --gamma 1.0 -lr_clf 0.01,0,0.01 -lr_expl 0,0.01,0.01 --aug_prob 0.2 --opt sgd -clf resnet18 --early_patience 100,100,10 --folder <path_to_destination_folder>
```
<p align="justify">
  To <b>test</b> your model just run <code>test.py</code> including the path to your model (a <code>.pt</code> file in the PyTorch version and a <code>.h5</code> file in the Keras version). The script will save the produced explanations and a <code>.txt</code> file with the report of the obtained values in the same directory where your model is stored. If you wish to define other parameter values, follow the example below.
 </p>

```
python3 test.py <path_to_model> --dataset imagenetHVZ --dataset_path <path_to_dataset> --alpha 0.9 -bs 8 -clf resnet18 --init_bias 3.0 --loss hybrid --beta 0.9 --gamma 1.0
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


## Contact Info

<p align="justify">
If you wish to know more about this project or have any questions/suggestions, do not hesitate to contact me at <a href="mailto:icrtto@gmail.com">icrtto@gmail.com</a> or to open an issue/pull request in the repo.
</p>
