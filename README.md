# Understanding the decisions of CNNs: an in-model approach
<p align="justify">
This is the official implementation of the paper [Understanding the decisions of CNNs: an in-model approach](https://doi.org/10.1016/j.patrec.2020.04.004). 

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
  With the advent of Deep Learning, in particular Convolutional Neural Networks, there is also a growing demand for Interpretability/Explainability of these highly complex and abstract models. Several interpretability methods have already been proposed, and these are for the most part post-model methods, i.e. methods applied after the main (classification) model is trained.

However, we argue that interpretability should be taken into account from the start, being a design requirement and a desirable property of the system. As such, we propose an in-model approach, i.e. an interpretability method applied during training of the main model.  
</p>

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/architecture.png">

## Loss
![JointLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D%20%3D%20%5Calpha%20%5Cmathcal%7BL%7D_%7Bclass%7D%20&plus;%20%281%20-%20%5Calpha%29%20%5Cmathcal%7BL%7D_%7Bexpl%7D)

### Unsupervised Explanation Loss
![UnsupervisedLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20%3D%20%5Cbeta%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D_i%29%20&plus;%20%281%20-%20%5Cbeta%29%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D_i%29)

![Sparsity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%20%7C%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20%5Clabel%7Beq%3Asparsity%7D)

![Contiguity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%7C%5Chat%7Bz%7D_%7Bi&plus;1%2Cj%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20&plus;%20%7C%5Chat%7Bz%7D_%7Bi%2Cj&plus;1%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C)

### Hybrid Explanation Loss
![HybridLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_hybrid%7D%20%3D%20%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20&plus;%20%5Cgamma%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%7D%5Cright%7C)

## Metrics
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
