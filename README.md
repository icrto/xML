# Understanding the decisions of CNNs: an in-model approach

This is the official implementation of the paper [Understanding the decisions of CNNs: an in-model approach](https://doi.org/10.1016/j.patrec.2020.04.004). It also includes my master thesis "Producing Decisions and Explanations: A Joint Approach Towards Explainable CNNs" (full document and final presentation), in which the paper was based and further extended.

If you use this repository, please cite

> I. Rio-Torto, K. Fernandes, and L. F. Teixeira, “Understanding the decisions of CNNs: An in-model approach,” Pattern Recognition Letters, vol. 133, no. C, pp. 373–380, Apr. 2020, doi: 10.1016/j.patrec.2020.04.004.

## Contents

[***Architecture***](https://github.com/icrto/xML##Architecture)

[***Joint Loss***](https://github.com/icrto/xML##JointLoss)


## Architecture
<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/architecture.png">

## Joint Loss
![JointLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D%20%3D%20%5Calpha%20%5Cmathcal%7BL%7D_%7Bclass%7D%20&plus;%20%281%20-%20%5Calpha%29%20%5Cmathcal%7BL%7D_%7Bexpl%7D)

### Unsupervised Explanation Loss
![UnsupervisedLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20%3D%20%5Cbeta%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D_i%29%20&plus;%20%281%20-%20%5Cbeta%29%20%5Csum_%7Bi%20%3D%201%7D%5E%7BN%7D%20%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D_i%29)

![Sparsity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bsparsity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%20%7C%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20%5Clabel%7Beq%3Asparsity%7D)

![Contiguity](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bcontiguity%7D%28%5Chat%7Bz%7D%29%20%3D%20%5Cfrac%7B1%7D%7Bm%20%5Ctimes%20n%7D%20%5Csum_%7Bi%2Cj%7D%5E%7B%20%7D%7C%5Chat%7Bz%7D_%7Bi&plus;1%2Cj%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C%20&plus;%20%7C%5Chat%7Bz%7D_%7Bi%2Cj&plus;1%7D%20-%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7C)

### Hybrid Explanation Loss
![HybridLoss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7Bexpl%5C_hybrid%7D%20%3D%20%5Cmathcal%7BL%7D_%7Bexpl%5C_unsup%7D%20&plus;%20%5Cgamma%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%7D%281-z_%7Bi%2Cj%7D%29%7D%5Cright%7C)

## Evaluation Metrics
![AOPC](https://latex.codecogs.com/gif.latex?AOPC%20%3D%20%5Cfrac%7B1%7D%7BL%20&plus;%201%7D%5Cbigg%5Clangle%5Csum%5Climits_%7Bk%3D0%7D%5E%7BL%7D%20f%28x_%7BMoRF%7D%5E%7B%280%29%7D%29%20-%20f%28x_%7BMoRF%7D%5E%7B%28k%29%7D%29%5Cbigg%5Crangle_%7Bp%28x%29%7D)

![POMPOM](https://latex.codecogs.com/gif.latex?POMPOM%20%3D%20%5Cleft%7C%5Cfrac%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%20%7D%5B%281-z_%7Bi%2Cj%7D%29%20%5Chat%7Bz%7D_%7Bi%2Cj%7D%5D%20%3E%20%5Cepsilon%7D%7B%5Csum%5Climits_%7Bi%2Cj%7D%5E%7B%20%7D%5B%281-z_%7Bi%2Cj%7D%29%5D%20%3E%20%5Cepsilon%20&plus;%20%5Cepsilon%7D%5Cright%7C)


## Requirements

## Implementation

## Training

### 3 Phase Process
### Hyperparameters

## Example Usage

## Previous work
  Rio-Torto I., Fernandes K., Teixeira L.F. (2019) Towards a Joint Approach to Produce Decisions and Explanations Using CNNs. In: Morales A., Fierrez J., Sánchez J., Ribeiro B. (eds) Pattern Recognition and Image Analysis. IbPRIA 2019. Lecture Notes in Computer Science, vol 11867. Springer, Cham


## Results
### ImagenetHVZ

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_explanations.png" width=1420 height=400>

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_avgfunction.png">

<img align="center" src="https://github.com/icrto/xML/blob/master/example_images/imagenetHVZ_grid_aopc.png">

### NIH-NCI Cervical Cancer
<img align="center" width=1420 height=400 src="https://github.com/icrto/xML/blob/master/example_images/cervix_grid_explanations.png">


