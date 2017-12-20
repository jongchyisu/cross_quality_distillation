# Cross Quality Distillation #

## Introduction ##
This repository contains the code for reproducing the results in "Adapting Models to Signal Degradation using Distillation", BMVC, 2017
(Originally with the title "Cross Quality Distillation" on [arXiv](https://arxiv.org/abs/1604.00433))

    @inproceedings{su2017adapting,
        Author    = {Jong-Chyi Su and Subhransu Maji},
        Title     = {Adapting Models to Signal Degradation using Distillation},
        Booktitle = {British Machine Vision Conference (BMVC)},
        Year      = {2017}
    }

Code is tested on Ubuntu 14.04 with MATLAB R2014b and MatConvNet package.  
Link to the [project page](http://people.cs.umass.edu/~jcsu/papers/cqd).  
Code is borrowed heavily from B-CNN (https://bitbucket.org/tsungyu/bcnn).  

## Instruction ##
1. Follow instructions on [VLFEAT](http://www.vlfeat.org) and [MatConvNet](http://www.vlfeat.org/matconvnet) project pages to install them first. Our code is built on MatConvNet version `1.0-beta18`.
2. Change the path in `setup.m`
3. Download datasets
    * Birds: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
    * Cars: [Stanford cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
4. Run `save_images.m` to create degraded images (Need to download and install [Structured Edge Detector](https://github.com/pdollar/edges) for generating edge images and [tpsWarp](https://www.mathworks.com/matlabcentral/fileexchange/24315-warping-using-thin-plate-splines?focused=5117319&tab=function) for generating distorted images)
5. Download and put pre-trained vgg models under `data/models/` (vgg-m and vgg-vd are used in the paper)
6. Run `run_CQD.m` for training all the baseline models and distillation model

## Results ##
Please see Table 1 in the paper.

## Acknowledgement ##
Thanks Tsung-Yu Lin for sharing the codebase and MatConvNet team.  
Please contact jcsu@cs.umass.edu if you have any question.  
