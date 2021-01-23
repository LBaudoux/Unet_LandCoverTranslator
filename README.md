# Land cover translation with an assymetrical-Unet

This repo is the official implementation of the paper "Toward a yearly country-scale CORINE Land-Cover map without using images: a map translation approach". 

### Table of Contents: 
- [Land cover translation with an assymetrical-Unet](#land-cover-translation-with-an-assymetrical-unet)
    - [Reference](#reference)
    - [Install](#install)
    - [Dataset](#dataset)
    - [Tutorial](#tutorial)
    - [Acknowledgement](#future-work)
    - [License](#license)
    
### Reference:



### Install

For conda users :

```
conda env create -f environment.yml
```

 For pip users : open with a text editor the environment.yml and install manually the packages.

### Dataset 

Datasets can be downloaded from the following [zenodo archive](https://lien_vers_zenodo). 

The folder holds two dataset : 

- the first one is the dataset presented in the paper (oso-to-clc) 
- the second one contains partially self-covering-patches (oso-to-clc-self-covering) in order to make possible to create a France wide map without patches-edge effects. Thus it should not be used in a training procedure. Example of result can be observe on this [website]().

Folder structure is presented below :

```
├── oso_to_clc
|  └── train
|  |  |  └── oso_2018
|  |  |  └── oso_2016
|  |  |  └── clc _2018
|  |  |  └── clc_2012
|  └── test
|  └── val
|  └── metadata.json
|
├── oso_to_clc_self_covering
|  └── full
|  |  |  └── oso_2018
|  |  |  └── oso_2016
|  |  |  └── clc _2018
|  |  |  └── clc_2012
|  └── metadata.json
|
├── ground_truth.gpkg
```

Useful consideration :

- The manually build ground_truth used in this study is made available in a geopackage fromat. 
- The metadata.json holds the localisation of each patch.
- Each patch is encoded in a numpy binary format (.npy)

### Tutorial:
This github repository follow the structure proposed [Hager Rady](https://github.com/hagerrady13/) and [Mo'men AbdelRazek](https://github.com/moemen95). For more information on the repository structure, and how to handle launch of the code refers to this [github repo](https://github.com/moemen95/Pytorch-Project-Template).

Once the python environnement is correctly set. Download the dataset and place it in the data folder. Then launch the run.sh script for a simple default train with default features.

Algorithm parameters are stored in the configs/ folder. By default parameters are set to run on computer with low ram and GPU capacity. On a powerful machine we recommend to adapt the following parameters :

- full test : False -> True . By default statistic in test phase are only computed on 6022 ground truth. If set to true, comparison between prediction and CLC will also be computed (agreement between prediction and CLC 2018, EPI). Note that this is option should only be set to true with at least 40 Go RAM.
- train_batch_size,test_batch_size,valid_batch_size : 4 ->n. batch size can be adapted depending on the capacity of your computer. Note that results presented in the paper are computed with batch size = 24.

### Acknowledgement

* This is a joint work between [Hager Rady](https://github.com/hagerrady13/) and [Mo'men AbdelRazek](https://github.com/moemen95)


### License:
This project is licensed under MIT License - see the LICENSE file for details
