# HINT
Code for the research paper "HINT: Hierarchical Invertible Neural Transport for Density Estimation and Bayesian Inference".
Pre-print available at [arXiv](https://arxiv.org/abs/1905.10687).

Please cite as:
```
@misc{kruse2019hint,
    title = {HINT: Hierarchical Invertible Neural Transport for Density Estimation and Bayesian Inference},
    author = {Jakob Kruse and Gianluca Detommaso and Robert Scheichl and Ullrich Köthe},
    year = {2019},
    eprint = {1905.10687},
    archivePrefix = {arXiv}
}
```


### Requirements

In order to run the code, you will need the following:

+ `PyTorch` (>= v1.0.0)
+ `Python` (>= v3.7)
  + Packages `numpy`, `scipy`, `shapely` & `visdom`
+ [`FrEIA`](https://github.com/VLL-HD/FrEIA/)


### Structure

There is one script for [training unconditional models](../master/train_unconditional.py) and one for [training conditional models](../master/train_conditional.py).
These scripts expect a running `visdom` server for [visualization](../master/monitoring.py) and the import of a config file specifying the model and hyperparameters.
Config files for all models used in the paper are supplied in the directory [configs](../master/configs).

The recursive affine coupling block is implemented as a `FrEIA`-compatible module in [hint.py](../master/hint.py) and will be officially added to the framework in the future.

Data sets for training and evaluation are created with [an additional script](../master/data.py), but note that we maintain [another repository](https://github.com/VLL-HD/inn_toy_data) with the latest versions of all our toy data sets. The script also contains code for the plots used in our paper.

The [last script file](../master/rejection_sampling.py) deals with the rejection sampling baseline and systematic comparisons between the trained models. The first run of this script will take some time as the baseline is very inefficient, but results are stored and subsequent runs go much faster.
