# [ICSE 2025] Diversity Drives Fairness: Ensemble of Higher Order Mutants for Intersectional Fairness of Machine Learning Software

Welcome to visit the homepage of our ICSE'25 paper "Diversity Drives Fairness: Ensemble of Higher Order Mutants for Intersectional Fairness of Machine Learning Software". The homepage contains data, scrips, and intermediate results used in this paper.

## Experimental environment

We use Python 3.7 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit to implement bias mitigation methods and compute fairness metrics. 

Installation instructions for Python 3.7 and AIF360 can be found at https://github.com/Trusted-AI/AIF360. That page provides several ways for the installation. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you want to try other installation ways or encounter any errors during the installation process, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name aif360 python=3.7
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install 'aif360'
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. Note that TensorFlow is only required for implementing the existing fairness improvement method named ADV. If you do not want to implement this method, you can skip the installation of TensorFlow (the last line of the following commands).
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
pip install --upgrade protobuf==3.20.0
pip install fairlearn
```

## Dataset

You can refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files. We use the data processing scripts provided by [previous work](https://ieeexplore.ieee.org/document/9951398) to process the datasets. We have included the data processing scripts and processed datasets in the ```Dataset``` folder.


## Scripts and results

You can reproduce all the results based on the intermediate results provided by us.

* ```RQ_code/``` contains the scripts for producing the results for all RQs. You can reproduce all the results based on the intermediate results provided by us by running ```rq1.py```, ```rq2.py```, ```rq3.py```, ```rq45.py```, ```rq6.py```, and ```rq7.py```.

* ```Results/``` contains the raw results of applying each bias mitigation method to datasets with two protected attributes. Each file in this folder has 21 columns, with the first column indicating the metric, and the next 20 columns the metric values of 20 runs.

*  ```Results_Three/``` contains the raw results of applying each bias mitigation method to datasets with three protected attributes. Each file in this folder has 21 columns, with the first column indicating the metric, and the next 20 columns the metric values of 20 runs.

You can also replicate the results from scratch using the datasets provided above and the following code of each bias mitigation method.

* ```BiasMitigation/``` contains the scripts for implementing FairHOME, its variants, and existing bias mitigation methods for datasets with two protected attributes.

* ```BiasMitigation_Three/``` contains the scripts for implementing FairHOME, its variants, and existing bias mitigation methods for datasets with three protected attributes.

* ```gerryfair/``` contains the raw code of GRY, a baseline method we adopt.

* ```Fairea_multi/``` and ```Fairea_multi_Three/``` contain the scripts of the benchmarking tool namely Fairea for datasets with two protected attributes and three protected attributes in the paper, respectively.

* ```Cal_baseline/``` and ```Cal_baseline_Three/``` contain the scripts for generating trade-off baselines using Fairea for datasets with two protected attributes and three protected attributes in the paper, respectively.

* ```Fairea_baseline_new/``` contains the generated baselines.

* ```aif360.zip``` contains the scripts (provided by [Chen et al.](https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness) at ICSE 2024) of adapting REW, ADV, and EOP to make them applicable to multiple protected attributes.

## Citation
Please consider citing the following paper when using our code or data.
```
@inproceedings{zpicse2025,
  title={Diversity Drives Fairness: Ensemble of Higher Order Mutants for Intersectional Fairness of Machine Learning Software},
  author={Zhenpeng Chen and Xinyue Li and Jie M. Zhang and Federica Sarro and Yang Liu},
  booktitle={Proceedings of the 47th IEEE/ACM International Conference on Software Engineering, ICSE 2025},
  year={2025}
}
```
