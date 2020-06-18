# Adversary's Utility in Backdoor Attacks for Federated Learning

## Team

* Research Engineer - Tejas Jain [@jaintj95](https://github.com/jaintj95)
* Research Scientist - Anupam Mediratta [@anupamme](https://github.com/anupamme)

## Note - This repo was created so that I could cleanup [AlphaPav's repo](https://github.com/AlphaPav/DBA) for personal use. All original work belongs to the concerned parties mentioned in the original repo. Please check Acknowledgement section below for details.

This repository contains the code for ICLR 2020 paper [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr)
___

## Installation

Create a new conda environment using the following command (remember to replace env_name with your preferred environment name)
```
conda create --name env_name
```

Install PyTorch and other missing packages
___

## Usage

### Prepare the dataset

#### LOAN dataset

- download the raw dataset [lending-club-loan-data.zip](https://www.kaggle.com/wendykan/lending-club-loan-data/) into dir `./utils`
- rename the zip to `lending-club-loan-data.zip`
- preprocess the dataset.

```
cd ./utils
python load_preprocess.py
```

Note: If the above command fails, comment out the following snippet from loan_preprocess.py

```python
with zipfile.ZipFile(source_dir, 'r') as zip_ref:
    if not os.path.exists("../data"):
        os.mkdir("../data")
    zip_ref.extractall("../data")
```

and then execute the `process_loan_data.sh` script in utils directory
```
cd ./utils
./process_loan_data.sh
```

#### Tiny-imagenet dataset

- download the dataset [tiny-imagenet-200.zip](https://tiny-imagenet.herokuapp.com/) into dir `./utils`
- preprocess the dataset

```
cd ./utils
python tinyimagenet_reformat.py
```

Note: If the above command fails, comment out the following snippet from tinyimagenet_reformat.py

```python
with zipfile.ZipFile(source_dir, 'r') as zip_ref:
    if not os.path.exists("../data"):
        os.mkdir("../data")
    zip_ref.extractall("../data")
```

and then execute the `process_tiny_data.sh` script in utils directory

#### Others

MNIST and CIFAR will be automatically download
___

## Run experiments

* prepare the pretrained model:
Our pretrained clean models for attack can be downloaded from [Google Drive](https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing). You can also train from the round 0 to obtain the pretrained clean model.

* we can use Visdom to monitor the training progress

```
python -m visdom.server
```

Note - If the default port 8097 is already in use, you can start visdom a custom port like so:

```
python -m visdom.server -p 8098
```

Remember to change the port number (`VIS_PORT`) in config.py  
  
* run experiments for the four datasets

```
python main.py -p utils/X.yaml
```

or

```
python main.py --params utils/X.yaml
```

`X` = `mnist_params`, `cifar_params`,`tiny_params` or `loan_params`. Parameters can be changed in those yaml files to reproduce our experiments.

___

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{
xie2020dba,
title={DBA: Distributed Backdoor Attacks against Federated Learning},
author={Chulin Xie and Keli Huang and Pin-Yu Chen and Bo Li},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=rkgyS0VFvr}
}
```

___

## Acknowledgement

* [AlphaPav/DBA](https://github.com/AlphaPav/DBA)
* [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
* [krishnap25/RFA](https://github.com/krishnap25/RFA)
* [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
