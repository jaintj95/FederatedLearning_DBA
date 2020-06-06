# DBA

## Note - This repo was created so that I could cleanup AlphaPav's repo (https://github.com/AlphaPav/DBA) for personal use. All original work belongs to the concerned parties mentioned in the original repo. 

In this repository, code is for our ICLR 2020 paper [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr)

## Installation
Install Pytorch

## Usage
### Prepare the dataset:
#### LOAN dataset:

- download the raw dataset [lending-club-loan-data.zip](https://www.kaggle.com/wendykan/lending-club-loan-data/) into dir `./utils` 
- preprocess the dataset. 

```
cd ./utils
./process_loan_data.sh
```

#### Tiny-imagenet dataset:

- download the dataset [tiny-imagenet-200.zip](https://tiny-imagenet.herokuapp.com/) into dir `./utils` 
- reformat the dataset.
```
cd ./utils
python tinyimagenet_reformat.py
```
Note: If the above command fails, comment out the following snippet from tinyimagenet_reformat.py
```
with zipfile.ZipFile(source_dir, 'r') as zip_ref:
    if not os.path.exists("../data"):
        os.mkdir("../data")
    zip_ref.extractall("../data")
```
and then execute the `process_tiny_data.sh` script in utils directory

#### Others:
MNIST and CIFAR will be automatically download

### Reproduce experiments: 

- prepare the pretrained model:
Our pretrained clean models for attack can be downloaded from [Google Drive](https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing). You can also train from the round 0 to obtain the pretrained clean model.

- we can use Visdom to monitor the training progress.
```
python -m visdom.server -p 8098
```

- run experiments for the four datasets:
```
python main.py --params utils/X.yaml
```
`X` = `mnist_params`, `cifar_params`,`tiny_params` or `loan_params`. Parameters can be changed in those yaml files to reproduce our experiments.



Stay tuned for further updates, thanks!

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
## Acknowledgement 
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
