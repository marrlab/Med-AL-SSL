# Annotation-efficient classification combining active learning, pre-training and semi-supervised learning for biomedical images

Repository for implementation of active learning and semi-supervised learning algorithms and applying it to medical imaging datasets

## Active Learning algorithms
* Least Confidence Sampling [1]
* Margin Sampling [1]
* Ratio Sampling [1]
* Maximum Entropy Sampling [1]
* Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [2]
* Learning Loss for Active Learning [3]
* BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [4]

## Semi-Supervised Learning algorithms
* Pseudo Labeling [5]
* Autoencoder [5]
* A Simple Framework for Contrastive Learning of Visual Representations [6]
* FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence [7]

## Requirements
```
numpy>=1.18.5
torch>=1.4.0
torchvision>=0.5.0
scikit-learn>=0.23.1
pandas>=1.0.4
Pillow>=7.1.2
matplotlib>=3.2.1
toma>=1.1.0
scikit-image>=0.17.2
pytorch-msssim
scikit-learn-extra
dataclasses
```

## Data
The train/test splits for the three datasets can be downloaded here:

A. White Blood Cell: [Download](https://drive.google.com/file/d/1AWmJPuIjlarQopiltD3r-Ce0jL3KZ2xi/view?usp=sharing "Google Drive Link") <br /> 
B. Skin Lesion: [Download](https://drive.google.com/file/d/1lCrQojKHPKop5UZvRSQ2tGdgTeUMoYdF/view?usp=sharing "Google Drive Link") <br /> 
C. Cell Cycle: [Download](https://drive.google.com/file/d/17fFhSgTL_QXCn0g5u-0LsPIWoaZ-bUx8/view?usp=sharing "Google Drive Link")


## Arguments and Usage
### Usage
```
usage: python3 train.py [-h] [-n NAME] [-e EPOCHS] [--start-epoch START_EPOCH] [-r]
               [--load-pretrained] [-b BATCH_SIZE] [--lr LR] [--mo MO]
               [--nesterov NESTEROV] [--wd WD] [-p PRINT_FREQ]
               [--layers LAYERS] [--widen-factor WIDEN_FACTOR]
               [--drop-rate DROP_RATE] [--no-augment]
               [--add-labeled-epochs ADD_LABELED_EPOCHS]
               [--add-labeled ADD_LABELED] [--start-labeled START_LABELED]
               [--stop-labeled STOP_LABELED]
               [--labeled-warmup-epochs LABELED_WARMUP_EPOCHS]
               [--unlabeled-subset UNLABELED_SUBSET] [-o] [-m] [--rem]
               [--arch {wideresnet,densenet,lenet,resnet}] [-l {ce,fl}]
               [--log-path LOG_PATH]
               [--al {least_confidence,margin_confidence,ratio_confidence,entropy_based,mc_dropout,learning_loss,augmentations_based}]
               [--mc-dropout-iterations MC_DROPOUT_ITERATIONS]
               [--augmentations_based_iterations AUGMENTATIONS_BASED_ITERATIONS]
               [--root ROOT]
               [--weak-supervision-strategy {active_learning,semi_supervised,random_sampling,fully_supervised}]
               [--ssl {pseudo_labeling,auto_encoder,simclr,fixmatch,auto_encoder_cl,auto_encoder_no_feat,simclr_with_al,auto_encoder_with_al,fixmatch_with_al}]
               [--semi-supervised-uncertainty-method {entropy_based,augmentations_based}]
               [--pseudo-labeling-threshold PSEUDO_LABELING_THRESHOLD]
               [--simclr-train-epochs SIMCLR_TRAIN_EPOCHS]
               [--simclr-temperature SIMCLR_TEMPERATURE] [--simclr-normalize]
               [--simclr-batch-size SIMCLR_BATCH_SIZE]
               [--simclr-arch {lenet,resnet}]
               [--simclr-base-lr SIMCLR_BASE_LR]
               [--simclr-optimizer {adam,lars}] [--simclr-resume] [--weighted]
               [--eval] [-d {cifar10,matek,cifar100,jurkat,plasmodium,isic}]
               [--checkpoint-path CHECKPOINT_PATH] [-s {6666,9999,2323,5555}]
               [--store-logs] [--run-batch] [--reset-model]
               [--fixmatch-mu FIXMATCH_MU]
               [--fixmatch-lambda-u FIXMATCH_LAMBDA_U]
               [--fixmatch-threshold FIXMATCH_THRESHOLD]
               [--fixmatch-k-img FIXMATCH_K_IMG]
               [--fixmatch-epochs FIXMATCH_EPOCHS]
               [--fixmatch-warmup FIXMATCH_WARMUP]
               [--fixmatch-init {None,random,pretrained,simclr,autoencoder}]
               [--learning-loss-weight LEARNING_LOSS_WEIGHT]
               [--dlctcs-loss-weight DLCTCS_LOSS_WEIGHT]
               [--autoencoder-train-epochs AUTOENCODER_TRAIN_EPOCHS]
               [--autoencoder-z-dim AUTOENCODER_Z_DIM] [--autoencoder-resume]
               [--k-medoids] [--k-medoids-n-clusters K_MEDOIDS_N_CLUSTERS]
               [--novel-class-detection] [--gpu-id GPU_ID]
```
### Arguments
#### Quick reference table
|Short|Long                                  |Default           |Description                                                         |
|-----|--------------------------------------|------------------|--------------------------------------------------------------------|
|`-h` |`--help`                              |                  |show this help message and exit                                     |
|`-n` |`--name`                              |`run_0`           |name of current running experiment                                  |
|`-e` |`--epochs`                            |`1000`            |number of total epochs for AL training                              |
|     |`--start-epoch`                       |`0`               |starting epoch number (useful when resuming)                        |
|`-r` |`--resume`                            |                  |flag to be set if an existing model is to be loaded                 |
|     |`--load-pretrained`                   |                  |load pretrained imagenet weights or not                             |
|`-b` |`--batch-size`                        |`256`             |batch size for AL training (default: 256)                           |
|     |`--learning-rate`                     |`0.001`           |initial learning rate for AL optimizer                              |
|     |`--momentum`                          |`0.9`             |momentum                                                            |
|     |`--nesterov`                          |                  |nesterov momentum                                                   |
|     |`--weight-decay`                      |`0.0005`          |weight decay for AL optimizer                                       |
|`-p` |`--print-freq`                        |`10`              |print frequency per step                                            |
|     |`--layers`                            |`28`              |total number of layers for ResNext architecture                     |
|     |`--widen-factor`                      |`10`              |widen factor for ResNext architecture                               |
|     |`--drop-rate`                         |`0.15`            |dropout probability for ResNet/LeNet architecture                   |
|     |`--no-augment`                        |                  |whether to use standard augmentations or not                        |
|     |`--add-labeled-epochs`                |`20`              |if recall doesn't improve perform AL cycle                          |
|     |`--add-labeled`                       |`100`             |amount of labeled data to be added during each AL cycle             |
|     |`--start-labeled`                     |`100`             |amount of labeled data to start the AL training                     |
|     |`--stop-labeled`                      |`1020`            |amount of labeled data to stop the AL training                      |
|     |`--labeled-warmup-epochs`             |`35`              |number of warmup epochs before AL training                          |
|     |`--unlabeled-subset`                  |`0.3`             |the subset of the unlabeled data to use for AL algorithms           |
|`-o` |`--oversampling`                      |                  |perform oversampling for labeled dataset or not                     |
|`-m` |`--merged`                            |                  |to merge certain classes in the dataset (see dataset scripts)       |
|     |`--remove-classes`                    |                  |to remove certain classes in the dataset (see dataset scripts)      |
|     |`--architecture`                      |`resnet`          |the architecture to use for AL training                             |
|`-l` |`--loss`                              |`ce`              |the loss to be used. ce = cross entropy and fl = focal loss         |
|     |`--log-path`                          |`~/logs/`         |the directory root for storing/retrieving the logs                  |
|     |`--uncertainty-sampling-method`       |`entropy_based`   |the AL algorithm to use                                             |
|     |`--mc-dropout-iterations`             |`25`              |number of iterations for mc dropout                                 |
|     |`--augmentations_based_iterations`    |`25`              |number of iterations for augmentations based AL algorithm           |
|     |`--root`                              |`~/datasets/`     |the root path for the datasets                                      |
|     |`--weak-supervision-strategy`         |`semi_supervised` |the weakly supervised strategy to use                               |
|     |`--semi-supervised-method`            |`fixmatch_with_al`|the SSL algorithm to use                                            |
|     |`--semi-supervised-uncertainty-method`|`entropy_based`   |the AL algorithm to use in conjunction with a SSL algorithm         |
|     |`--pseudo-labeling-threshold`         |`0.9`             |the threshold for considering the pseudo label as the actual label  |
|     |`--simclr-train-epochs`               |`200`             |number of total epochs for SimCLR training                          |
|     |`--simclr-temperature`                |`0.1`             |the temperature term for simclr loss                                |
|     |`--simclr-normalize`                  |                  |normalize the hidden feat vectors in simclr or not                  |
|     |`--simclr-batch-size`                 |`1024`            |batch size for simclr training (default: 1024)                      |
|     |`--simclr-arch`                       |`resnet`          |which encoder architecture to use for simclr                        |
|     |`--simclr-base-lr`                    |`0.25`            |base learning rate for SimCLR optimizer                             |
|     |`--simclr-optimizer`                  |`adam`            |which optimizer to use for simclr training                          |
|     |`--simclr-resume`                     |                  |flag to be set if an existing simclr model is to be loaded          |
|     |`--weighted`                          |                  |to use weighted loss or not (only in case of ce)                    |
|     |`--eval`                              |                  |only perform evaluation and exit                                    |
|`-d` |`--dataset`                           |`matek`           |the dataset to train on                                             |
|     |`--checkpoint-path`                   |`~/runs/`         |the directory root for saving/resuming checkpoints from             |
|`-s` |`--seed`                              |`9999`            |the random seed to set                                              |
|     |`--store-logs`                        |                  |store the logs after training                                       |
|     |`--run-batch`                         |                  |run all methods in batch mode                                       |
|     |`--reset-model`                       |                  |reset models after every labels injection cycle                     |
|     |`--fixmatch-mu`                       |`8`               |coefficient of unlabeled batch size i.e. mu.B from paper            |
|     |`--fixmatch-lambda-u`                 |`1`               |coefficient of unlabeled loss                                       |
|     |`--fixmatch-threshold`                |`0.95`            |pseudo label threshold                                              |
|     |`--fixmatch-k-img`                    |`8192`            |number of labeled examples                                          |
|     |`--fixmatch-epochs`                   |`1000`            |epochs for SSL or SSL + AL training                                 |
|     |`--fixmatch-warmup`                   |`0`               |warmup epochs with unlabeled data                                   |
|     |`--fixmatch-init`                     |`None`            |the semi supervised method to use                                   |
|     |`--learning-loss-weight`              |`1.0`             |the weight for the loss network, loss term in the objective function|
|     |`--dlctcs-loss-weight`                |`100`             |the weight for classification loss in dlctcs                        |
|     |`--autoencoder-train-epochs`          |`20`              |number of total epochs for autoencoder training                     |
|     |`--autoencoder-z-dim`                 |`128`             |the bottleneck dimension for the autoencoder architecture           |
|     |`--autoencoder-resume`                |                  |flag to be set if an existing autoencoder model is to be loaded     |
|     |`--k-medoids`                         |                  |to perform k medoids init with SimCLR                               |
|     |`--k-medoids-n-clusters`              |`10`              |number of k medoids clusters                                        |
|     |`--novel-class-detection`             |                  |turn on novel class detection                                       |
|     |`--gpu-id`                            |`0`               |the id of the GPU to use                                            |

#### `-h`, `--help`
show this help message and exit

#### `-n`, `--name` (Default: run_0)
name of current running experiment

#### `-e`, `--epochs` (Default: 1000)
number of total epochs for AL training

#### `--start-epoch` (Default: 0)
starting epoch number (useful when resuming)

#### `-r`, `--resume`
flag to be set if an existing model is to be loaded

#### `--load-pretrained`
load pretrained imagenet weights or not

#### `-b`, `--batch-size` (Default: 256)
batch size for AL training (default: 256)

#### `--lr`, `--learning-rate` (Default: 0.001)
initial learning rate for AL optimizer

#### `--mo`, `--momentum` (Default: 0.9)
momentum

#### `--nesterov`
nesterov momentum

#### `--wd`, `--weight-decay` (Default: 0.0005)
weight decay for AL optimizer

#### `-p`, `--print-freq` (Default: 10)
print frequency per step

#### `--layers` (Default: 28)
total number of layers for ResNext architecture

#### `--widen-factor` (Default: 10)
widen factor for ResNext architecture

#### `--drop-rate` (Default: 0.15)
dropout probability for ResNet/LeNet architecture

#### `--no-augment`
whether to use standard augmentations or not

#### `--add-labeled-epochs` (Default: 20)
if recall doesn't improve perform AL cycle

#### `--add-labeled` (Default: 100)
amount of labeled data to be added during each AL cycle

#### `--start-labeled` (Default: 100)
amount of labeled data to start the AL training

#### `--stop-labeled` (Default: 1020)
amount of labeled data to stop the AL training

#### `--labeled-warmup-epochs` (Default: 35)
number of warmup epochs before AL training

#### `--unlabeled-subset` (Default: 0.3)
the subset of the unlabeled data to use for AL algorithms

#### `-o`, `--oversampling`
perform oversampling for labeled dataset or not

#### `-m`, `--merged`
to merge certain classes in the dataset (see dataset scripts)

#### `--rem`, `--remove-classes`
to remove certain classes in the dataset (see dataset scripts)

#### `--arch`, `--architecture` (Default: resnet)
the architecture to use for AL training

#### `-l`, `--loss` (Default: ce)
the loss to be used. ce = cross entropy and fl = focal loss

#### `--log-path` (Default: ~/logs/)
the directory root for storing/retrieving the logs

#### `--al`, `--uncertainty-sampling-method` (Default: entropy_based)
the AL algorithm to use

#### `--mc-dropout-iterations` (Default: 25)
number of iterations for mc dropout

#### `--augmentations_based_iterations` (Default: 25)
number of iterations for augmentations based AL algorithm

#### `--root` (Default: ~/datasets/)
the root path for the datasets

#### `--weak-supervision-strategy` (Default: semi_supervised)
the weakly supervised strategy to use

#### `--ssl`, `--semi-supervised-method` (Default: fixmatch_with_al)
the SSL algorithm to use

#### `--semi-supervised-uncertainty-method` (Default: entropy_based)
the AL algorithm to use in conjunction with a SSL algorithm

#### `--pseudo-labeling-threshold` (Default: 0.9)
the threshold for considering the pseudo label as the actual label

#### `--simclr-train-epochs` (Default: 200)
number of total epochs for SimCLR training

#### `--simclr-temperature` (Default: 0.1)
the temperature term for simclr loss

#### `--simclr-normalize`
normalize the hidden feat vectors in simclr or not

#### `--simclr-batch-size` (Default: 1024)
batch size for simclr training (default: 1024)

#### `--simclr-arch` (Default: resnet)
which encoder architecture to use for simclr

#### `--simclr-base-lr` (Default: 0.25)
base learning rate for SimCLR optimizer

#### `--simclr-optimizer` (Default: adam)
which optimizer to use for simclr training

#### `--simclr-resume`
flag to be set if an existing simclr model is to be loaded

#### `--weighted`
to use weighted loss or not (only in case of ce)

#### `--eval`
only perform evaluation and exit

#### `-d`, `--dataset` (Default: matek)
the dataset to train on

#### `--checkpoint-path` (Default: ~/runs/)
the directory root for saving/resuming checkpoints from

#### `-s`, `--seed` (Default: 9999)
the random seed to set

#### `--store-logs`
store the logs after training

#### `--run-batch`
run all methods in batch mode

#### `--reset-model`
reset models after every labels injection cycle

#### `--fixmatch-mu` (Default: 8)
coefficient of unlabeled batch size i.e. mu.B from paper

#### `--fixmatch-lambda-u` (Default: 1)
coefficient of unlabeled loss

#### `--fixmatch-threshold` (Default: 0.95)
pseudo label threshold

#### `--fixmatch-k-img` (Default: 8192)
number of labeled examples

#### `--fixmatch-epochs` (Default: 1000)
epochs for SSL or SSL + AL training

#### `--fixmatch-warmup` (Default: 0)
warmup epochs with unlabeled data

#### `--fixmatch-init` (Default: None)
the semi supervised method to use

#### `--learning-loss-weight` (Default: 1.0)
the weight for the loss network, loss term in the objective function

#### `--dlctcs-loss-weight` (Default: 100)
the weight for classification loss in dlctcs

#### `--autoencoder-train-epochs` (Default: 20)
number of total epochs for autoencoder training

#### `--autoencoder-z-dim` (Default: 128)
the bottleneck dimension for the autoencoder architecture

#### `--autoencoder-resume`
flag to be set if an existing autoencoder model is to be loaded

#### `--k-medoids`
to perform k medoids init with SimCLR

#### `--k-medoids-n-clusters` (Default: 10)
number of k medoids clusters

#### `--novel-class-detection`
turn on novel class detection

#### `--gpu-id` (Default: 0)
the id of the GPU to use

## Examples

```
To run supervised training with augmentations-based sampling on white blood cell dataset:
python3 train.py --dataset matek --seed <seed> --root <datasets_root> --al augmentations_based --weak-supervision-strategy active_learning

To run semi-supervised training with augmentations-based sampling on cell cycle dataset:
python3 train.py --dataset jurkat --seed <seed> --root <datasets_root> --ssl fixmatch_with_al --weak-supervision-strategy semi_supervised --semi-supervised-uncertainty-method augmentations_based

To run SimCLR self-supervised pre-training on skin lesions dataset:
python3 train.py --dataset isic --seed <seed> --root <datasets_root> --ssl simclr --simclr-resume 
```

## Results

![Result Table](https://github.com/marrlab/Med-AL-SSL/blob/master/results/table.png?raw=true)


Results for the three datasets:

A. White Blood Cell <br /> 
B. Skin Lesion <br /> 
C. Cell Cycle <br />

All results can be downloaded from here: [Download](https://drive.google.com/file/d/1cYHGHTIwpnjjox1Iwg3EwypsOAJlUCD1/view?usp=sharing "Google Drive Link")

## References
[1] Settles, B. (2009). Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences.

[2] Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning (pp. 1050-1059).

[3] Yoo, D., & Kweon, I. S. (2019). Learning loss for active learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 93-102).

[4] Kirsch, A., van Amersfoort, J., & Gal, Y. (2019). Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning. In Advances in Neural Information Processing Systems (pp. 7026-7037).

[5] Van Engelen, J. E., & Hoos, H. H. (2020). A survey on semi-supervised learning. Machine Learning, 109(2), 373-440.

[6] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.

[7] Sohn, K., Berthelot, D., Li, C. L., Zhang, Z., Carlini, N., Cubuk, E. D., ... & Raffel, C. (2020). Fixmatch: Simplifying semi-supervised learning with consistency and confidence. arXiv preprint arXiv:2001.07685.
