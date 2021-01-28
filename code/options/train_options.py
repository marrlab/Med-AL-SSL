import argparse
from os.path import expanduser

home = expanduser("~")
code_dir = 'med_active_learning'

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging', allow_abbrev=False)

parser.add_argument('--name', default='run_0', type=str, help='name of current running experiment')

parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs for AL training')

parser.add_argument('--start-epoch', default=0, type=int, help='starting epoch number (useful when resuming)')

parser.add_argument('--resume', action='store_true', help='flag to be set if an existing model is to be loaded')

parser.add_argument('--load-pretrained', action='store_false', help='load pretrained imagenet weights or not')

parser.add_argument('--batch-size', default=256, type=int, help='batch size for AL training (default: 256)')

parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for AL optimizer')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')

parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay for AL optimizer')

parser.add_argument('--print-freq', default=10, type=int, help='print frequency per step')

parser.add_argument('--layers', default=28, type=int, help='total number of layers for ResNext architecture')

parser.add_argument('--widen-factor', default=10, type=int, help='widen factor for ResNext architecture')

parser.add_argument('--drop-rate', default=0.15, type=float, help='dropout probability for ResNet/LeNet architecture')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentations or not')

parser.add_argument('--add-labeled-epochs', default=2, type=int,
                    help='if recall doesn\'t improve perform AL cycle')

parser.add_argument('--add-labeled', default=100, type=int,
                    help='amount of labeled data to be added during each AL cycle')

parser.add_argument('--start-labeled', default=100, type=int,
                    help='amount of labeled data to start the AL training')

parser.add_argument('--stop-labeled', default=1020, type=int,
                    help='amount of labeled data to stop the AL training')

parser.add_argument('--labeled-warmup-epochs', default=0, type=int,
                    help='number of warmup epochs before AL training')

parser.add_argument('--unlabeled-subset', default=0.3, type=float,
                    help='the subset of the unlabeled data to use for AL algorithms')

parser.add_argument('--oversampling', action='store_true', help='perform oversampling for labeled dataset or not')

parser.add_argument('--merged', action='store_false',
                    help='to merge certain classes in the dataset (see dataset scripts)')

parser.add_argument('--remove-classes', action='store_true',
                    help='to remove certain classes in the dataset (see dataset scripts)')

parser.add_argument('--arch', default='resnet',
                    type=str, choices=['wideresnet', 'densenet', 'lenet', 'resnet'],
                    help='the architecture to use for AL training')

parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'fl'],
                    help='the loss to be used. ce = cross entropy and fl = focal loss')

parser.add_argument('--log-path', default='/home/qasima/med_active_learning/code/logs/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.add_argument('--al', '--uncertainty-sampling-method', default='entropy_based', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based',
                             'mc_dropout', 'learning_loss', 'augmentations_based'],
                    help='the AL algorithm to use')

parser.add_argument('--mc-dropout-iterations', default=25, type=int,
                    help='number of iterations for mc dropout')

parser.add_argument('--augmentations_based_iterations', default=25, type=int,
                    help='number of iterations for augmentations based AL algorithm')

parser.add_argument('--root', default='~/datasets/thesis/stratified/', type=str,
                    help='the root path for the datasets')

parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling', 'fully_supervised'],
                    help='the weakly supervised strategy to use')

parser.add_argument('--semi-supervised-method', default='fixmatch_with_al', type=str,
                    choices=['pseudo_labeling', 'auto_encoder', 'simclr', 'fixmatch', 'auto_encoder_cl',
                             'auto_encoder_no_feat', 'simclr_with_al', 'auto_encoder_with_al', 'fixmatch_with_al'],
                    help='the SSL algorithm to use')

parser.add_argument('--semi-supervised-uncertainty-method', default='entropy_based', type=str,
                    choices=['entropy_based', 'augmentations_based'],
                    help='the AL algorithm to use in conjunction with a SSL algorithm')

parser.add_argument('--pseudo-labeling-threshold', default=0.99, type=int,
                    help='the threshold for considering the pseudo label as the actual label')

parser.add_argument('--pseudo-labeling-num', default=30, type=int,
                    help='the number of points to be pseudo labeled at each iteration')

parser.add_argument('--simclr-train-epochs', default=200, type=int, help='number of total epochs for SimCLR training')

parser.add_argument('--simclr-temperature', default=0.1, type=float, help='the temperature term for simclr loss')

parser.add_argument('--simclr-normalize', action='store_false',
                    help='normalize the hidden feat vectors in simclr or not')

parser.add_argument('--simclr-batch-size', default=1024, type=int,
                    help='batch size for simclr training (default: 1024)')

parser.add_argument('--simclr-arch', default='resnet', type=str, choices=['lenet', 'resnet'],
                    help='which encoder architecture to use for simclr')

parser.add_argument('--simclr-base-lr', default=0.25, type=float, help='base learning rate for SimCLR optimizer')

parser.add_argument('--simclr-optimizer', default='adam', type=str, choices=['adam', 'lars'],
                    help='which optimizer to use for simclr training')

parser.add_argument('--simclr-resume', action='store_true',
                    help='flag to be set if an existing simclr model is to be loaded')

parser.add_argument('--weighted', action='store_false', help='to use weighted loss or not (only in case of ce)')

parser.add_argument('--eval', action='store_true', help='only perform evaluation and exit')

parser.add_argument('--dataset', default='matek', type=str, choices=['cifar10', 'matek', 'cifar100', 'jurkat',
                                                                     'plasmodium', 'isic', 'retinopathy'],
                    help='the dataset to train on')

parser.add_argument('--checkpoint-path', default=f'/home/qasima/med_active_learning/code/runs/', type=str,
                    help='the directory root for saving/resuming checkpoints from')

parser.add_argument('--seed', default=9999, type=int, choices=[6666, 9999, 2323, 5555],
                    help='the random seed to set')

parser.add_argument('--store-logs', action='store_false', help='store the logs after training')

parser.add_argument('--run-batch', action='store_false', help='run all methods in batch mode')

parser.add_argument('--reset-model', action='store_true', help='reset models after every labels injection cycle')

parser.add_argument('--fixmatch-mu', default=8, type=int,
                    help='coefficient of unlabeled batch size i.e. mu.B from paper')

parser.add_argument('--fixmatch-lambda-u', default=1, type=float,
                    help='coefficient of unlabeled loss')

parser.add_argument('--fixmatch-threshold', default=0.95, type=float,
                    help='pseudo label threshold')

parser.add_argument('--fixmatch-k-img', default=8192, type=int,
                    help='number of labeled examples')

parser.add_argument('--fixmatch-epochs', default=1000, type=int,
                    help='epochs for SSL or SSL + AL training')

parser.add_argument('--fixmatch-warmup', default=0, type=int,
                    help='warmup epochs with unlabeled data')

parser.add_argument('--semi-supervised-init', default=None, type=str,
                    choices=[None, 'random', 'pretrained', 'simclr', 'autoencoder'],
                    help='the self-supervised method to use for semi-supervised methods')

parser.add_argument('--learning-loss-weight', default=1.0, type=float,
                    help='the weight for the loss network, loss term in the objective function')

parser.add_argument('--dlctcs-loss-weight', default=100, type=float,
                    help='the weight for classification loss in dlctcs')

parser.add_argument('--autoencoder-train-epochs', default=20, type=int,
                    help='number of total epochs for autoencoder training')

parser.add_argument('--autoencoder-z-dim', default=128, type=float,
                    help='the bottleneck dimension for the autoencoder architecture')

parser.add_argument('--autoencoder-resume', action='store_true',
                    help='flag to be set if an existing autoencoder model is to be loaded')

parser.add_argument('--k-medoids', action='store_true', help='to perform k medoids init with SimCLR')

parser.add_argument('--k-medoids-n-clusters', default=10, type=int, help='number of k medoids clusters')

parser.add_argument('--novel-class-detection', action='store_true', help='turn on novel class detection')

parser.add_argument('--gpu-id', default='0', type=str, help='the id of the GPU to use')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
