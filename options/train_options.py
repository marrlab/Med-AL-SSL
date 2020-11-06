import argparse
from os.path import expanduser

home = expanduser("~")
code_dir = 'med_active_learning'

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')

parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run')

parser.add_argument('--autoencoder-train-epochs', default=20, type=int,
                    help='number of total epochs to run')

parser.add_argument('--simclr-train-epochs', default=200, type=int,
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')

parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')

parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')

parser.add_argument('--drop-rate', default=0.15, type=float,
                    help='dropout probability (default: 0.3)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')

parser.add_argument('--resume', action='store_true',
                    help='flag to be set if an existing model is to be loaded')

parser.add_argument('--load-pretrained', action='store_false',
                    help='load pretrained imagenet weights for some methods')

parser.add_argument('--simclr-resume', action='store_true',
                    help='flag to be set if an existing simclr model is to be loaded')

parser.add_argument('--autoencoder-resume', action='store_false',
                    help='flag to be set if an existing autoencoder model is to be loaded')

parser.add_argument('--name', default=' ', type=str,
                    help='name of experiment')

parser.add_argument('--add-labeled-epochs', default=20, type=int,
                    help='add labeled data through sampling strategy after epochs')

parser.add_argument('--add-labeled', default=100, type=int,
                    help='amount of labeled data to be added in each cycle')

parser.add_argument('--start-labeled', default=100, type=int,
                    help='amount of labeled data to start the training process with')

parser.add_argument('--stop-labeled', default=1020, type=int,
                    help='amount of labeled data to stop the training process at')

parser.add_argument('--labeled-warmup-epochs', default=35, type=int,
                    help='how many epochs to warmup for, without sampling or pseudo labeling')

parser.add_argument('--unlabeled-subset', default=0.3, type=float,
                    help='the subset of the unlabeled data to use, to avoid choosing similar data points')

parser.add_argument('--oversampling', action='store_true', help='perform oversampling for labeled dataset')

parser.add_argument('--merged', action='store_false',
                    help='to merge certain classes in the dataset (see dataset scripts to see which classes)')

parser.add_argument('--remove-classes', action='store_true',
                    help='to remove certain classes in the dataset (see dataset scripts to see which classes)')

parser.add_argument('--arch', default='resnet', type=str, choices=['wideresnet', 'densenet', 'lenet', 'resnet'],
                    help='arch name')

parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'fl'],
                    help='the loss to be used. ce = cross entropy and fl = focal loss')

parser.add_argument('--log-path', default=f'{home}/{code_dir}/logs_isic_novel/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.add_argument('--uncertainty-sampling-method', default='entropy_based', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based',
                             'mc_dropout', 'learning_loss', 'augmentations_based'],
                    help='the uncertainty sampling method to use')

parser.add_argument('--mc-dropout-iterations', default=25, type=int,
                    help='number of iterations for mc dropout')

parser.add_argument('--augmentations_based_iterations', default=25, type=int,
                    help='number of iterations for augmentations based uncertainty sampling')

parser.add_argument('--root', default=home+'/datasets/thesis/stratified/', type=str,
                    help='the root path for the datasets')

parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling', 'fully_supervised'],
                    help='the weakly supervised strategy to use')

parser.add_argument('--semi-supervised-method', default='fixmatch_with_al', type=str,
                    choices=['pseudo_labeling', 'auto_encoder', 'simclr', 'fixmatch', 'auto_encoder_cl',
                             'auto_encoder_no_feat', 'simclr_with_al', 'auto_encoder_with_al', 'fixmatch_with_al'],
                    help='the semi supervised method to use')

parser.add_argument('--semi-supervised-uncertainty-method', default='entropy_based', type=str,
                    choices=['entropy_based', 'augmentations_based'],
                    help='the uncertainty sampling method to use for SSL methods')

parser.add_argument('--pseudo-labeling-threshold', default=0.9, type=int,
                    help='the threshold for considering the pseudo label as the actual label')

parser.add_argument('--simclr-temperature', default=0.1, type=float, help='the temperature term for simclr loss')

parser.add_argument('--simclr-normalize', action='store_false', help='normalize the hidden feat vectors in simclr')

parser.add_argument('--simclr-batch-size', default=1024, type=int,
                    help='mini-batch size for simclr (default: 1024)')

parser.add_argument('--simclr-arch', default='resnet', type=str, choices=['lenet', 'resnet'],
                    help='which encoder architecture to use for simclr')

parser.add_argument('--simclr-base-lr', default=0.25, type=float, help='base learning rate, rescaled by batch_size/256')
parser.add_argument('--simclr-optimizer', default='adam', type=str, choices=['adam', 'lars'],
                    help='which optimizer to use for simclr')

parser.add_argument('--weighted', action='store_false', help='to use weighted loss or not')

parser.add_argument('--eval', action='store_true', help='only perform evaluation and exit')

parser.add_argument('--dataset', default='matek', type=str, choices=['cifar10', 'matek', 'cifar100', 'jurkat',
                                                                     'plasmodium', 'isic'],
                    help='the dataset to train on')

parser.add_argument('--checkpoint-path', default=f'{home}/{code_dir}/runs/', type=str,
                    help='the directory root for saving/resuming checkpoints from')

parser.add_argument('--seed', default=9999, type=int, choices=[6666, 9999, 2323, 5555], help='the random seed to set')

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

parser.add_argument('--fixmatch-epochs', default=600, type=int,
                    help='epochs for fixmatch algorithm')

parser.add_argument('--fixmatch-warmup', default=0, type=int,
                    help='warmup epochs with unlabeled data')

parser.add_argument('--fixmatch-init', default=None, type=str,
                    choices=[None, 'random', 'pretrained', 'simclr', 'autoencoder'],
                    help='the semi supervised method to use')

parser.add_argument('--learning-loss-weight', default=1.0, type=float,
                    help='the weight for the loss network, loss term in the objective function')

parser.add_argument('--dlctcs-loss-weight', default=100, type=float,
                    help='the weight for classification loss in dlctcs')

parser.add_argument('--autoencoder-z-dim', default=128, type=float,
                    help='the bottleneck dimension for the autoencoder architecture')

parser.add_argument('--k-medoids', action='store_true', help='to perform k medoids init with SimCLR')

parser.add_argument('--k-medoids-n-clusters', default=10, type=int, help='number of k medoids clusters')

parser.add_argument('--novel-class-detection', action='store_true', help='turn on novel class detection')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
