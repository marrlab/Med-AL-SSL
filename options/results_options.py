import argparse

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling'],
                    help='the weakly supervised strategy to use')
parser.add_argument('--semi-supervised-method', default='simclr', type=str,
                    choices=['pseudo_labeling', 'auto_encoder', 'simclr'],
                    help='the semi supervised method to use')
parser.add_argument('--uncertainty-sampling-method', default='mc_dropout', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based',
                             'density_weighted', 'mc_dropout'],
                    help='the uncertainty sampling method to use')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'matek', 'cifar100'],
                    help='the dataset to train on')
parser.add_argument('--arch', default='lenet', type=str, choices=['wideresnet', 'densenet', 'lenet'],
                    help='arch name')
parser.add_argument('--log-path', default='/home/qasima/med_active_learning/logs_backup/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
