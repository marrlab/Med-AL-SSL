import argparse
from os.path import expanduser

home = expanduser("~")

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--metric', default='recall', type=str,
                    choices=['precision', 'recall', 'f1-score'],
                    help='the class wise metric to display')

parser.add_argument('--metric-ratio', default='macro avg', type=str,
                    choices=['macro avg', 'weighted avg', 'accuracy'],
                    help='the overall metric mode')

parser.add_argument('--root', default=home+'/datasets/thesis/stratified/', type=str,
                    help='the root path for the datasets')

parser.add_argument('--dataset', default='matek', type=str, choices=['cifar10', 'matek', 'cifar100', 'jurkat',
                                                                     'plasmodium'],
                    help='the dataset to train on')

parser.add_argument('--oversampling', action='store_true', help='perform oversampling for labeled dataset')

parser.add_argument('--merged', action='store_false',
                    help='to merge certain classes in the dataset (see dataset scripts to see which classes)')

parser.add_argument('--remove_classes', action='store_false',
                    help='to remove certain classes in the dataset (see dataset scripts to see which classes)')

parser.add_argument('--method-id', default=1, type=int, help='the id of the method')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
