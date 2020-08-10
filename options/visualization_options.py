import argparse

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--metric', default='recall', type=str,
                    choices=['acc1', 'acc5', 'prec', 'recall', 'f1'],
                    help='the weakly supervised strategy to use')

parser.add_argument('--class_specific', action='store_false', help='get metrics for a specific class')

parser.add_argument('--class-id', default=1, type=int, help='class id to get the results for')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
