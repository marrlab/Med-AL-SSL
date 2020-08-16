import argparse

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')

parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling', 'fully_supervised'],
                    help='the weakly supervised strategy to use')

parser.add_argument('--semi-supervised-method', default='auto_encoder', type=str,
                    choices=['pseudo_labeling', 'auto_encoder', 'simclr'],
                    help='the semi supervised method to use')

parser.add_argument('--uncertainty-sampling-method', default='least_confidence', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based',
                             'density_weighted', 'mc_dropout'],
                    help='the uncertainty sampling method to use')

parser.add_argument('--arch', default='resnet', type=str, choices=['wideresnet', 'densenet', 'lenet', 'resnet'],
                    help='arch name')

parser.add_argument('--log-path', default='/home/qasima/med_active_learning/logs/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.set_defaults(augment=True)

arguments = parser.parse_args()


def get_arguments():
    return arguments
