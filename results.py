import argparse
from utils import print_metrics


parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling'],
                    help='the weakly supervised strategy to use')
parser.add_argument('--semi-supervised-method', default='pseudo_labeling', type=str,
                    choices=['pseudo_labeling'],
                    help='the semi supervised method to use')
parser.add_argument('--uncertainty-sampling-method', default='least_confidence', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based'],
                    help='the uncertainty sampling method to use')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'matek', 'cifar100'],
                    help='the dataset to train on')
parser.add_argument('--arch', default='lenet', type=str, choices=['wideresnet', 'densenet', 'lenet'],
                    help='arch name')
parser.add_argument('--log-path', default='/home/qasima/med_active_learning/logs/', type=str,
                    help='the directory root for storing/retrieving the logs')

parser.set_defaults(augment=True)

args = parser.parse_args()


def main():
    if args.weak_supervision_strategy == 'semi_supervised':
        args.name = f"{args.dataset}@{args.arch}@{args.semi_supervised_method}"
    elif args.weak_supervision_strategy == 'active_learning':
        args.name = f"{args.dataset}@{args.arch}@{args.uncertainty_sampling_method}"
    else:
        args.name = f"{args.dataset}@{args.arch}@{args.weak_supervision_strategy}"

    print_metrics(args.name, args.log_path)


if __name__ == '__main__':
    main()
