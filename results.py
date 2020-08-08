from options.results_options import get_arguments
import numpy as np
import os
import json


args = get_arguments()

states = [
    ('random_sampling', 'least_confidence', 'pseudo_labeling'),
    ('active_learning', 'margin_confidence', 'pseudo_labeling'),
    ('active_learning', 'least_confidence', 'pseudo_labeling'),
    ('active_learning', 'ratio_confidence', 'pseudo_labeling'),
    ('active_learning', 'entropy_based', 'pseudo_labeling'),
    ('active_learning', 'density_weighted', 'pseudo_labeling'),
    ('active_learning', 'mc_dropout', 'pseudo_labeling'),
    ('semi_supervised', 'least_confidence', 'pseudo_labeling'),
    ('semi_supervised', 'least_confidence', 'auto_encoder'),
    ('semi_supervised', 'least_confidence', 'simclr'),
]


# noinspection PyTypeChecker,PyUnresolvedReferences
def get_metrics(name, log_path):
    filenames = os.listdir(log_path)
    metrics = {'acc1': [[], []], 'acc5': [[], []], 'prec': [[], []], 'recall': [[], []],
               'acc1_std': [], 'acc5_std': [], 'prec_std': [], 'recall_std': []}
    ratios = []

    for filename in filenames:
        with open(os.path.join(log_path, filename), 'r') as fp:
            file = json.load(fp)
        if name in file['name']:
            for k, v in file['metrics'].items():
                v = np.round(v[0:4], decimals=2)
                if k not in ratios:
                    metrics['acc1_' + k] = [v[0]]
                    metrics['acc5_' + k] = [v[1]]
                    metrics['prec_' + k] = [v[2]]
                    metrics['recall_' + k] = [v[3]]
                    ratios.append(k)
                else:
                    metrics['acc1_' + k].append(v[0])
                    metrics['acc5_' + k].append(v[1])
                    metrics['prec_' + k].append(v[2])
                    metrics['recall_' + k].append(v[3])
        else:
            continue

    for ratio in ratios:
        acc1_m = np.round(np.mean(metrics["acc1_" + ratio]), decimals=2)
        acc1_std = np.round(np.std(metrics["acc1_" + ratio]), decimals=2)
        acc5_m = np.round(np.mean(metrics["acc5_" + ratio]), decimals=2)
        acc5_std = np.round(np.std(metrics["acc5_" + ratio]), decimals=2)
        prec_m = np.round(np.mean(metrics["prec_" + ratio]), decimals=2)
        prec_std = np.round(np.std(metrics["prec_" + ratio]), decimals=2)
        recall_m = np.round(np.mean(metrics["recall_" + ratio]), decimals=2)
        recall_std = np.round(np.std(metrics["recall_" + ratio]), decimals=2)
        metrics['acc1'][0].append(acc1_m)
        metrics['acc1'][1].append(acc1_std)
        metrics['acc5'][0].append(acc5_m)
        metrics['acc5'][1].append(acc5_std)
        metrics['prec'][0].append(prec_m)
        metrics['prec'][1].append(prec_std)
        metrics['recall'][0].append(recall_m)
        metrics['recall'][1].append(recall_std)
        metrics['acc1_std'].append(str(acc1_m) + '±' + str(acc1_std))
        metrics['acc5_std'].append(str(acc5_m) + '±' + str(acc5_std))
        metrics['prec_std'].append(str(prec_m) + '±' + str(prec_std))
        metrics['recall_std'].append(str(recall_m) + '±' + str(recall_std))

    metrics['acc1'][0] = np.array(metrics['acc1'][0])
    metrics['acc1'][1] = np.array(metrics['acc1'][1])
    metrics['acc5'][0] = np.array(metrics['acc5'][0])
    metrics['acc5'][1] = np.array(metrics['acc5'][1])
    metrics['prec'][0] = np.array(metrics['prec'][0]) * 100
    metrics['prec'][1] = np.array(metrics['prec'][1]) * 100
    metrics['recall'][0] = np.array(metrics['recall'][0]) * 100
    metrics['recall'][1] = np.array(metrics['recall'][1]) * 100

    return metrics, ratios


# noinspection PyTypeChecker,PyUnresolvedReferences
def get_class_specific_metrics(name, log_path, class_id):
    filenames = os.listdir(log_path)
    metrics = {'acc1': [[], []], 'acc5': [[], []], 'prec': [[], []], 'recall': [[], []]}
    ratios = []

    for filename in filenames:
        with open(os.path.join(log_path, filename), 'r') as fp:
            file = json.load(fp)
        if name in file['name']:
            for k, v in file['metrics'].items():
                v = np.array(v[5])
                TP = v[class_id, class_id]
                FN = np.sum(v[class_id, :]) - TP
                FP = np.sum(v[:, class_id]) - TP
                TN = np.sum(v) - TP - FN - FP
                acc1 = ((TP + TN) / (TP + TN + FP + FN)) * 100
                prec = (TP / (TP + FP)) * 100
                recall = (TP / (TP + FN)) * 100
                if k not in ratios:
                    metrics['acc1_' + k] = [acc1]
                    metrics['acc5_' + k] = [0]
                    metrics['prec_' + k] = [prec]
                    metrics['recall_' + k] = [recall]
                    ratios.append(k)
                else:
                    metrics['acc1_' + k].append(acc1)
                    metrics['acc5_' + k].append(0)
                    metrics['prec_' + k].append(prec)
                    metrics['recall_' + k].append(recall)
        else:
            continue

    for ratio in ratios:
        acc1_m = np.round(np.mean(metrics["acc1_" + ratio]), decimals=2)
        acc1_std = np.round(np.std(metrics["acc1_" + ratio]), decimals=2)
        acc5_m = np.round(np.mean(metrics["acc5_" + ratio]), decimals=2)
        acc5_std = np.round(np.std(metrics["acc5_" + ratio]), decimals=2)
        prec_m = np.round(np.mean(metrics["prec_" + ratio]), decimals=2)
        prec_std = np.round(np.std(metrics["prec_" + ratio]), decimals=2)
        recall_m = np.round(np.mean(metrics["recall_" + ratio]), decimals=2)
        recall_std = np.round(np.std(metrics["recall_" + ratio]), decimals=2)
        metrics['acc1'][0].append(acc1_m)
        metrics['acc1'][1].append(acc1_std)
        metrics['acc5'][0].append(acc5_m)
        metrics['acc5'][1].append(acc5_std)
        metrics['prec'][0].append(prec_m)
        metrics['prec'][1].append(prec_std)
        metrics['recall'][0].append(recall_m)
        metrics['recall'][1].append(recall_std)

    metrics['acc1'][0] = np.array(metrics['acc1'][0])
    metrics['acc1'][1] = np.array(metrics['acc1'][1])
    metrics['acc5'][0] = np.array(metrics['acc5'][0])
    metrics['acc5'][1] = np.array(metrics['acc5'][1])
    metrics['prec'][0] = np.array(metrics['prec'][0])
    metrics['prec'][1] = np.array(metrics['prec'][1])
    metrics['recall'][0] = np.array(metrics['recall'][0])
    metrics['recall'][1] = np.array(metrics['recall'][1])

    return metrics, ratios


def get_batch_metrics(met='acc1', class_specific=False, class_id=0):
    metric = []
    ratios = []

    for (m, u, s) in states:
        if m == 'semi_supervised':
            args.name = s
        elif m == 'active_learning':
            args.name = u
        else:
            args.name = m

        if class_specific:
            metrics, ratio = get_class_specific_metrics(args.name, args.log_path, class_id=class_id)
        else:
            metrics, ratio = get_metrics(args.name, args.log_path)

        metric.append(metrics[met][0] - metrics[met][1])
        metric.append(metrics[met][0].tolist())
        metric.append(metrics[met][0] + metrics[met][1])
        ratios.append([float(x) for x in ratio])

    return metric, ratios


def print_individual_metric():
    if args.weak_supervision_strategy == 'semi_supervised':
        args.name = f"{args.dataset}@{args.arch}@{args.semi_supervised_method}"
    elif args.weak_supervision_strategy == 'active_learning':
        args.name = f"{args.dataset}@{args.arch}@{args.uncertainty_sampling_method}"
    else:
        args.name = f"{args.dataset}@{args.arch}@{args.weak_supervision_strategy}"

    metrics, ratios = get_metrics(args.name, args.log_path)

    print(f'* Name: {args.name}\n\n'
          f'* Metrics: \n'
          f'* Ratios: {ratios}\n'
          f'* Acc1: {metrics["acc1_std"]}\n'
          f'* Acc5: {metrics["acc5_std"]}\n'
          f'* Prec: {metrics["prec_std"]}\n'
          f'* Recall: {metrics["recall_std"]}\n\n')

    print(f'* Metrics for visualization:\n'
          f'* Ratios: {[float(x) for x in ratios]}\n'
          f'{np.round(metrics["acc1"][0] - metrics["acc1"][1], decimals=2).tolist()},\n'
          f'{metrics["acc1"][0].tolist()},\n'
          f'{np.round(metrics["acc1"][0] + metrics["acc1"][1], decimals=2).tolist()},\n'
          f'{np.round(metrics["acc5"][0] - metrics["acc5"][1], decimals=2).tolist()},\n'
          f'{metrics["acc5"][0].tolist()},\n'
          f'{np.round(metrics["acc5"][0] + metrics["acc5"][1], decimals=2).tolist()},\n'
          f'{np.round(metrics["prec"][0] - metrics["prec"][1], decimals=2).tolist()},\n'
          f'{metrics["prec"][0].tolist()},\n'
          f'{np.round(metrics["prec"][0] + metrics["prec"][1], decimals=2).tolist()},\n'
          f'{np.round(metrics["recall"][0] - metrics["recall"][1], decimals=2).tolist()},\n'
          f'{metrics["recall"][0].tolist()},\n'
          f'{np.round(metrics["recall"][0] + metrics["recall"][1], decimals=2).tolist()},\n')
