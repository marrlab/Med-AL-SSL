from options.results_options import get_arguments
import numpy as np
import os
import pandas as pd

args = get_arguments()

methods = [
    'fixmatch',
    'fixmatch_pretrained',
    'fixmatch_with_al_augmentations_based',
    'fixmatch_with_al_augmentations_based_pretrained',
    'fixmatch_with_al_entropy_based',
    'fixmatch_with_al_entropy_based_pretrained',
    'fixmatch_with_al_mc_dropout',
    'fixmatch_with_al_mc_dropout_pretrained'
]


def ratio_class_wise_metrics(metric, classes, dataset):
    logs = os.listdir(args.log_path)
    dump_log = []
    for i, method in enumerate(methods):
        dump_log.append([])
        for filename in logs:
            if f"{dataset}@resnet@{method}" != filename.split('-')[1] or 'epoch' in filename or 'ae-loss' in filename:
                continue
            df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
            for j, cls in enumerate(classes):
                dump_log[i].append([])
                dump = df[cls][metric].tolist()
                max_metric = 0
                for k, m in enumerate(dump):
                    max_metric = m if m > max_metric else max_metric
                    dump[k] = max_metric
                dump_log[i][j].append(dump)

    metrics_log = []

    for i, method in enumerate(methods):
        metrics_log.append([])
        if len(dump_log[i]) == 0:
            continue
        for j, cls in enumerate(classes):
            metrics_log[i].append([])
            dump = np.array(dump_log[i][j])
            mean = dump.mean(axis=0)
            std = dump.std(axis=0)
            metrics_log[i][j].append((mean - std))
            metrics_log[i][j].append(mean)
            metrics_log[i][j].append((mean + std))

    return metrics_log


def ratio_metrics(metric, dataset, cls):
    logs = os.listdir(args.log_path)
    dump_log = []
    for i, method in enumerate(methods):
        dump_log.append([])
        for filename in logs:
            if f"{dataset}@resnet@{method}" != filename.split('-')[1] or 'epoch' in filename or 'ae-loss' in filename:
                continue
            df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
            dump = df[cls][metric].tolist()
            max_metric = 0
            for k, m in enumerate(dump):
                max_metric = m if m > max_metric else max_metric
                dump[k] = max_metric
            dump_log[i].append(dump)

    metrics_log = []

    for i, method in enumerate(methods):
        metrics_log.append([])
        if len(dump_log[i]) == 0:
            continue
        dump = np.array(dump_log[i])
        mean = dump.mean(axis=0)
        std = dump.std(axis=0)
        metrics_log[i].append((mean - std))
        metrics_log[i].append(mean)
        metrics_log[i].append((mean + std))

    return metrics_log


# noinspection PyTypeChecker
def epoch_class_wise_loss(classes, method, dataset):
    logs = os.listdir(args.log_path)
    dump_log = []

    for i, cls in enumerate(classes):
        dump_log.append([[], []])
        for filename in logs:
            if f"{dataset}@resnet@{method}" != filename.split('-')[1] or \
                    'epoch' not in filename or 'ae-loss' in filename:
                continue
            df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
            dump_log[i][0].append(df[f'{cls}-train-loss']['0'].tolist())
            dump_log[i][1].append(df[f'{cls}-val-loss']['0'].tolist())

    metrics_log = []

    for i, cls in enumerate(classes):
        metrics_log.append([[], []])
        dump = np.array(dump_log[i][0][0])
        metrics_log[i][0].extend(dump.tolist())
        dump = np.array(dump_log[i][1][0])
        metrics_log[i][1].extend(dump.tolist())

    return metrics_log


def ae_loss(dataset):
    logs = os.listdir(args.log_path)
    dump_log = []

    for filename in logs:
        if 'ae-loss' not in filename or dataset not in filename:
            continue
        df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
        for col in df.columns:
            dump_log.append(df[col].tolist())

    return dump_log
