from options.results_options import get_arguments
import numpy as np
import os
import pandas as pd

args = get_arguments()

methods = [
    'random_sampling',
    'entropy_based',
    'learning_loss',
    'mc_dropout',
    'pseudo_labeling',
    'auto_encoder',
    'fixmatch',
    'simclr',
]


def ratio_class_wise_metrics(metric, classes, dataset):
    logs = os.listdir(args.log_path)
    dump_log = []
    for i, method in enumerate(methods):
        dump_log.append([])
        for filename in logs:
            if method not in filename or 'epoch' in filename or dataset not in filename or 'ae-loss' in filename:
                continue
            df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
            for j, cls in enumerate(classes):
                dump_log[i].append([])
                dump_log[i][j].append(df[cls][metric].tolist())

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


def ratio_metrics(metric, dataset, weighted=False):
    logs = os.listdir(args.log_path)
    dump_log = []
    for i, method in enumerate(methods):
        dump_log.append([])
        for filename in logs:
            if method not in filename or 'epoch' in filename or dataset not in filename or 'ae-loss' in filename:
                continue
            df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
            cls = 'weighted avg' if weighted else 'macro avg'
            dump_log[i].append(df[cls][metric].tolist())

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


def epoch_class_wise_loss(classes, method, dataset):
    logs = os.listdir(args.log_path)
    dump_log = []

    for i, cls in enumerate(classes):
        dump_log.append([[], []])
        for filename in logs:
            if method not in filename or 'epoch' not in filename or dataset not in filename or 'ae-loss' in filename:
                continue
            df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
            dump_log[i][0].append(df[f'{cls}-train-loss']['0'].tolist())
            dump_log[i][1].append(df[f'{cls}-val-loss']['0'].tolist())

    metrics_log = []

    for i, cls in enumerate(classes):
        metrics_log.append([[], []])
        dump = np.array(dump_log[i][0])
        mean = dump.mean(axis=0)
        metrics_log[i][0].extend(mean.tolist())
        dump = np.array(dump_log[i][1])
        mean = dump.mean(axis=0)
        metrics_log[i][1].extend(mean.tolist())

    return metrics_log


def ae_loss():
    logs = os.listdir(args.log_path)
    dump_log = []

    for filename in logs:
        if 'ae-loss' not in filename:
            continue
        df = pd.read_csv(os.path.join(args.log_path, filename), index_col=0)
        for col in df.columns:
            dump_log.append(df[col].tolist())

    return dump_log
