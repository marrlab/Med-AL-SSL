import matplotlib.pyplot as plt
import matplotlib.style as style

from data.cifar10_dataset import Cifar10Dataset
from data.config.cifar10_config import set_cifar_configs
from data.config.isic_config import set_isic_configs
from data.isic_dataset import ISICDataset
from data.jurkat_dataset import JurkatDataset
from data.matek_dataset import MatekDataset
from data.plasmodium_dataset import PlasmodiumDataset
from data.config.matek_config import set_matek_configs
from data.config.jurkat_config import set_jurkat_configs
from data.config.plasmodium_config import set_plasmodium_configs
from results import ratio_metrics
from options.visualization_options import get_arguments
import os
import numpy as np

"""
plot the accuracy vs data proportion being used, graph
credits to: Alex Olteanu (https://www.dataquest.io/blog/making-538-plots/) for the plot style
:return: None
"""

datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset,
            'jurkat': JurkatDataset, 'isic': ISICDataset}
configs = {'matek': set_matek_configs, 'jurkat': set_jurkat_configs,
           'plasmodium': set_plasmodium_configs, 'cifar10': set_cifar_configs, 'isic': set_isic_configs}

plot_configs = {'matek': (2, 5),
                'jurkat': (2, 4),
                'plasmodium': (1, 2),
                'cifar10': (2, 5),
                'isic': (2, 4)
                }

fully_supervised = {
                'matek': {'recall': 0.8621, 'f1-score': 0.7348, 'precision': 0.7112, 'accuracy': 0.9469},
                'jurkat': {'recall': 0.6351, 'f1-score': 0.6138, 'precision': 0.7056, 'accuracy': 0.7445},
                'plasmodium': 0.9763,
                'cifar10': 0.75,
                'isic': {'recall': 0.6752, 'f1-score': 0.6702, 'precision': 0.6707, 'accuracy': 0.7452}
                }

fully_supervised_std = {
    'matek': {'recall': 0.0318, 'f1-score': 0.0224, 'precision': 0.0103, 'accuracy': 0.0249},
    'jurkat': {'recall': 0.0326, 'f1-score': 0.0326, 'precision': 0.0216, 'accuracy': 0.0265},
    'plasmodium': 0.9763,
    'cifar10': 0.75,
    'isic': {'recall': 0.0159, 'f1-score': 0.0165, 'precision': 0.0119, 'accuracy': 0.0356}
}


methods_default = [
    'random_sampling',
    'mc_dropout',
    'entropy_based',
    'augmentations_based',
]


def plot_ratio_class_wise_metrics(metric, classes, label_y, prop, plot_config):
    fig = plt.figure(figsize=(20, 7))
    style.use('fivethirtyeight')

    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1],
              [238 / 255, 136 / 255, 102 / 255, 1]]
    ax_main = fig.add_subplot(111)
    for i, cls in enumerate(classes):
        ax = fig.add_subplot(plot_config[0], plot_config[1], i+1)
        for j, method in enumerate(methods_default):
            if len(metric[j]) == 0:
                continue
            linestyle = '-'
            ax.errorbar(prop, metric[j][i][1], yerr=(metric[j][i][0]-metric[j][i][2])/2, color=colors[j % len(colors)],
                        label=methods_default[j], linewidth=2, linestyle=linestyle, marker='o', capsize=3)
            # ax.fill_between(prop, metric[j][i][0], metric[j][i][2], color=colors[i % len(colors)], alpha=0.05)
            ax.set_title(classes[i])

    ax_main.spines['top'].set_color('none')
    ax_main.spines['bottom'].set_color('none')
    ax_main.spines['left'].set_color('none')
    ax_main.spines['right'].set_color('none')
    ax_main.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax_main.set_xlabel("Active Learning Cycles", fontsize=20, weight='bold', alpha=.75)
    ax_main.set_ylabel(label_y, fontsize=20, weight='bold', alpha=.75)
    plt.show()


def plot_ratio_metrics(prop, metric, label_y, fully_supervised_metric, save_path, methods, title,
                       fully_supervised_std_metric):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "ultralight"
    plt.rcParams["axes.labelweight"] = "ultralight"

    plt.figure(figsize=(14, 10))
    plt.rc('xtick', labelsize=45)
    plt.rc('ytick', labelsize=45)
    # plt.grid(color='black')
    style.use(['science', 'no-latex'])

    colors = [[86 / 255, 180 / 255, 233 / 255, 1], [230 / 255, 159 / 255, 0, 1], [212 / 255, 16 / 255, 16 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1],
              [238 / 255, 136 / 255, 102 / 255, 1]]

    if 'Recall' in label_y:
        plt.errorbar(prop, [fully_supervised_metric['recall']] * len(prop),
                     yerr=[fully_supervised_std_metric['recall']] * len(prop),
                     color=[0, 0, 0, 1], label='fully_supervised', linewidth=2, linestyle='--', marker=',', capsize=3)
        plt.fill_between(prop,
                         np.array([fully_supervised_metric['recall']] * len(prop)) - fully_supervised_std_metric[
                             'recall'],
                         np.array([fully_supervised_metric['recall']] * len(prop)) + fully_supervised_std_metric[
                             'recall'],
                         color=[0, 0, 0, 1], alpha=0.05)
    elif 'Precision' in label_y:
        plt.errorbar(prop, [fully_supervised_metric['precision']] * len(prop),
                     yerr=[fully_supervised_std_metric['precision']] * len(prop), color=[0, 0, 0, 1],
                     label='fully_supervised', linewidth=2, linestyle='--', marker=',', capsize=3)
        plt.fill_between(prop,
                         np.array([fully_supervised_metric['precision']] * len(prop)) - fully_supervised_std_metric[
                             'precision'],
                         np.array([fully_supervised_metric['precision']] * len(prop)) + fully_supervised_std_metric[
                             'precision'],
                         color=[0, 0, 0, 1], alpha=0.05)
    elif 'F1-score' in label_y:
        plt.errorbar(prop, [fully_supervised_metric['f1-score']] * len(prop),
                     yerr=[fully_supervised_std_metric['f1-score']] * len(prop), color=[0, 0, 0, 1],
                     label='fully_supervised', linewidth=2, linestyle='--', marker=',', capsize=3)
        plt.fill_between(prop,
                         np.array([fully_supervised_metric['f1-score']] * len(prop)) - fully_supervised_std_metric[
                             'f1-score'],
                         np.array([fully_supervised_metric['f1-score']] * len(prop)) + fully_supervised_std_metric[
                             'f1-score'],
                         color=[0, 0, 0, 1], alpha=0.05)
    else:
        plt.errorbar(prop, [fully_supervised_metric['accuracy']] * len(prop),
                     yerr=[fully_supervised_std_metric['accuracy']] * len(prop), color=[0, 0, 0, 1],
                     label='fully_supervised', linewidth=2, linestyle='--', marker=',', capsize=3)
        plt.fill_between(prop,
                         np.array([fully_supervised_metric['accuracy']] * len(prop)) - fully_supervised_std_metric[
                             'accuracy'],
                         np.array([fully_supervised_metric['accuracy']] * len(prop)) + fully_supervised_std_metric[
                             'accuracy'],
                         color=[0, 0, 0, 1], alpha=0.05)

    for i, method in enumerate(methods):
        if len(metric[i]) == 0:
            continue
        if 'Semi-supervised' in method:
            linestyle = '-'
        else:
            linestyle = '--'

        if 'Entropy Based' in method:
            c = colors[3]
        elif 'MC Dropout' in method:
            c = colors[1]
        elif 'Augmentations Based' in method:
            c = colors[2]
        else:
            c = colors[0]

        if 'SimCLR' in method:
            marker = 's'
        elif 'Autoencoder' in method:
            marker = 'o'
        elif 'ImageNet' in method:
            marker = '^'
        else:
            marker = ','
        plt.errorbar(prop, metric[i][1], yerr=(metric[i][0]-metric[i][2])/2, color=c, markersize=15,
                     label=method, linewidth=2, linestyle=linestyle, marker=marker, capsize=3)
        plt.fill_between(prop, metric[i][0], metric[i][2], color=c, alpha=0.05)

    plt.xlabel("Amount of added annotated data (%)", fontsize=45)
    plt.ylabel(label_y, fontsize=45)
    # plt.legend(loc='lower right', fontsize=18)
    # plt.title(title, fontsize=45, weight='bold', alpha=.75)
    plt.xticks(ticks=prop)
    plt.yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    plt.savefig(save_path)


def plot_epoch_class_wise_loss(values, classes, label_y, epochs, plot_config):
    fig = plt.figure(figsize=(20, 7))
    style.use('fivethirtyeight')

    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1],
              [238 / 255, 136 / 255, 102 / 255, 1]]
    ax_main = fig.add_subplot(111)
    for i, cls in enumerate(classes):
        ax = fig.add_subplot(plot_config[0], plot_config[1], i+1)
        if len(values[i]) == 0:
            continue
        linestyle = '-'
        ax.plot(epochs, values[i][0], color=colors[0], label='Train Loss',
                linewidth=2, linestyle=linestyle)
        ax.plot(epochs, values[i][1], color=colors[1], label='Valid Loss',
                linewidth=2, linestyle=linestyle)
        ax.set_title(classes[i])

    ax_main.spines['top'].set_color('none')
    ax_main.spines['bottom'].set_color('none')
    ax_main.spines['left'].set_color('none')
    ax_main.spines['right'].set_color('none')
    ax_main.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax_main.set_xlabel("Epochs", fontsize=20, weight='bold')
    ax_main.set_ylabel(label_y, fontsize=15, weight='bold')
    plt.show()


def plot_ae_loss(losses, logs, epochs):
    plt.figure(figsize=(15, 10))
    style.use('fivethirtyeight')

    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1],
              [238 / 255, 136 / 255, 102 / 255, 1]]

    for i, log in enumerate(logs):
        if i >= len(losses):
            break
        plt.plot(epochs, log, color=colors[i], label=losses[i], linewidth=2)

    plt.xlabel("Epochs", fontsize=20, weight='bold', alpha=.75)
    plt.ylabel("Loss Value", fontsize=20, weight='bold', alpha=.75)
    plt.legend(loc='lower right', fontsize=18)
    plt.show()


def main(args):
    args = configs[args.dataset](args)

    num = [i for i in range(0, 21, 5)]

    if 'macro' in args.metric_ratio:
        y_label = f'Macro {args.metric.capitalize()}'
    else:
        y_label = 'Accuracy'

    dataset_title = {'matek': 'White blood cells', 'jurkat': 'Jurkat cell cycle', 'isic': 'Skin lesions',
                     'plasmodium': 'Red blood cells'}

    '''
    ratio_class_wise_metrics_log = ratio_class_wise_metrics(args.metric, dataset.classes, args.dataset)
    plot_ratio_class_wise_metrics(ratio_class_wise_metrics_log, dataset.classes, y_label, num,
                                  plot_configs[args.dataset])
    '''

    ratio_metrics_logs = ratio_metrics(args.metric, args.dataset, cls=args.metric_ratio,
                                       methods=args.methods_default_results)
    plot_ratio_metrics(num[:5], ratio_metrics_logs, y_label, fully_supervised[args.dataset],
                       save_path=args.save_path, methods=args.methods_default,
                       title=dataset_title[args.dataset],
                       fully_supervised_std_metric=fully_supervised_std[args.dataset])

    '''
    epoch_class_wise_log = epoch_class_wise_loss(dataset.classes, methods[args.method_id], args.dataset)
    plot_epoch_class_wise_loss(epoch_class_wise_log, dataset.classes, y_label_alt,
                               list(range(len(epoch_class_wise_log[0][0]))), plot_configs[args.dataset])

    ae_loss_logs = ae_loss(args.dataset)
    plot_ae_loss(losses=['bce', 'l1', 'l2', 'ssim'], logs=ae_loss_logs, epochs=list(range(len(ae_loss_logs[0]))))
    '''


if __name__ == '__main__':
    root_vis = '/home/ahmad/thesis/visualization'
    arguments = get_arguments()
    methods_states = {
        'a': ['Random Sampling', 'MC Dropout', 'Entropy Based', 'Augmentations Based'],
        'b': ['Random', 'ImageNet', 'SimCLR', 'Autoencoder'],
        'c': ['Supervised', 'Semi-supervised', 'Semi-supervised + ImageNet', 'Semi-supervised + Autoencoder',
              'Semi-supervised + SimCLR'],
        'd': ['Random Sampling',
              'Augmentations Based',
              'Augmentations Based + ImageNet',
              'Augmentations Based + SimCLR',
              'Augmentations Based + Autoencoder'],
        'e': ['Random Sampling', 'Semi-supervised',
              'Semi-supervised + Augmentations Based',
              'Semi-supervised + Entropy Based',
              'Semi-supervised + MC Dropout'],
        'f': ['Random Sampling', 'Semi-supervised', 'Semi-supervised + ImageNet',
              'Semi-supervised + Augmentations Based + ImageNet',
              'Semi-supervised + Entropy Based + ImageNet',
              'Semi-supervised + MC Dropout + ImageNet'],
        'g': [
                'Random Sampling',
                'MC Dropout',
                'Entropy Based',
                'Augmentations Based',
                'Random Sampling + ImageNet',
                'MC Dropout + ImageNet',
                'Entropy Based + ImageNet',
                'Augmentations Based + ImageNet',
                'Random Sampling + Autoencoder',
                'MC Dropout + Autoencoder',
                'Entropy Based + Autoencoder',
                'Augmentations Based + Autoencoder',
                'Random Sampling + SimCLR',
                'MC Dropout + SimCLR',
                'Entropy Based + SimCLR',
                'Augmentations Based + SimCLR',
                'Semi-supervised',
                'Semi-supervised + MC Dropout',
                'Semi-supervised + Entropy Based',
                'Semi-supervised + Augmentations Based',
                'Semi-supervised + ImageNet',
                'Semi-supervised + MC Dropout + ImageNet'
                'Semi-supervised + Entropy Based + ImageNet',
                'Semi-supervised + Augmentations Based + ImageNet',
                'Semi-supervised + Autoencoder',
                'Semi-supervised + MC Dropout + Autoencoder',
                'Semi-supervised + Entropy Based + Autoencoder',
                'Semi-supervised + Augmentations Based + Autoencoder',
                'Semi-supervised + SimCLR',
                'Semi-supervised + MC Dropout + SimCLR',
                'Semi-supervised + Entropy Based + SimCLR',
                'Semi-supervised + Augmentations Based + SimCLR',
        ],
        'h': ['Semi-supervised + Augmentations Based + SimCLR', 'Semi-supervised + Augmentations Based + ImageNet',
              'MC Dropout + ImageNet',
              'Entropy Based + ImageNet', 'Augmentations Based + ImageNet',
              'Random Sampling'],
        'i': ['Semi-supervised + Augmentations Based + SimCLR', 'Semi-supervised + Augmentations Based + ImageNet',
              'Augmentations Based + SimCLR', 'Augmentations Based + ImageNet',
              'Random Sampling']
    }
    methods_states_results = {
        'a': ['random_sampling', 'mc_dropout', 'entropy_based', 'augmentations_based'],
        'b': ['random_sampling', 'random_sampling_pretrained', 'simclr', 'auto_encoder'],
        'c': ['random_sampling', 'fixmatch', 'fixmatch_pretrained',
              'fixmatch_pretrained_autoencoder', 'fixmatch_pretrained_simclr'],
        'd': ['random_sampling', 'augmentations_based',
              'augmentations_based_pretrained',
              'simclr_with_al_augmentations_based',
              'auto_encoder_with_al_augmentations_based'],
        'e': ['random_sampling', 'fixmatch',
              'fixmatch_with_al_augmentations_based',
              'fixmatch_with_al_entropy_based',
              'fixmatch_with_al_mc_dropout'],
        'f': ['random_sampling', 'fixmatch', 'fixmatch_pretrained',
              'fixmatch_with_al_augmentations_based_pretrained',
              'fixmatch_with_al_entropy_based_pretrained',
              'fixmatch_with_al_mc_dropout_pretrained'],
        'g': [
                'random_sampling',
                'mc_dropout',
                'entropy_based',
                'augmentations_based',
                'random_sampling_pretrained',
                'mc_dropout_pretrained',
                'entropy_based_pretrained',
                'augmentations_based_pretrained',
                'auto_encoder',
                'auto_encoder_with_al_mc_dropout',
                'auto_encoder_with_al_entropy_based',
                'auto_encoder_with_al_augmentations_based',
                'simclr',
                'simclr_with_al_mc_dropout',
                'simclr_with_al_entropy_based',
                'simclr_with_al_augmentations_based',
                'fixmatch',
                'fixmatch_with_al_mc_dropout',
                'fixmatch_with_al_entropy_based',
                'fixmatch_with_al_augmentations_based',
                'fixmatch_pretrained',
                'fixmatch_with_al_mc_dropout_pretrained',
                'fixmatch_with_al_entropy_based_pretrained',
                'fixmatch_with_al_augmentations_based_pretrained',
                'fixmatch_pretrained_autoencoder',
                'fixmatch_with_al_mc_dropout_pretrained_autoencoder',
                'fixmatch_with_al_entropy_based_pretrained_autoencoder',
                'fixmatch_with_al_augmentations_based_pretrained_autoencoder',
                'fixmatch_pretrained_simclr',
                'fixmatch_with_al_mc_dropout_pretrained_simclr',
                'fixmatch_with_al_entropy_based_pretrained_simclr',
                'fixmatch_with_al_augmentations_based_pretrained_simclr',
        ],
        'h': ['fixmatch_with_al_augmentations_based_pretrained_simclr',
              'fixmatch_with_al_augmentations_based_pretrained',
              'mc_dropout_pretrained',
              'entropy_based_pretrained',
              'augmentations_based_pretrained', 'random_sampling'],
        'i': ['fixmatch_with_al_augmentations_based_pretrained_simclr',
              'fixmatch_with_al_augmentations_based_pretrained',
              'simclr_with_al_augmentations_based',
              'augmentations_based_pretrained',
              'random_sampling']
    }

    dataset = 'matek'

    for k, method_state in methods_states.items():
        if arguments.run_batch:
            states = [
                (dataset, 'recall', 'macro avg'),
                (dataset, 'precision', 'macro avg'),
                (dataset, 'f1-score', 'macro avg'),
                (dataset, 'recall', 'accuracy'),
            ]

            for (d, m, r) in states:
                root_path = os.path.join(root_vis, d, k)
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                arguments.dataset = d
                arguments.metric = m
                arguments.metric_ratio = r
                arguments.methods_default = method_state
                arguments.methods_default_results = methods_states_results[k]
                arguments.save_path = os.path.join(root_path, f'{m}_{r}.png')
                main(args=arguments)
        else:
            main(args=arguments)
    '''
    states = [
        'random_sampling',
        'mc_dropout',
        'entropy_based',
        'augmentations_based',
        'random_sampling_pretrained',
        'mc_dropout_pretrained',
        'entropy_based_pretrained',
        'augmentations_based_pretrained',
        'auto_encoder',
        'auto_encoder_with_al_mc_dropout',
        'auto_encoder_with_al_entropy_based',
        'auto_encoder_with_al_augmentations_based',
        'simclr',
        'simclr_with_al_mc_dropout',
        'simclr_with_al_entropy_based',
        'simclr_with_al_augmentations_based',
        'fixmatch',
        'fixmatch_with_al_mc_dropout',
        'fixmatch_with_al_entropy_based',
        'fixmatch_with_al_augmentations_based',
        'fixmatch_pretrained',
        'fixmatch_with_al_mc_dropout_pretrained',
        'fixmatch_with_al_entropy_based_pretrained',
        'fixmatch_with_al_augmentations_based_pretrained',
        'fixmatch_pretrained_autoencoder',
        'fixmatch_with_al_mc_dropout_pretrained_autoencoder',
        'fixmatch_with_al_entropy_based_pretrained_autoencoder',
        'fixmatch_with_al_augmentations_based_pretrained_autoencoder',
        'fixmatch_pretrained_simclr',
        'fixmatch_with_al_mc_dropout_pretrained_simclr',
        'fixmatch_with_al_entropy_based_pretrained_simclr',
        'fixmatch_with_al_augmentations_based_pretrained_simclr',
    ]
    
    datasets = ['isic', 'matek', 'jurkat']
    datasets_rep = ['Skin Lesions', 'White blood cells', 'Jurkat cells cycle']
    inits = ['Random', 'ImageNet', 'Autoencoder', 'SimCLR']
    trainings = ['Supervised learning', 'Semi-supervised learning']
    uncertainty_samplings = ['Random', 'MC dropout', 'Entropy Based', 'Augmentations Based']
    metrics = ['recall', 'precision', 'f1-score', 'accuracy']
    metrics_rep = ['Recall', 'Precision', 'F1-score', 'Accuracy']

    rows = []
    for dataset, dataset_rep in zip(datasets, datasets_rep):
        i = 0
        for training in trainings:
            for init in inits:
                for uncertainty_sampling in uncertainty_samplings:
                    row = {'Dataset': dataset_rep, 'Network Initialization': init, 'Training Method': training,
                           'Uncertainty Sampling': uncertainty_sampling}
                    for metric, metric_rep in zip(metrics, metrics_rep):
                        if metric == 'accuracy':
                            ratio_metrics_logs = ratio_metrics('recall', dataset, cls='accuracy', methods=[states[i]])
                        else:
                            print(i)
                            ratio_metrics_logs = ratio_metrics(metric, dataset, cls='macro avg', methods=[states[i]])
                        if len(ratio_metrics_logs[0]) == 0:
                            print(row, states[i])
                            continue
                        row.update({metric_rep: ratio_metrics_logs[0][1][-1],
                                    f'{metric_rep} STD.': ratio_metrics_logs[0][1][-1] - ratio_metrics_logs[0][0][-1]})
                    i = i + 1
                    if len(ratio_metrics_logs[0]) == 0:
                        continue
                    else:
                        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv('results.csv')
    '''
"""
Combinations:
    'random_sampling',
    'mc_dropout',
    'entropy_based',
    'augmentations_based',

    'fixmatch',
    'fixmatch_pretrained',
    'fixmatch_with_al_augmentations_based',
    'fixmatch_with_al_augmentations_based_pretrained',
    'fixmatch_with_al_entropy_based',
    'fixmatch_with_al_entropy_based_pretrained',
    'fixmatch_with_al_mc_dropout',
    'fixmatch_with_al_mc_dropout_pretrained'
    
    'simclr',
    'simclr_pretrained',
    'simclr_with_al_augmentations_based',
    'fixmatch_with_al_augmentations_based_pretrained_simclr',
    'simclr_with_al_entropy_based',
    'fixmatch_with_al_entropy_based_pretrained_simclr',
    'simclr_with_al_mc_dropout',
    'fixmatch_with_al_mc_dropout_pretrained_simclr'
    
    'random_sampling',
    'mc_dropout',
    'entropy_based',
    'augmentations_based',
    'random_sampling_pretrained',
    'mc_dropout_pretrained',
    'entropy_based_pretrained',
    'augmentations_based_pretrained',
    
    'auto_encoder',
    'auto_encoder_pretrained',
    'auto_encoder_with_al_augmentations_based',
    'auto_encoder_with_al_augmentations_based_pretrained',
    'auto_encoder_with_al_entropy_based',
    'auto_encoder_with_al_entropy_based_pretrained',
    'auto_encoder_with_al_mc_dropout',
    'auto_encoder_with_al_mc_dropout_pretrained'
    
    'augmentations_based',
    'augmentations_based_pretrained',
    'simclr_with_al_augmentations_based',
    'auto_encoder_with_al_augmentations_based'
    
    'fixmatch',
    'fixmatch_pretrained',
    'fixmatch_pretrained_simclr',
    'fixmatch_pretrained_autoencoder',
"""
