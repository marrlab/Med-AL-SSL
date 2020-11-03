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
                'matek': 0.8621,
                'jurkat': 0.6125,
                'plasmodium': 0.9763,
                'cifar10': 0.75,
                'isic': 0.6752
                }


methods = [
    'fixmatch_with_al_augmentations_based',
    'fixmatch_with_al_augmentations_based_pretrained',
    'fixmatch_with_al_augmentations_based_pretrained_simclr',
    'fixmatch_with_al_augmentations_based_pretrained_autoencoder'
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
        for j, method in enumerate(methods):
            if len(metric[j]) == 0:
                continue
            linestyle = '-'
            ax.errorbar(prop, metric[j][i][1], yerr=(metric[j][i][0]-metric[j][i][2])/2, color=colors[j % len(colors)],
                        label=methods[j], linewidth=2, linestyle=linestyle, marker='o', capsize=3)
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


def plot_ratio_metrics(prop, metric, label_y, fully_supervised_metric):
    plt.figure(figsize=(14, 10))
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    # plt.grid(color='black')
    style.use('fivethirtyeight')

    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1],
              [238 / 255, 136 / 255, 102 / 255, 1]]

    plt.errorbar(prop, [fully_supervised_metric] * len(prop), yerr=[0] * len(prop), color=colors[0],
                 label='fully_supervised', linewidth=2, linestyle='--', marker='o', capsize=3)

    for i, method in enumerate(methods):
        if len(metric[i]) == 0:
            continue
        linestyle = '-'
        plt.errorbar(prop, metric[i][1], yerr=(metric[i][0]-metric[i][2])/2, color=colors[i % len(colors)],
                     label=method, linewidth=2, linestyle=linestyle, marker='o', capsize=3)
        plt.fill_between(prop, metric[i][0], metric[i][2], color=colors[i % len(colors)], alpha=0.05)

    plt.xlabel("Active Learning Cycles", fontsize=25, weight='bold', alpha=.75)
    plt.ylabel(label_y, fontsize=25, weight='bold', alpha=.75)

    plt.legend(loc='lower right', fontsize=18)
    plt.show()


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

    ax_main.set_xlabel("Epochs", fontsize=20, weight='bold', alpha=.75)
    ax_main.set_ylabel(label_y, fontsize=15, weight='bold', alpha=.75)
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

    num = [i for i, n in enumerate(range(args.add_labeled, args.stop_labeled + 10, args.add_labeled))]

    dataset_class = datasets[args.dataset](root=args.root, oversampling=args.oversampling, merged=args.merged,
                                           remove_classes=args.remove_classes)

    dataset, _, _, _, _, _ = dataset_class.get_dataset()

    dataset_title = {'cifar10': ' cifar-10 dataset', 'matek': ' matek dataset', 'jurkat': ' jurkat dataset',
                     'plasmodium': ' plasmodium dataset', 'isic': 'isic dataset'}
    y_label = f'{args.metric_ratio} of {args.metric} on {dataset_title[args.dataset]}'

    '''
    ratio_class_wise_metrics_log = ratio_class_wise_metrics(args.metric, dataset.classes, args.dataset)
    plot_ratio_class_wise_metrics(ratio_class_wise_metrics_log, dataset.classes, y_label, num,
                                  plot_configs[args.dataset])
    '''

    ratio_metrics_logs = ratio_metrics(args.metric, args.dataset, cls=args.metric_ratio)
    plot_ratio_metrics(num, ratio_metrics_logs, y_label, fully_supervised[args.dataset])

    '''
    epoch_class_wise_log = epoch_class_wise_loss(dataset.classes, methods[args.method_id], args.dataset)
    plot_epoch_class_wise_loss(epoch_class_wise_log, dataset.classes, y_label_alt,
                               list(range(len(epoch_class_wise_log[0][0]))), plot_configs[args.dataset])

    ae_loss_logs = ae_loss(args.dataset)
    plot_ae_loss(losses=['bce', 'l1', 'l2', 'ssim'], logs=ae_loss_logs, epochs=list(range(len(ae_loss_logs[0]))))
    '''


if __name__ == '__main__':
    root_vis = '/home/qasima/med_active_learning/visualization'
    arguments = get_arguments()
    if arguments.run_batch:
        states = [
            ('matek', 'recall', 'macro avg'),
            ('matek', 'precision', 'macro avg'),
            ('matek', 'f1-score', 'macro avg'),
            ('matek', 'recall', 'accuracy'),
        ]

        for (d, m, r) in states:
            root_path = os.path.join(root_vis, d, 'a')
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            arguments.dataset = d
            arguments.metric = m
            arguments.metric_ratio = r
            arguments.save_path = os.path.join(root_path, f'{m}_{r}.png')
            main(args=arguments)
    else:
        main(args=arguments)


"""
Combinations:
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
