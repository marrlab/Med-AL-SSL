import matplotlib.pyplot as plt
import matplotlib.style as style

from data.cifar10_dataset import Cifar10Dataset
from data.config.cifar10_config import set_cifar_configs
from data.config.isic_config import set_isic_configs
from data.config.retinopathy_config import set_retinopathy_configs
from data.isic_dataset import ISICDataset
from data.jurkat_dataset import JurkatDataset
from data.matek_dataset import MatekDataset
from data.plasmodium_dataset import PlasmodiumDataset
from data.config.matek_config import set_matek_configs
from data.config.jurkat_config import set_jurkat_configs
from data.config.plasmodium_config import set_plasmodium_configs
from data.retinopathy_dataset import RetinopathyDataset
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
            'jurkat': JurkatDataset, 'isic': ISICDataset, 'retinopathy': RetinopathyDataset}
configs = {'matek': set_matek_configs, 'jurkat': set_jurkat_configs,
           'plasmodium': set_plasmodium_configs, 'cifar10': set_cifar_configs, 'isic': set_isic_configs,
           'retinopathy': set_retinopathy_configs}

plot_configs = {'matek': (2, 5),
                'jurkat': (2, 4),
                'plasmodium': (1, 2),
                'cifar10': (2, 5),
                'isic': (2, 4)
                }

fully_supervised = {
    'matek': {'recall': 0.9121, 'f1-score': 0.8348, 'precision': 0.8912, 'accuracy': 0.9652},
    'jurkat': {'recall': 0.7151, 'f1-score': 0.6338, 'precision': 0.8156, 'accuracy': 0.8265},
    'plasmodium': 0.9763,
    'cifar10': 0.75,
    'isic': {'recall': 0.6852, 'f1-score': 0.6802, 'precision': 0.6807, 'accuracy': 0.8194},
    'retinopathy': {'recall': 0.6752, 'f1-score': 0.6702, 'precision': 0.7507, 'accuracy': 0.8255}
}

fully_supervised_std = {
    'matek': {'recall': 0.0318, 'f1-score': 0.0224, 'precision': 0.0103, 'accuracy': 0.0249},
    'jurkat': {'recall': 0.0326, 'f1-score': 0.0326, 'precision': 0.0216, 'accuracy': 0.0265},
    'plasmodium': 0.9763,
    'cifar10': 0.75,
    'isic': {'recall': 0.0159, 'f1-score': 0.0165, 'precision': 0.0119, 'accuracy': 0.0356},
    'retinopathy': {'recall': 0.0128, 'f1-score': 0.0265, 'precision': 0.0325, 'accuracy': 0.0287},
}

methods_default = [
    'random_sampling',
    'mc_dropout',
    'entropy_based',
    'augmentations_based',
]

dataset_rep = {
    'matek': 'White Blood Cell Dataset',
    'isic': 'Skin Lesion Dataset',
    'jurkat': 'Cell Cycle Dataset',
    'retinopathy': 'Retinopathy Dataset',
}


def plot_ratio_class_wise_metrics(metric, classes, label_y, prop, plot_config):
    fig = plt.figure(figsize=(20, 7))
    style.use('fivethirtyeight')

    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1],
              [238 / 255, 136 / 255, 102 / 255, 1]]
    ax_main = fig.add_subplot(111)
    for i, cls in enumerate(classes):
        ax = fig.add_subplot(plot_config[0], plot_config[1], i + 1)
        for j, method in enumerate(methods_default):
            if len(metric[j]) == 0:
                continue
            linestyle = '-'
            ax.errorbar(prop, metric[j][i][1], yerr=(metric[j][i][0] - metric[j][i][2]) / 2,
                        color=colors[j % len(colors)],
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
        plt.errorbar(prop, metric[i][1], yerr=(metric[i][0] - metric[i][2]) / 2, color=c, markersize=15,
                     label=method, linewidth=2, linestyle=linestyle, marker=marker, capsize=3)
        plt.fill_between(prop, metric[i][0], metric[i][2], color=c, alpha=0.05)

    plt.xlabel("Added annotated data (%)", fontsize=45)
    plt.ylabel(label_y, fontsize=45)
    plt.legend(loc='lower right', fontsize=18)
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
        ax = fig.add_subplot(plot_config[0], plot_config[1], i + 1)
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
    """
    root_vis = '/home/ahmad/thesis/visualization'
    arguments = get_arguments()
    methods_states = {
        'a': ['Augmentations Based +  Random + Supervised',
              'MC Dropout +  Random + Supervised',
              'Entropy Based +  Random + Supervised',
              'Random Sampling +  Random + Supervised',
              'Least Confidence +  Random + Supervised',
              'Margin Confidence +  Random + Supervised',
              'Learning Loss +  Random + Supervised',
              'Badge +  Random + Supervised'],
        'd': ['Augmentations Based + ImageNet + Supervised',
              'Augmentations Based + SimCLR + Supervised',
              'Learning Loss + ImageNet + Supervised',
              'Learning Loss + SimCLR + Supervised',
              'Badge + ImageNet + Supervised',
              'Badge + SimCLR + Supervised',
              'Random Sampling +  Random + Supervised'],
        'i': ['Augmentations Based + ImageNet + FixMatch',
              'Augmentations Based + SimCLR + FixMatch',
              'Augmentations Based + ImageNet + Supervised',
              'Augmentations Based + SimCLR + Supervised',
              'Learning Loss + ImageNet + FixMatch',
              'Learning Loss + SimCLR + FixMatch',
              'Learning Loss + ImageNet + Supervised',
              'Learning Loss + SimCLR + Supervised',
              'Badge + ImageNet + FixMatch',
              'Badge + SimCLR + FixMatch',
              'Badge + ImageNet + Supervised',
              'Badge + SimCLR + Supervised',
              'Augmentations Based + SimCLR + Pseudo',
              'Learning Loss + SimCLR + Pseudo',
              'Badge + SimCLR + Pseudo',
              'Random Sampling +  Random + Supervised']
    }
    methods_states_results = {
        'a': ['augmentations_based', 'mc_dropout', 'entropy_based',
              'random_sampling', 'least_confidence', 'margin_confidence', 'learning_loss', 'badge'],
        'd': ['augmentations_based_pretrained',
              'simclr_with_al_augmentations_based',
              'learning_loss_pretrained',
              'simclr_with_al_learning_loss',
              'badge_pretrained',
              'simclr_with_al_badge',
              'random_sampling'],
        'i': ['fixmatch_with_al_augmentations_based_pretrained_pretrained',
              'fixmatch_with_al_augmentations_based_pretrained_simclr',
              'augmentations_based_pretrained',
              'simclr_with_al_augmentations_based',
              'fixmatch_with_al_learning_loss_pretrained_pretrained',
              'fixmatch_with_al_learning_loss_pretrained_simclr',
              'learning_loss_pretrained',
              'simclr_with_al_learning_loss',
              'fixmatch_with_al_badge_pretrained_pretrained',
              'fixmatch_with_al_badge_pretrained_simclr',
              'badge_pretrained',
              'simclr_with_al_badge',
              'pseudo_label_with_al_augmentations_based_pretrained_simclr',
              'pseudo_label_with_al_learning_loss_pretrained_simclr',
              'pseudo_label_with_al_badge_pretrained_simclr',
              'random_sampling']
    }

    dataset = 'isic'

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "ultralight"
    plt.rcParams["axes.labelweight"] = "ultralight"
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rcParams['legend.fontsize'] = 25

    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle(dataset_rep[dataset], fontsize=45)

    for itera, (k, method_state) in enumerate(methods_states.items()):
        if arguments.run_batch:
            states = [
                # (dataset, 'recall', 'accuracy'),
                # (dataset, 'precision', 'macro avg'),
                (dataset, 'recall', 'macro avg'),
                (dataset, 'f1-score', 'macro avg'),
            ]

            for j, (d, m, r) in enumerate(states):
                root_path = os.path.join(root_vis, d, k)
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                arguments.dataset = d
                arguments.metric = m
                arguments.metric_ratio = r
                arguments.methods_default = method_state
                arguments.methods_default_results = methods_states_results[k]
                arguments.save_path = os.path.join(root_path, f'{m}_{r}.png')
                args = configs[arguments.dataset](arguments)

                num = [i for i in range(0, 41, 5)]

                if 'macro' in args.metric_ratio:
                    y_label = f'Macro {args.metric.capitalize()}'
                else:
                    y_label = 'Accuracy'

                dataset_title = {'matek': 'White blood cells', 'jurkat': 'Jurkat cell cycle',
                                 'isic': 'Skin lesions',
                                 'plasmodium': 'Red blood cells', 'retinopathy': 'Retina'}

                '''
                ratio_class_wise_metrics_log = ratio_class_wise_metrics(args.metric, dataset.classes, args.dataset)
                plot_ratio_class_wise_metrics(ratio_class_wise_metrics_log, dataset.classes, y_label, num,
                                              plot_configs[args.dataset])
                '''

                ratio_metrics_logs = ratio_metrics(args.metric, args.dataset, cls=args.metric_ratio,
                                                   methods=args.methods_default_results)

                prop = num[:9]
                metric = ratio_metrics_logs
                label_y = y_label
                fully_supervised_metric = fully_supervised[args.dataset]
                save_path = args.save_path
                methods = args.methods_default
                title = dataset_title[args.dataset]
                fully_supervised_std_metric = fully_supervised_std[args.dataset]

                # plt.figure(figsize=(14, 10))
                # plt.grid(color='black')
                style.use(['science', 'no-latex'])

                colors = [[86 / 255, 180 / 255, 233 / 255, 1], [230 / 255, 159 / 255, 0, 1],
                          [212 / 255, 16 / 255, 16 / 255, 1],
                          [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
                          [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1],
                          [211 / 255, 95 / 255, 183 / 255, 1],
                          [238 / 255, 136 / 255, 102 / 255, 1]]

                if 'Recall' in label_y:
                    ax[itera, j].errorbar(prop, [fully_supervised_metric['recall']] * len(prop),
                                          yerr=[fully_supervised_std_metric['recall']] * len(prop),
                                          color=[0, 0, 0, 1], label='Fully Supervised', linewidth=2, linestyle='--',
                                          marker=',',
                                          capsize=3)
                    ax[itera, j].fill_between(prop,
                                              np.array([fully_supervised_metric['recall']] * len(prop)) -
                                              fully_supervised_std_metric[
                                                  'recall'],
                                              np.array([fully_supervised_metric['recall']] * len(prop)) +
                                              fully_supervised_std_metric[
                                                  'recall'],
                                              color=[0, 0, 0, 1], alpha=0.05)
                elif 'Precision' in label_y:
                    ax[itera, j].errorbar(prop, [fully_supervised_metric['precision']] * len(prop),
                                          yerr=[fully_supervised_std_metric['precision']] * len(prop),
                                          color=[0, 0, 0, 1],
                                          label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                          capsize=3)
                    ax[itera, j].fill_between(prop,
                                              np.array([fully_supervised_metric['precision']] * len(prop)) -
                                              fully_supervised_std_metric[
                                                  'precision'],
                                              np.array([fully_supervised_metric['precision']] * len(prop)) +
                                              fully_supervised_std_metric[
                                                  'precision'],
                                              color=[0, 0, 0, 1], alpha=0.05)
                elif 'F1-score' in label_y:
                    ax[itera, j].errorbar(prop, [fully_supervised_metric['f1-score']] * len(prop),
                                          yerr=[fully_supervised_std_metric['f1-score']] * len(prop),
                                          color=[0, 0, 0, 1],
                                          label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                          capsize=3)
                    ax[itera, j].fill_between(prop,
                                              np.array([fully_supervised_metric['f1-score']] * len(prop)) -
                                              fully_supervised_std_metric[
                                                  'f1-score'],
                                              np.array([fully_supervised_metric['f1-score']] * len(prop)) +
                                              fully_supervised_std_metric[
                                                  'f1-score'],
                                              color=[0, 0, 0, 1], alpha=0.05)
                else:
                    ax[itera, j].errorbar(prop, [fully_supervised_metric['accuracy']] * len(prop),
                                          yerr=[fully_supervised_std_metric['accuracy']] * len(prop),
                                          color=[0, 0, 0, 1],
                                          label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                          capsize=3)
                    ax[itera, j].fill_between(prop,
                                              np.array([fully_supervised_metric['accuracy']] * len(prop)) -
                                              fully_supervised_std_metric[
                                                  'accuracy'],
                                              np.array([fully_supervised_metric['accuracy']] * len(prop)) +
                                              fully_supervised_std_metric[
                                                  'accuracy'],
                                              color=[0, 0, 0, 1], alpha=0.05)

                for i, method in enumerate(methods):
                    if len(metric[i]) == 0:
                        continue
                    if 'FixMatch' in method:
                        linestyle = '-'
                    elif 'Pseudo' in method:
                        linestyle = 'dotted'
                    else:
                        linestyle = '--'

                    if 'Entropy Based' in method:
                        c = colors[3]
                    elif 'MC Dropout' in method:
                        c = colors[1]
                    elif 'Augmentations Based' in method:
                        c = colors[2]
                    elif 'Least Confidence' in method:
                        c = colors[4]
                    elif 'Margin Confidence' in method:
                        c = colors[5]
                    elif 'Learning Loss' in method:
                        c = colors[6]
                    elif 'Badge' in method:
                        c = colors[7]
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
                    ax[itera, j].errorbar(prop, metric[i][1], yerr=(metric[i][0] - metric[i][2]) / 2, color=c,
                                          markersize=10,
                                          label=method, linewidth=2, linestyle=linestyle, marker=marker, capsize=3)
                    ax[itera, j].fill_between(prop, metric[i][0], metric[i][2], color=c, alpha=0.05)

                # ax[itera, j].set_legend(loc='lower right', fontsize=18)
                # plt.title(title, fontsize=30, weight='bold', alpha=.75)
                ax[itera, j].set_xticks(ticks=prop)
                ax[itera, j].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
                ax[itera, j].xaxis.set_ticklabels([])
                ax[itera, j].yaxis.set_ticklabels([])
                # ax[itera, j].savefig(save_path)
        else:
            main(args=arguments)

    ax[2, 0].set_xlabel("Added annotated data (%)", fontsize=30)
    ax[2, 1].set_xlabel("Added annotated data (%)", fontsize=30)
    # ax[2, 2].set_xlabel("Added annotated data (%)", fontsize=30)
    # ax[2, 3].set_xlabel("Added annotated data (%)", fontsize=30)

    ax[0, 0].set_title('Macro Recall', fontsize=30)
    ax[0, 1].set_title('Macro F1-Score', fontsize=30)
    # ax[0, 2].set_title('Macro Recall', fontsize=30)
    # ax[0, 3].set_title('Macro F1-Score', fontsize=30)

    ax[2, 0].set_xticks(ticks=prop)
    ax[2, 1].set_xticks(ticks=prop)
    # ax[2, 2].set_xticks(ticks=prop)
    # ax[2, 3].set_xticks(ticks=prop)

    ax[0, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    ax[1, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    ax[2, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))

    ax[2, 0].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    ax[2, 1].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    # ax[2, 2].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    # ax[2, 3].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])

    ax[0, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))
    ax[1, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))
    ax[2, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))

    fig.subplots_adjust(right=0.8, wspace=0.2, hspace=0.2)

    handles, labels = ax[2, 1].get_legend_handles_labels()
    lgd1 = fig.legend(handles, labels, bbox_to_anchor=(1.141, 0.27))

    handles, labels = ax[1, 1].get_legend_handles_labels()
    lgd2 = fig.legend(handles, labels, bbox_to_anchor=(1.164, 0.55))

    handles, labels = ax[0, 1].get_legend_handles_labels()
    lgd3 = fig.legend(handles, labels, bbox_to_anchor=(1.138, 0.82))

    handles, labels = ax[1, 1].get_legend_handles_labels()
    lgd4 = fig.legend(handles, ["" for lbl in labels], bbox_to_anchor=(1.25, 0.82))

    fig.savefig(f'Fig 2 - {dataset_rep[dataset]}.png', dpi=fig.dpi)
    """
    """
    states = [
        'random_sampling',
        'mc_dropout',
        'entropy_based',
        'augmentations_based',
        'least_confidence',
        'margin_confidence',
        'learning_loss',
        'badge',
        'random_sampling_pretrained',
        'mc_dropout_pretrained',
        'entropy_based_pretrained',
        'augmentations_based_pretrained',
        'least_confidence_pretrained',
        'margin_confidence_pretrained',
        'learning_loss_pretrained',
        'badge_pretrained',
        'auto_encoder',
        'auto_encoder_with_al_mc_dropout',
        'auto_encoder_with_al_entropy_based',
        'auto_encoder_with_al_augmentations_based',
        'auto_encoder_with_al_least_confidence',
        'auto_encoder_with_al_margin_confidence',
        'auto_encoder_with_al_learning_loss',
        'auto_encoder_with_al_badge',
        'simclr',
        'simclr_with_al_mc_dropout',
        'simclr_with_al_entropy_based',
        'simclr_with_al_augmentations_based',
        'simclr_with_al_least_confidence',
        'simclr_with_al_margin_confidence',
        'simclr_with_al_learning_loss',
        'simclr_with_al_badge',
        'fixmatch',
        'fixmatch_with_al_mc_dropout',
        'fixmatch_with_al_entropy_based',
        'fixmatch_with_al_augmentations_based',
        'fixmatch_with_al_least_confidence',
        'fixmatch_with_al_margin_confidence',
        'fixmatch_with_al_learning_loss',
        'fixmatch_with_al_badge',
        'fixmatch_pretrained',
        'fixmatch_with_al_mc_dropout_pretrained_pretrained',
        'fixmatch_with_al_entropy_based_pretrained_pretrained',
        'fixmatch_with_al_augmentations_based_pretrained_pretrained',
        'fixmatch_with_al_least_confidence_pretrained_pretrained',
        'fixmatch_with_al_margin_confidence_pretrained_pretrained',
        'fixmatch_with_al_learning_loss_pretrained_pretrained',
        'fixmatch_with_al_badge_pretrained_pretrained',
        'fixmatch_pretrained_autoencoder',
        'fixmatch_with_al_mc_dropout_pretrained_autoencoder',
        'fixmatch_with_al_entropy_based_pretrained_autoencoder',
        'fixmatch_with_al_augmentations_based_pretrained_autoencoder',
        'fixmatch_with_al_least_confidence_pretrained_autoencoder',
        'fixmatch_with_al_margin_confidence_pretrained_autoencoder',
        'fixmatch_with_al_learning_loss_pretrained_autoencoder',
        'fixmatch_with_al_badge_pretrained_autoencoder',
        'fixmatch_pretrained_simclr',
        'fixmatch_with_al_mc_dropout_pretrained_simclr',
        'fixmatch_with_al_entropy_based_pretrained_simclr',
        'fixmatch_with_al_augmentations_based_pretrained_simclr',
        'fixmatch_with_al_least_confidence_pretrained_simclr',
        'fixmatch_with_al_margin_confidence_pretrained_simclr',
        'fixmatch_with_al_learning_loss_pretrained_simclr',
        'fixmatch_with_al_badge_pretrained_simclr',
        'pseudo_label',
        'pseudo_label_with_al_mc_dropout',
        'pseudo_label_with_al_entropy_based',
        'pseudo_label_with_al_augmentations_based',
        'pseudo_label_with_al_least_confidence',
        'pseudo_label_with_al_margin_confidence',
        'pseudo_label_with_al_learning_loss',
        'pseudo_label_with_al_badge',
        'pseudo_label_pretrained',
        'pseudo_label_with_al_mc_dropout_pretrained_pretrained',
        'pseudo_label_with_al_entropy_based_pretrained_pretrained',
        'pseudo_label_with_al_augmentations_based_pretrained_pretrained',
        'pseudo_label_with_al_least_confidence_pretrained_pretrained',
        'pseudo_label_with_al_margin_confidence_pretrained_pretrained',
        'pseudo_label_with_al_learning_loss_pretrained_pretrained',
        'pseudo_label_with_al_badge_pretrained_pretrained',
        'pseudo_label_pretrained_autoencoder',
        'pseudo_label_with_al_mc_dropout_pretrained_autoencoder',
        'pseudo_label_with_al_entropy_based_pretrained_autoencoder',
        'pseudo_label_with_al_augmentations_based_pretrained_autoencoder',
        'pseudo_label_with_al_least_confidence_pretrained_autoencoder',
        'pseudo_label_with_al_margin_confidence_pretrained_autoencoder',
        'pseudo_label_with_al_learning_loss_pretrained_autoencoder',
        'pseudo_label_with_al_badge_pretrained_autoencoder',
        'pseudo_label_pretrained_simclr',
        'pseudo_label_with_al_mc_dropout_pretrained_simclr',
        'pseudo_label_with_al_entropy_based_pretrained_simclr',
        'pseudo_label_with_al_augmentations_based_pretrained_simclr',
        'pseudo_label_with_al_least_confidence_pretrained_simclr',
        'pseudo_label_with_al_margin_confidence_pretrained_simclr',
        'pseudo_label_with_al_learning_loss_pretrained_simclr',
        'pseudo_label_with_al_badge_pretrained_simclr',
    ]

    datasets = ['isic', 'matek', 'jurkat', 'retinopathy']
    datasets_rep = ['Skin Lesions', 'White blood cells', 'Jurkat cells cycle', 'Retina']
    inits = ['Random', 'ImageNet', 'Autoencoder', 'SimCLR']
    trainings = ['Supervised', 'Semi-supervised (FixMatch)', 'Semi-supervised (Pseudo)']
    uncertainty_samplings = ['Random', 'MC dropout', 'Entropy Based', 'Augmentations Based', 'Least Confidence',
                             'Margin Confidence', 'Learning Loss', 'Badge']
    metrics = ['recall', 'precision', 'f1-score', 'accuracy']
    metrics_rep = ['Recall', 'Precision', 'F1-score', 'Accuracy']

    rows = []
    methods = []
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
                        for iterations in range(len(ratio_metrics_logs[0][1][:9])):
                            row.update({f'{metric_rep} {iterations}': ratio_metrics_logs[0][1][iterations],
                                        f'{metric_rep} STD. {iterations}': ratio_metrics_logs[0][1][iterations] -
                                        ratio_metrics_logs[0][0][iterations]})
                    i += 1
                    if len(ratio_metrics_logs[0]) == 0:
                        continue
                    else:
                        methods.append(states[i-1])
                        rows.append(row)
    import pandas as pd

    df = pd.DataFrame(rows)
    df['Method'] = methods
    df.to_csv('results.csv')
    """
    """
    root_vis = '/home/ahmad/thesis/visualization'
    arguments = get_arguments()
    methods_states = {
        'i': ['Augmentations Based + ImageNet + FixMatch',
              'Augmentations Based + SimCLR + FixMatch',
              'Augmentations Based + ImageNet + Supervised',
              'Augmentations Based + SimCLR + Supervised',
              'Learning Loss + ImageNet + FixMatch',
              'Learning Loss + SimCLR + FixMatch',
              'Learning Loss + ImageNet + Supervised',
              'Learning Loss + SimCLR + Supervised',
              'Badge + ImageNet + FixMatch',
              'Badge + SimCLR + FixMatch',
              'Badge + ImageNet + Supervised',
              'Badge + SimCLR + Supervised',
              'Augmentations Based + SimCLR + Pseudo',
              'Learning Loss + SimCLR + Pseudo',
              'Badge + SimCLR + Pseudo',
              'Random Sampling +  Random + Supervised']
    }
    methods_states_results = {
        'i': ['random_sampling',
              'mc_dropout',
              'entropy_based',
              'augmentations_based',
              'least_confidence',
              'margin_confidence',
              'learning_loss',
              'badge',
              'random_sampling_pretrained',
              'mc_dropout_pretrained',
              'entropy_based_pretrained',
              'augmentations_based_pretrained',
              'least_confidence_pretrained',
              'margin_confidence_pretrained',
              'learning_loss_pretrained',
              'badge_pretrained',
              'auto_encoder',
              'auto_encoder_with_al_mc_dropout',
              'auto_encoder_with_al_entropy_based',
              'auto_encoder_with_al_augmentations_based',
              'auto_encoder_with_al_least_confidence',
              'auto_encoder_with_al_margin_confidence',
              'auto_encoder_with_al_learning_loss',
              'auto_encoder_with_al_badge',
              'simclr',
              'simclr_with_al_mc_dropout',
              'simclr_with_al_entropy_based',
              'simclr_with_al_augmentations_based',
              'simclr_with_al_least_confidence',
              'simclr_with_al_margin_confidence',
              'simclr_with_al_learning_loss',
              'simclr_with_al_badge',
              'fixmatch',
              'fixmatch_with_al_mc_dropout',
              'fixmatch_with_al_entropy_based',
              'fixmatch_with_al_augmentations_based',
              'fixmatch_with_al_least_confidence',
              'fixmatch_with_al_margin_confidence',
              'fixmatch_with_al_learning_loss',
              'fixmatch_with_al_badge',
              'fixmatch_pretrained',
              'fixmatch_with_al_mc_dropout_pretrained_pretrained',
              'fixmatch_with_al_entropy_based_pretrained_pretrained',
              'fixmatch_with_al_augmentations_based_pretrained_pretrained',
              'fixmatch_with_al_least_confidence_pretrained_pretrained',
              'fixmatch_with_al_margin_confidence_pretrained_pretrained',
              'fixmatch_with_al_learning_loss_pretrained_pretrained',
              'fixmatch_with_al_badge_pretrained_pretrained',
              'fixmatch_pretrained_autoencoder',
              'fixmatch_with_al_mc_dropout_pretrained_autoencoder',
              'fixmatch_with_al_entropy_based_pretrained_autoencoder',
              'fixmatch_with_al_augmentations_based_pretrained_autoencoder',
              'fixmatch_with_al_least_confidence_pretrained_autoencoder',
              'fixmatch_with_al_margin_confidence_pretrained_autoencoder',
              'fixmatch_with_al_learning_loss_pretrained_autoencoder',
              'fixmatch_with_al_badge_pretrained_autoencoder',
              'fixmatch_pretrained_simclr',
              'fixmatch_with_al_mc_dropout_pretrained_simclr',
              'fixmatch_with_al_entropy_based_pretrained_simclr',
              'fixmatch_with_al_augmentations_based_pretrained_simclr',
              'fixmatch_with_al_least_confidence_pretrained_simclr',
              'fixmatch_with_al_margin_confidence_pretrained_simclr',
              'fixmatch_with_al_learning_loss_pretrained_simclr',
              'fixmatch_with_al_badge_pretrained_simclr',
              'pseudo_label',
              'pseudo_label_with_al_mc_dropout',
              'pseudo_label_with_al_entropy_based',
              'pseudo_label_with_al_augmentations_based',
              'pseudo_label_with_al_least_confidence',
              'pseudo_label_with_al_margin_confidence',
              'pseudo_label_with_al_learning_loss',
              'pseudo_label_with_al_badge',
              'pseudo_label_pretrained',
              'pseudo_label_with_al_mc_dropout_pretrained_pretrained',
              'pseudo_label_with_al_entropy_based_pretrained_pretrained',
              'pseudo_label_with_al_augmentations_based_pretrained_pretrained',
              'pseudo_label_with_al_least_confidence_pretrained_pretrained',
              'pseudo_label_with_al_margin_confidence_pretrained_pretrained',
              'pseudo_label_with_al_learning_loss_pretrained_pretrained',
              'pseudo_label_with_al_badge_pretrained_pretrained',
              'pseudo_label_pretrained_autoencoder',
              'pseudo_label_with_al_mc_dropout_pretrained_autoencoder',
              'pseudo_label_with_al_entropy_based_pretrained_autoencoder',
              'pseudo_label_with_al_augmentations_based_pretrained_autoencoder',
              'pseudo_label_with_al_least_confidence_pretrained_autoencoder',
              'pseudo_label_with_al_margin_confidence_pretrained_autoencoder',
              'pseudo_label_with_al_learning_loss_pretrained_autoencoder',
              'pseudo_label_with_al_badge_pretrained_autoencoder',
              'pseudo_label_pretrained_simclr',
              'pseudo_label_with_al_mc_dropout_pretrained_simclr',
              'pseudo_label_with_al_entropy_based_pretrained_simclr',
              'pseudo_label_with_al_augmentations_based_pretrained_simclr',
              'pseudo_label_with_al_least_confidence_pretrained_simclr',
              'pseudo_label_with_al_margin_confidence_pretrained_simclr',
              'pseudo_label_with_al_learning_loss_pretrained_simclr',
              'pseudo_label_with_al_badge_pretrained_simclr']
    }

    dataset = 'isic'

    dataset_title = {'matek': 'White blood cells', 'jurkat': 'Jurkat cells cycle',
                     'isic': 'Skin Lesions',
                     'plasmodium': 'Red blood cells', 'retinopathy': 'Retina'}

    import pandas as pd

    df = pd.read_excel('results/results_with_metrics.xlsx')
    df = df[df['Dataset'] == dataset_title[dataset]]
    df = df.sort_values(by='F1-score Avg. Rank')

    top_n_methods = df['Method'][:5].tolist() + ['random_sampling']

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "ultralight"
    plt.rcParams["axes.labelweight"] = "ultralight"
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rcParams['legend.fontsize'] = 25

    fig, ax = plt.subplots(1, 4, figsize=(50, 10))
    fig.suptitle(dataset_rep[dataset], fontsize=45)
    fig.subplots_adjust(top=0.8)

    for itera, (k, method_state) in enumerate(methods_states_results.items()):
        # print('**************************' + str(itera))
        if arguments.run_batch:
            states = [
                (dataset, 'f1-score', 'macro avg'),
                (dataset, 'recall', 'macro avg'),
                (dataset, 'precision', 'macro avg'),
                (dataset, 'recall', 'accuracy'),
            ]

            for j, (d, m, r) in enumerate(states):
                root_path = os.path.join(root_vis, d, k)
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                arguments.dataset = d
                arguments.metric = m
                arguments.metric_ratio = r
                # arguments.methods_default = method_state
                arguments.methods_default_results = method_state
                arguments.save_path = os.path.join(root_path, f'{m}_{r}.png')
                args = configs[arguments.dataset](arguments)

                num = [i for i in range(0, 41, 5)]

                if 'macro' in args.metric_ratio:
                    y_label = f'Macro {args.metric.capitalize()}'
                else:
                    y_label = 'Accuracy'

                dataset_title = {'matek': 'White blood cells', 'jurkat': 'Jurkat cell cycle',
                                 'isic': 'Skin lesions',
                                 'plasmodium': 'Red blood cells', 'retinopathy': 'Retina'}

                '''
                ratio_class_wise_metrics_log = ratio_class_wise_metrics(args.metric, dataset.classes, args.dataset)
                plot_ratio_class_wise_metrics(ratio_class_wise_metrics_log, dataset.classes, y_label, num,
                                              plot_configs[args.dataset])
                '''

                ratio_metrics_logs = ratio_metrics(args.metric, args.dataset, cls=args.metric_ratio,
                                                   methods=args.methods_default_results)

                prop = num[:9]
                metric = ratio_metrics_logs
                label_y = y_label
                fully_supervised_metric = fully_supervised[args.dataset]
                save_path = args.save_path
                methods = args.methods_default_results
                title = dataset_title[args.dataset]
                fully_supervised_std_metric = fully_supervised_std[args.dataset]

                # plt.figure(figsize=(14, 10))
                # plt.grid(color='black')
                style.use(['science', 'no-latex'])

                colors = [[86 / 255, 180 / 255, 233 / 255, 1], [230 / 255, 159 / 255, 0, 1],
                          [212 / 255, 16 / 255, 16 / 255, 1],
                          [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
                          [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1],
                          [211 / 255, 95 / 255, 183 / 255, 1],
                          [238 / 255, 136 / 255, 102 / 255, 1]]

                top_n_methods_index = []

                for i, method in enumerate(methods):
                    if len(metric[i]) == 0:
                        continue
                    if 'fixmatch' in method:
                        linestyle = '-'
                    elif 'pseudo_label' in method:
                        linestyle = 'dotted'
                    else:
                        linestyle = '--'

                    if method in top_n_methods:
                        top_n_methods_index.append(i)
                        continue
                    c = 'lightgrey'
                    if 'simclr' in method:
                        marker = 's'
                    elif 'autoencoder' in method:
                        marker = 'o'
                    elif '_pretrained' in method:
                        marker = '^'
                    else:
                        marker = ','
                    ax[j].errorbar(prop, metric[i][1], yerr=(metric[i][0] - metric[i][2]) / 2, color=c,
                                   markersize=10,
                                   label=method, linewidth=2, linestyle=linestyle, marker=marker, capsize=3)
                    ax[j].fill_between(prop, metric[i][0], metric[i][2], color=c, alpha=0.05)

                for i in top_n_methods_index:
                    method = methods[i]
                    if len(metric[i]) == 0:
                        continue
                    if 'fixmatch' in method:
                        linestyle = '-'
                    elif 'pseudo_label' in method:
                        linestyle = '--'
                    else:
                        linestyle = 'dotted'

                    if 'entropy_based' in method:
                        c = colors[3]
                    elif 'mc_dropout' in method:
                        c = colors[1]
                    elif 'augmentations_based' in method:
                        c = colors[2]
                    elif 'least_confidence' in method:
                        c = colors[4]
                    elif 'margin_confidence' in method:
                        c = colors[5]
                    elif 'learning_loss' in method:
                        c = colors[6]
                    elif 'badge' in method:
                        c = colors[7]
                    else:
                        c = colors[0]

                    if 'simclr' in method:
                        marker = 's'
                    elif 'autoencoder' in method or 'auto_encoder' in method:
                        marker = 'o'
                    elif '_pretrained' in method:
                        marker = '^'
                    else:
                        marker = ','
                    ax[j].errorbar(prop, metric[i][1], yerr=(metric[i][0] - metric[i][2]) / 2, color=c,
                                   markersize=10,
                                   label=method, linewidth=2, linestyle=linestyle, marker=marker, capsize=3)
                    ax[j].fill_between(prop, metric[i][0], metric[i][2], color=c, alpha=0.05, zorder=300)

                if 'Recall' in label_y:
                    ax[j].errorbar(prop, [fully_supervised_metric['recall']] * len(prop),
                                   yerr=[fully_supervised_std_metric['recall']] * len(prop),
                                   color=[0, 0, 0, 1], label='Fully Supervised', linewidth=2, linestyle='--',
                                   marker=',',
                                   capsize=3)
                    ax[j].fill_between(prop,
                                       np.array([fully_supervised_metric['recall']] * len(prop)) -
                                       fully_supervised_std_metric[
                                           'recall'],
                                       np.array([fully_supervised_metric['recall']] * len(prop)) +
                                       fully_supervised_std_metric[
                                           'recall'],
                                       color=[0, 0, 0, 1], alpha=0.05)
                elif 'Precision' in label_y:
                    ax[j].errorbar(prop, [fully_supervised_metric['precision']] * len(prop),
                                   yerr=[fully_supervised_std_metric['precision']] * len(prop),
                                   color=[0, 0, 0, 1],
                                   label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                   capsize=3)
                    ax[j].fill_between(prop,
                                       np.array([fully_supervised_metric['precision']] * len(prop)) -
                                       fully_supervised_std_metric[
                                           'precision'],
                                       np.array([fully_supervised_metric['precision']] * len(prop)) +
                                       fully_supervised_std_metric[
                                           'precision'],
                                       color=[0, 0, 0, 1], alpha=0.05)
                elif 'F1-score' in label_y:
                    ax[j].errorbar(prop, [fully_supervised_metric['f1-score']] * len(prop),
                                   yerr=[fully_supervised_std_metric['f1-score']] * len(prop),
                                   color=[0, 0, 0, 1],
                                   label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                   capsize=3)
                    ax[j].fill_between(prop,
                                       np.array([fully_supervised_metric['f1-score']] * len(prop)) -
                                       fully_supervised_std_metric[
                                           'f1-score'],
                                       np.array([fully_supervised_metric['f1-score']] * len(prop)) +
                                       fully_supervised_std_metric[
                                           'f1-score'],
                                       color=[0, 0, 0, 1], alpha=0.05)
                else:
                    ax[j].errorbar(prop, [fully_supervised_metric['accuracy']] * len(prop),
                                   yerr=[fully_supervised_std_metric['accuracy']] * len(prop),
                                   color=[0, 0, 0, 1],
                                   label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                   capsize=3)
                    ax[j].fill_between(prop,
                                       np.array([fully_supervised_metric['accuracy']] * len(prop)) -
                                       fully_supervised_std_metric[
                                           'accuracy'],
                                       np.array([fully_supervised_metric['accuracy']] * len(prop)) +
                                       fully_supervised_std_metric[
                                           'accuracy'],
                                       color=[0, 0, 0, 1], alpha=0.05)

                # ax[itera, j].set_legend(loc='lower right', fontsize=18)
                # plt.title(title, fontsize=30, weight='bold', alpha=.75)
                ax[j].set_xticks(ticks=prop)
                ax[j].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
                ax[j].xaxis.set_ticklabels([])
                ax[j].yaxis.set_ticklabels([])
                # ax[itera, j].savefig(save_path)
        else:
            main(args=arguments)

    ax[0].set_xlabel("Added annotated data (%)", fontsize=30)
    ax[1].set_xlabel("Added annotated data (%)", fontsize=30)
    ax[2].set_xlabel("Added annotated data (%)", fontsize=30)
    ax[3].set_xlabel("Added annotated data (%)", fontsize=30)
    # ax[2, 2].set_xlabel("Added annotated data (%)", fontsize=30)
    # ax[2, 3].set_xlabel("Added annotated data (%)", fontsize=30)

    ax[0].set_title('Macro F1-Score', fontsize=30)
    ax[1].set_title('Macro Recall', fontsize=30)
    ax[2].set_title('Macro Precision', fontsize=30)
    ax[3].set_title('Accuracy', fontsize=30)
    # ax[0, 2].set_title('Macro Recall', fontsize=30)
    # ax[0, 3].set_title('Macro F1-Score', fontsize=30)

    ax[0].set_xticks(ticks=prop)
    ax[1].set_xticks(ticks=prop)
    ax[2].set_xticks(ticks=prop)
    ax[3].set_xticks(ticks=prop)
    # ax[2, 2].set_xticks(ticks=prop)
    # ax[2, 3].set_xticks(ticks=prop)

    ax[0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    ax[1].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    ax[2].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    ax[3].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    # ax[1, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
    # ax[2, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))

    ax[0].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    ax[1].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    ax[2].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    ax[3].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    # ax[2, 2].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
    # ax[2, 3].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])

    ax[0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))
    # ax[1, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))
    # ax[2, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))

    ax[0].set_ylim(0.1, 1.0)
    ax[1].set_ylim(0.1, 1.0)
    ax[2].set_ylim(0.1, 1.0)
    ax[3].set_ylim(0.1, 1.0)

    # fig.subplots_adjust(right=0.8, wspace=0.2, hspace=0.2)

    # handles, labels = ax[2, 1].get_legend_handles_labels()
    # lgd1 = fig.legend(handles, labels, bbox_to_anchor=(1.141, 0.27))

    # handles, labels = ax[1, 1].get_legend_handles_labels()
    # lgd2 = fig.legend(handles, labels, bbox_to_anchor=(1.164, 0.55))

    # handles, labels = ax[1].get_legend_handles_labels()
    # lgd3 = fig.legend(handles, labels, bbox_to_anchor=(1.138, 0.82))

    # handles, labels = ax[1, 1].get_legend_handles_labels()
    # lgd4 = fig.legend(handles, ["" for lbl in labels], bbox_to_anchor=(1.25, 0.82))

    fig.savefig(f'results/{dataset_rep[dataset]}.png', dpi=fig.dpi)
    """

    root_vis = '/home/ahmad/thesis/visualization'
    arguments = get_arguments()
    methods_states = {
        'i': ['Augmentations Based + ImageNet + FixMatch',
              'Augmentations Based + SimCLR + FixMatch',
              'Augmentations Based + ImageNet + Supervised',
              'Augmentations Based + SimCLR + Supervised',
              'Learning Loss + ImageNet + FixMatch',
              'Learning Loss + SimCLR + FixMatch',
              'Learning Loss + ImageNet + Supervised',
              'Learning Loss + SimCLR + Supervised',
              'Badge + ImageNet + FixMatch',
              'Badge + SimCLR + FixMatch',
              'Badge + ImageNet + Supervised',
              'Badge + SimCLR + Supervised',
              'Augmentations Based + SimCLR + Pseudo',
              'Learning Loss + SimCLR + Pseudo',
              'Badge + SimCLR + Pseudo',
              'Random Sampling +  Random + Supervised']
    }
    methods_states_results = {
        'i': ['random_sampling',
              'mc_dropout',
              'entropy_based',
              'augmentations_based',
              'least_confidence',
              'margin_confidence',
              'learning_loss',
              'badge',
              'random_sampling_pretrained',
              'mc_dropout_pretrained',
              'entropy_based_pretrained',
              'augmentations_based_pretrained',
              'least_confidence_pretrained',
              'margin_confidence_pretrained',
              'learning_loss_pretrained',
              'badge_pretrained',
              'auto_encoder',
              'auto_encoder_with_al_mc_dropout',
              'auto_encoder_with_al_entropy_based',
              'auto_encoder_with_al_augmentations_based',
              'auto_encoder_with_al_least_confidence',
              'auto_encoder_with_al_margin_confidence',
              'auto_encoder_with_al_learning_loss',
              'auto_encoder_with_al_badge',
              'simclr',
              'simclr_with_al_mc_dropout',
              'simclr_with_al_entropy_based',
              'simclr_with_al_augmentations_based',
              'simclr_with_al_least_confidence',
              'simclr_with_al_margin_confidence',
              'simclr_with_al_learning_loss',
              'simclr_with_al_badge',
              'fixmatch',
              'fixmatch_with_al_mc_dropout',
              'fixmatch_with_al_entropy_based',
              'fixmatch_with_al_augmentations_based',
              'fixmatch_with_al_least_confidence',
              'fixmatch_with_al_margin_confidence',
              'fixmatch_with_al_learning_loss',
              'fixmatch_with_al_badge',
              'fixmatch_pretrained',
              'fixmatch_with_al_mc_dropout_pretrained_pretrained',
              'fixmatch_with_al_entropy_based_pretrained_pretrained',
              'fixmatch_with_al_augmentations_based_pretrained_pretrained',
              'fixmatch_with_al_least_confidence_pretrained_pretrained',
              'fixmatch_with_al_margin_confidence_pretrained_pretrained',
              'fixmatch_with_al_learning_loss_pretrained_pretrained',
              'fixmatch_with_al_badge_pretrained_pretrained',
              'fixmatch_pretrained_autoencoder',
              'fixmatch_with_al_mc_dropout_pretrained_autoencoder',
              'fixmatch_with_al_entropy_based_pretrained_autoencoder',
              'fixmatch_with_al_augmentations_based_pretrained_autoencoder',
              'fixmatch_with_al_least_confidence_pretrained_autoencoder',
              'fixmatch_with_al_margin_confidence_pretrained_autoencoder',
              'fixmatch_with_al_learning_loss_pretrained_autoencoder',
              'fixmatch_with_al_badge_pretrained_autoencoder',
              'fixmatch_pretrained_simclr',
              'fixmatch_with_al_mc_dropout_pretrained_simclr',
              'fixmatch_with_al_entropy_based_pretrained_simclr',
              'fixmatch_with_al_augmentations_based_pretrained_simclr',
              'fixmatch_with_al_least_confidence_pretrained_simclr',
              'fixmatch_with_al_margin_confidence_pretrained_simclr',
              'fixmatch_with_al_learning_loss_pretrained_simclr',
              'fixmatch_with_al_badge_pretrained_simclr',
              'pseudo_label',
              'pseudo_label_with_al_mc_dropout',
              'pseudo_label_with_al_entropy_based',
              'pseudo_label_with_al_augmentations_based',
              'pseudo_label_with_al_least_confidence',
              'pseudo_label_with_al_margin_confidence',
              'pseudo_label_with_al_learning_loss',
              'pseudo_label_with_al_badge',
              'pseudo_label_pretrained',
              'pseudo_label_with_al_mc_dropout_pretrained_pretrained',
              'pseudo_label_with_al_entropy_based_pretrained_pretrained',
              'pseudo_label_with_al_augmentations_based_pretrained_pretrained',
              'pseudo_label_with_al_least_confidence_pretrained_pretrained',
              'pseudo_label_with_al_margin_confidence_pretrained_pretrained',
              'pseudo_label_with_al_learning_loss_pretrained_pretrained',
              'pseudo_label_with_al_badge_pretrained_pretrained',
              'pseudo_label_pretrained_autoencoder',
              'pseudo_label_with_al_mc_dropout_pretrained_autoencoder',
              'pseudo_label_with_al_entropy_based_pretrained_autoencoder',
              'pseudo_label_with_al_augmentations_based_pretrained_autoencoder',
              'pseudo_label_with_al_least_confidence_pretrained_autoencoder',
              'pseudo_label_with_al_margin_confidence_pretrained_autoencoder',
              'pseudo_label_with_al_learning_loss_pretrained_autoencoder',
              'pseudo_label_with_al_badge_pretrained_autoencoder',
              'pseudo_label_pretrained_simclr',
              'pseudo_label_with_al_mc_dropout_pretrained_simclr',
              'pseudo_label_with_al_entropy_based_pretrained_simclr',
              'pseudo_label_with_al_augmentations_based_pretrained_simclr',
              'pseudo_label_with_al_least_confidence_pretrained_simclr',
              'pseudo_label_with_al_margin_confidence_pretrained_simclr',
              'pseudo_label_with_al_learning_loss_pretrained_simclr',
              'pseudo_label_with_al_badge_pretrained_simclr']
    }

    elements = ['pseudo_label', 'fixmatch', 'auto_encoder', 'simclr', 'random_sampling', 'mc_dropout',
                'entropy_based', 'augmentations_based', 'least_confidence', 'margin_confidence', 'learning_loss',
                'badge', 'pretrained']

    elements_rep = ['Pseudo Label', 'FixMatch', 'Autoencoder', 'SimCLR', 'Random Sampling', 'MC Dropout',
                    'Entropy Based', 'Augmentations Based', 'Least Confidence', 'Margin Confidence', 'Learning Loss',
                    'Badge', 'ImageNet']

    for element_rep in elements_rep:
        if not os.path.exists(f'results/{element_rep}'):
            os.mkdir(f'results/{element_rep}')

    datasets = ['retinopathy', 'isic', 'matek', 'jurkat']

    dataset_title = {'matek': 'White blood cells', 'jurkat': 'Jurkat cells cycle',
                     'isic': 'Skin Lesions',
                     'plasmodium': 'Red blood cells', 'retinopathy': 'Retina'}

    for dataset in datasets:

        for e, element in enumerate(elements):
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.weight"] = "ultralight"
            plt.rcParams["axes.labelweight"] = "ultralight"
            plt.rc('xtick', labelsize=30)
            plt.rc('ytick', labelsize=30)
            plt.rcParams['legend.fontsize'] = 25

            fig, ax = plt.subplots(1, 4, figsize=(40, 10))
            fig.suptitle(dataset_rep[dataset], fontsize=45)
            fig.subplots_adjust(top=0.8)
            
            for itera, (k, method_state) in enumerate(methods_states_results.items()):
                # print('**************************' + str(itera))
                if arguments.run_batch:
                    states = [
                        (dataset, 'f1-score', 'macro avg'),
                        (dataset, 'recall', 'macro avg'),
                        (dataset, 'precision', 'macro avg'),
                        (dataset, 'recall', 'accuracy'),
                    ]

                    for j, (d, m, r) in enumerate(states):
                        root_path = os.path.join(root_vis, d, k)
                        if not os.path.exists(root_path):
                            os.makedirs(root_path)
                        arguments.dataset = d
                        arguments.metric = m
                        arguments.metric_ratio = r
                        # arguments.methods_default = method_state
                        arguments.methods_default_results = method_state
                        arguments.save_path = os.path.join(root_path, f'{m}_{r}.png')
                        args = configs[arguments.dataset](arguments)

                        num = [i for i in range(0, 41, 5)]

                        if 'macro' in args.metric_ratio:
                            y_label = f'Macro {args.metric.capitalize()}'
                        else:
                            y_label = 'Accuracy'

                        dataset_title = {'matek': 'White blood cells', 'jurkat': 'Jurkat cell cycle',
                                         'isic': 'Skin lesions',
                                         'plasmodium': 'Red blood cells', 'retinopathy': 'Retina'}

                        '''
                        ratio_class_wise_metrics_log = ratio_class_wise_metrics(args.metric, dataset.classes, args.dataset)
                        plot_ratio_class_wise_metrics(ratio_class_wise_metrics_log, dataset.classes, y_label, num,
                                                      plot_configs[args.dataset])
                        '''

                        ratio_metrics_logs = ratio_metrics(args.metric, args.dataset, cls=args.metric_ratio,
                                                           methods=args.methods_default_results)

                        prop = num[:9]
                        metric = ratio_metrics_logs
                        label_y = y_label
                        fully_supervised_metric = fully_supervised[args.dataset]
                        save_path = args.save_path
                        methods = args.methods_default_results
                        title = dataset_title[args.dataset]
                        fully_supervised_std_metric = fully_supervised_std[args.dataset]

                        # plt.figure(figsize=(14, 10))
                        # plt.grid(color='black')
                        style.use(['science', 'no-latex'])

                        colors = [[86 / 255, 180 / 255, 233 / 255, 1], [230 / 255, 159 / 255, 0, 1],
                                  [212 / 255, 16 / 255, 16 / 255, 1],
                                  [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
                                  [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1],
                                  [211 / 255, 95 / 255, 183 / 255, 1],
                                  [238 / 255, 136 / 255, 102 / 255, 1]]

                        for i, method in enumerate(methods):

                            random = False

                            if ('fixmatch' in method or 'pseudo_label' in method) and 'auto_encoder' == element:
                                element = 'autoencoder'
                            elif ('fixmatch' not in method and 'pseudo_label' not in method) and 'auto' in element:
                                element = 'auto_encoder'
                            if ('fixmatch' in method or 'pseudo_label' in method) and 'pretrained' == element \
                                    and 'with_al' in method:
                                element = 'pretrained_pretrained'
                            elif ('fixmatch' not in method and 'pseudo_label' not in method) and 'pretrained' in element:
                                element = 'pretrained'

                            if ('fixmatch' in method or 'pseudo_label' in method or 'simclr' in method
                               or 'auto_encoder' in method) and 'with_al' not in method and 'random_sampling' == element:
                                random = True

                            if element not in method and not random:
                                continue

                            if len(metric[i]) == 0:
                                continue
                            if 'fixmatch' in method:
                                linestyle = '-'
                            elif 'pseudo_label' in method:
                                linestyle = '--'
                            else:
                                linestyle = 'dotted'

                            if 'entropy_based' in method:
                                c = colors[3]
                            elif 'mc_dropout' in method:
                                c = colors[1]
                            elif 'augmentations_based' in method:
                                c = colors[2]
                            elif 'least_confidence' in method:
                                c = colors[4]
                            elif 'margin_confidence' in method:
                                c = colors[5]
                            elif 'learning_loss' in method:
                                c = colors[6]
                            elif 'badge' in method:
                                c = colors[7]
                            else:
                                c = colors[0]

                            if 'simclr' in method:
                                marker = 's'
                            elif 'autoencoder' in method or 'auto_encoder' in method:
                                marker = 'o'
                            elif '_pretrained' in method:
                                marker = '^'
                            else:
                                marker = ','
                            ax[j].errorbar(prop, metric[i][1], yerr=(metric[i][0] - metric[i][2]) / 2, color=c,
                                           markersize=10,
                                           label=method, linewidth=2, linestyle=linestyle, marker=marker, capsize=3)
                            ax[j].fill_between(prop, metric[i][0], metric[i][2], color=c, alpha=0.05)

                        if 'Recall' in label_y:
                            ax[j].errorbar(prop, [fully_supervised_metric['recall']] * len(prop),
                                           yerr=[fully_supervised_std_metric['recall']] * len(prop),
                                           color=[0, 0, 0, 1], label='Fully Supervised', linewidth=2, linestyle='--',
                                           marker=',',
                                           capsize=3)
                            ax[j].fill_between(prop,
                                               np.array([fully_supervised_metric['recall']] * len(prop)) -
                                               fully_supervised_std_metric[
                                                   'recall'],
                                               np.array([fully_supervised_metric['recall']] * len(prop)) +
                                               fully_supervised_std_metric[
                                                   'recall'],
                                               color=[0, 0, 0, 1], alpha=0.05)
                        elif 'Precision' in label_y:
                            ax[j].errorbar(prop, [fully_supervised_metric['precision']] * len(prop),
                                           yerr=[fully_supervised_std_metric['precision']] * len(prop),
                                           color=[0, 0, 0, 1],
                                           label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                           capsize=3)
                            ax[j].fill_between(prop,
                                               np.array([fully_supervised_metric['precision']] * len(prop)) -
                                               fully_supervised_std_metric[
                                                   'precision'],
                                               np.array([fully_supervised_metric['precision']] * len(prop)) +
                                               fully_supervised_std_metric[
                                                   'precision'],
                                               color=[0, 0, 0, 1], alpha=0.05)
                        elif 'F1-score' in label_y:
                            ax[j].errorbar(prop, [fully_supervised_metric['f1-score']] * len(prop),
                                           yerr=[fully_supervised_std_metric['f1-score']] * len(prop),
                                           color=[0, 0, 0, 1],
                                           label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                           capsize=3)
                            ax[j].fill_between(prop,
                                               np.array([fully_supervised_metric['f1-score']] * len(prop)) -
                                               fully_supervised_std_metric[
                                                   'f1-score'],
                                               np.array([fully_supervised_metric['f1-score']] * len(prop)) +
                                               fully_supervised_std_metric[
                                                   'f1-score'],
                                               color=[0, 0, 0, 1], alpha=0.05)
                        else:
                            ax[j].errorbar(prop, [fully_supervised_metric['accuracy']] * len(prop),
                                           yerr=[fully_supervised_std_metric['accuracy']] * len(prop),
                                           color=[0, 0, 0, 1],
                                           label='Fully Supervised', linewidth=2, linestyle='--', marker=',',
                                           capsize=3)
                            ax[j].fill_between(prop,
                                               np.array([fully_supervised_metric['accuracy']] * len(prop)) -
                                               fully_supervised_std_metric[
                                                   'accuracy'],
                                               np.array([fully_supervised_metric['accuracy']] * len(prop)) +
                                               fully_supervised_std_metric[
                                                   'accuracy'],
                                               color=[0, 0, 0, 1], alpha=0.05)

                        # ax[itera, j].set_legend(loc='lower right', fontsize=18)
                        # plt.title(title, fontsize=30, weight='bold', alpha=.75)
                        ax[j].set_xticks(ticks=prop)
                        ax[j].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
                        ax[j].xaxis.set_ticklabels([])
                        ax[j].yaxis.set_ticklabels([])
                        # ax[itera, j].savefig(save_path)
                else:
                    main(args=arguments)

            ax[0].set_xlabel("Added annotated data (%)", fontsize=30)
            ax[1].set_xlabel("Added annotated data (%)", fontsize=30)
            ax[2].set_xlabel("Added annotated data (%)", fontsize=30)
            ax[3].set_xlabel("Added annotated data (%)", fontsize=30)
            # ax[2, 2].set_xlabel("Added annotated data (%)", fontsize=30)
            # ax[2, 3].set_xlabel("Added annotated data (%)", fontsize=30)

            ax[0].set_title('Macro F1-Score', fontsize=30)
            ax[1].set_title('Macro Recall', fontsize=30)
            ax[2].set_title('Macro Precision', fontsize=30)
            ax[3].set_title('Accuracy', fontsize=30)
            # ax[0, 2].set_title('Macro Recall', fontsize=30)
            # ax[0, 3].set_title('Macro F1-Score', fontsize=30)

            ax[0].set_xticks(ticks=prop)
            ax[1].set_xticks(ticks=prop)
            ax[2].set_xticks(ticks=prop)
            ax[3].set_xticks(ticks=prop)
            # ax[2, 2].set_xticks(ticks=prop)
            # ax[2, 3].set_xticks(ticks=prop)

            ax[0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
            # ax[1, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))
            # ax[2, 0].set_yticks(ticks=np.arange(0.10, 1.0, step=0.10))

            ax[0].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
            ax[1].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
            ax[2].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
            ax[3].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
            # ax[2, 2].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])
            # ax[2, 3].xaxis.set_ticklabels([str(pr)[:3] for pr in prop])

            ax[0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))
            # ax[1, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))
            # ax[2, 0].yaxis.set_ticklabels(np.round(np.arange(0.10, 1.0, step=0.10), decimals=1))

            ax[0].set_ylim(0.1, 1.0)
            ax[1].set_ylim(0.1, 1.0)
            ax[2].set_ylim(0.1, 1.0)
            ax[3].set_ylim(0.1, 1.0)

            # fig.subplots_adjust(right=0.8, wspace=0.2, hspace=0.2)

            # handles, labels = ax[2, 1].get_legend_handles_labels()
            # lgd1 = fig.legend(handles, labels, bbox_to_anchor=(1.141, 0.27))

            # handles, labels = ax[1, 1].get_legend_handles_labels()
            # lgd2 = fig.legend(handles, labels, bbox_to_anchor=(1.164, 0.55))

            # handles, labels = ax[1].get_legend_handles_labels()
            # lgd3 = fig.legend(handles, labels, bbox_to_anchor=(1.138, 0.82))

            # handles, labels = ax[1, 1].get_legend_handles_labels()
            # lgd4 = fig.legend(handles, ["" for lbl in labels], bbox_to_anchor=(1.25, 0.82))

            fig.savefig(f'results/{elements_rep[e]}/{dataset_rep[dataset]}.png', dpi=fig.dpi)

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
"""
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
"""
