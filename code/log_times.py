from options.train_options import get_arguments
from utils import set_model_name
import os
import pathlib
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('fivethirtyeight')

root = '/home/ahmad/thesis/med_active_learning/logs_final'
arguments = get_arguments()
files = os.listdir(root)
seeds = ['9999', '5555', '2323', '6666']

prev_time = {}
prev_time.update({seed: 0 for seed in seeds})

dic_times = {}


def main(args):
    global prev_time, dic_times
    args.dataset = 'matek'
    name = set_model_name(args)

    dic_times.update({name: []})

    for seed in seeds:
        for file in files:
            if 'class-num' in file or 'epoch' in file:
                continue
            file_info = pathlib.Path(os.path.join(root, file))
            mod_file = f"{file.split('-')[1]}-{file.split('-')[2]}"

            if mod_file == f"{name}-seed:{seed}":
                if prev_time[seed] == 0:
                    prev_time.update({seed: file_info.stat().st_mtime})
                else:
                    dic_times[name].append(file_info.stat().st_mtime - prev_time[seed])
                    prev_time.update({seed: file_info.stat().st_mtime})


def find_median():
    global dic_times

    secs = []
    labels = []

    for k, v in dic_times.items():
        if len(v) == 0:
            continue
        print(f"{k} -- {datetime.timedelta(seconds=np.median(v))}")
        labels.append(k.split('matek@resnet@')[-1])
        secs.append(np.median(v))

    fig, ax = plt.subplots(figsize=(20, 10))

    # Save the chart so we can loop through the bars below.
    bars = ax.bar(
        x=np.arange(len(labels)),
        height=secs,
        tick_label=labels
    )

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False, labelrotation=90.0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 270000)

    # Add text annotations to the top of the bars.
    bar_color = bars[0].get_facecolor()
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=bar_color,
            weight='bold',
            rotation=90.0
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            270000,
            labels[i],
            horizontalalignment='center',
            color=bar_color,
            weight='bold',
            rotation=90.0
        )

    # Add labels and a title.
    ax.set_xlabel('Methods', labelpad=15, color='#333333')
    ax.set_ylabel('Seconds', labelpad=15, color='#333333')
    ax.set_title('Time Elapsed Matek', pad=15, color='#333333',
                 weight='bold')

    # fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    states = [
        ('active_learning', 'entropy_based', None, None, False, None),
        ('active_learning', 'mc_dropout', None, None, False, None),
        ('active_learning', 'augmentations_based', None, None, False, None),
        ('active_learning', 'least_confidence', None, None, False, None),
        ('active_learning', 'margin_confidence', None, None, False, None),
        ('active_learning', 'learning_loss', None, None, False, None),
        ('random_sampling', None, None, None, False, None),
        ('semi_supervised', None, 'simclr', None, False, None),
        ('semi_supervised', None, 'simclr_with_al', 'augmentations_based', False, None),
        ('semi_supervised', None, 'simclr_with_al', 'entropy_based', False, None),
        ('semi_supervised', None, 'simclr_with_al', 'mc_dropout', False, None),
        ('semi_supervised', None, 'simclr_with_al', 'least_confidence', False, None),
        ('semi_supervised', None, 'simclr_with_al', 'margin_confidence', False, None),
        ('semi_supervised', None, 'simclr_with_al', 'learning_loss', False, None),
        ('semi_supervised', None, 'auto_encoder', None, False, None),
        ('semi_supervised', None, 'auto_encoder_with_al', 'augmentations_based', False, None),
        ('semi_supervised', None, 'auto_encoder_with_al', 'entropy_based', False, None),
        ('semi_supervised', None, 'auto_encoder_with_al', 'mc_dropout', False, None),
        ('semi_supervised', None, 'auto_encoder_with_al', 'least_confidence', False, None),
        ('semi_supervised', None, 'auto_encoder_with_al', 'margin_confidence', False, None),
        ('semi_supervised', None, 'auto_encoder_with_al', 'learning_loss', False, None),
        ('semi_supervised', None, 'fixmatch', None, False, None),
        ('semi_supervised', None, 'fixmatch_with_al', 'augmentations_based', False, None),
        ('semi_supervised', None, 'fixmatch_with_al', 'entropy_based', False, None),
        ('semi_supervised', None, 'fixmatch_with_al', 'mc_dropout', False, None),
        ('semi_supervised', None, 'fixmatch_with_al', 'least_confidence', False, None),
        ('semi_supervised', None, 'fixmatch_with_al', 'margin_confidence', False, None),
        ('semi_supervised', None, 'fixmatch_with_al', 'learning_loss', False, None),
        ('semi_supervised', None, 'pseudo_label', None, False, None),
        ('semi_supervised', None, 'pseudo_label_with_al', 'augmentations_based', False, None),
        ('semi_supervised', None, 'pseudo_label_with_al', 'entropy_based', False, None),
        ('semi_supervised', None, 'pseudo_label_with_al', 'mc_dropout', False, None),
        ('semi_supervised', None, 'pseudo_label_with_al', 'least_confidence', False, None),
        ('semi_supervised', None, 'pseudo_label_with_al', 'margin_confidence', False, None),
        ('semi_supervised', None, 'pseudo_label_with_al', 'learning_loss', False, None),
        ('active_learning', 'entropy_based', None, None, True, None),
        ('active_learning', 'mc_dropout', None, None, True, None),
        ('active_learning', 'augmentations_based', None, None, True, None),
        ('active_learning', 'least_confidence', None, None, True, None),
        ('active_learning', 'margin_confidence', None, None, True, None),
        ('active_learning', 'learning_loss', None, None, True, None),
        ('random_sampling', None, None, None, True, None),
        ('semi_supervised', None, 'fixmatch', None, True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch_with_al', 'augmentations_based', True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch_with_al', 'entropy_based', True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch_with_al', 'mc_dropout', True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch_with_al', 'least_confidence', True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch_with_al', 'margin_confidence', True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch_with_al', 'learning_loss', True, 'pretrained'),
        ('semi_supervised', None, 'fixmatch', None, True, 'simclr'),
        ('semi_supervised', None, 'fixmatch_with_al', 'augmentations_based', True, 'simclr'),
        ('semi_supervised', None, 'fixmatch_with_al', 'entropy_based', True, 'simclr'),
        ('semi_supervised', None, 'fixmatch_with_al', 'mc_dropout', True, 'simclr'),
        ('semi_supervised', None, 'fixmatch_with_al', 'least_confidence', True, 'simclr'),
        ('semi_supervised', None, 'fixmatch_with_al', 'margin_confidence', True, 'simclr'),
        ('semi_supervised', None, 'fixmatch_with_al', 'learning_loss', True, 'simclr'),
        ('semi_supervised', None, 'fixmatch', None, True, 'autoencoder'),
        ('semi_supervised', None, 'fixmatch_with_al', 'augmentations_based', True, 'autoencoder'),
        ('semi_supervised', None, 'fixmatch_with_al', 'entropy_based', True, 'autoencoder'),
        ('semi_supervised', None, 'fixmatch_with_al', 'mc_dropout', True, 'autoencoder'),
        ('semi_supervised', None, 'fixmatch_with_al', 'least_confidence', True, 'autoencoder'),
        ('semi_supervised', None, 'fixmatch_with_al', 'margin_confidence', True, 'autoencoder'),
        ('semi_supervised', None, 'fixmatch_with_al', 'learning_loss', True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label', None, True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'augmentations_based', True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'entropy_based', True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'mc_dropout', True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'least_confidence', True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'margin_confidence', True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'learning_loss', True, 'pretrained'),
        ('semi_supervised', None, 'pseudo_label', None, True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'augmentations_based', True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'entropy_based', True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'mc_dropout', True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'least_confidence', True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'margin_confidence', True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'learning_loss', True, 'simclr'),
        ('semi_supervised', None, 'pseudo_label', None, True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'augmentations_based', True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'entropy_based', True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'mc_dropout', True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'least_confidence', True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'margin_confidence', True, 'autoencoder'),
        ('semi_supervised', None, 'pseudo_label_with_al', 'learning_loss', True, 'autoencoder'),
    ]

    for (m, u, s, us, p, init) in states:
        arguments.weak_supervision_strategy = m
        arguments.uncertainty_sampling_method = u
        arguments.semi_supervised_method = s
        arguments.semi_supervised_uncertainty_method = us
        arguments.load_pretrained = p
        arguments.semi_supervised_init = init
        main(args=arguments)

    find_median()
