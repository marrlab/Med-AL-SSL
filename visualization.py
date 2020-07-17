import matplotlib.pyplot as plt
import matplotlib.style as style


def plot_acc_dataprop(prop, acc, methods):
    """
    plot the accuracy vs data proportion being used, graph
    credits to: Alex Olteanu (https://www.dataquest.io/blog/making-538-plots/) for the plot style
    :return: None
    """
    plt.figure(figsize=(14, 10))
    style.use('fivethirtyeight')

    colors = [[0, 0, 0, 1], [230 / 255, 159 / 255, 0, 1], [86 / 255, 180 / 255, 233 / 255, 1],
              [0, 158 / 255, 115 / 255, 1], [213 / 255, 94 / 255, 0, 1], [0, 114 / 255, 178 / 255, 1],
              [93 / 255, 58 / 255, 155 / 255, 1], [153 / 255, 79 / 255, 0, 1], [211 / 255, 95 / 255, 183 / 255, 1]]

    for i, j in enumerate(range(1, len(acc), 3)):
        # plt.plot(prop[i], acc[j-1], linestyle='dashed', color=colors[i % len(colors)], linewidth=2, alpha=0.7)
        plt.plot(prop[i], acc[j], color=colors[i % len(colors)], label=methods[i], linewidth=2)
        # plt.plot(prop[i], acc[j+1], linestyle='dashed', color=colors[i % len(colors)], linewidth=2, alpha=0.7)
        plt.fill_between(prop[i], acc[j-1], acc[j+1], color=colors[i % len(colors)], alpha=0.05)

    plt.title("A comparison of active and semi-supervised learning on Cifar-10",
              fontsize=15, weight='bold', alpha=.75)
    plt.xlabel("Labeled percentage of dataset (%)", fontsize=15, weight='bold', alpha=.75)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=15, weight='bold', alpha=.75)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    plot_acc_dataprop(prop=[
        [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66],
        [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66],
        [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66],
        [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66],
        [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66],
        [0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56, 0.61, 0.66]
    ], acc=[
        [30.82, 34.15, 42.01, 48.20, 49.97, 51.64, 52.44, 54.69, 56.51, 57.46, 58.7, 59.7, 59.98, 60.21],
        [32.74, 35.90, 44.03, 50.37, 52.3, 53.92, 55.32, 57.12, 58.9, 59.97, 61.34, 62.35, 62.6, 62.86],
        [34.66, 37.65, 46.05, 52.53, 54.62, 56.2, 58.2, 59.55, 61.28, 62.48, 63.98, 65.0, 65.22, 65.51],
        [30.82, 32.64, 40.66, 48.81, 50.78, 52.43, 53.44, 55.81, 57.6, 58.89, 59.9, 60.6, 60.77, 60.98],
        [32.74, 36.61, 43.72, 50.79, 52.74, 54.38, 55.5, 57.36, 58.9, 60.39, 61.32, 62.0, 62.27, 62.41],
        [34.66, 40.58, 46.78, 52.77, 54.7, 56.33, 57.56, 58.91, 60.20, 61.89, 62.74, 63.4, 63.77, 63.84],
        [30.82, 33.83, 41.9, 48.44, 50.51, 52.66, 53.84, 55.13, 56.72, 57.96, 58.99, 60.0, 60.28, 60.43],
        [32.74, 36.6, 44.19, 50.43, 52.38, 54.35, 55.64, 57.28, 58.8, 60.2, 61.42, 62.44, 62.73, 62.88],
        [34.66, 39.37, 46.48, 52.42, 54.25, 56.04, 57.44, 59.43, 60.88, 62.44, 63.85, 64.88, 65.18, 65.33],
        [30.82, 33.02, 43.12, 49.92, 51.29, 53.35, 53.84, 55.43, 56.79, 57.97, 59.2, 60.02, 60.29, 60.53],
        [32.74, 35.52, 44.72, 52.23, 54.12, 56.01, 57.26, 58.83, 60.06, 61.28, 62.21, 63.16, 63.34, 63.56],
        [34.66, 38.02, 46.32, 54.54, 56.95, 58.67, 60.68, 62.23, 63.33, 64.59, 65.22, 66.3, 66.39, 66.59],
        [30.82, 33.96, 43.37, 49.62, 52.2, 53.87, 54.53, 56.53, 57.98, 59.18, 60.37, 61.13, 61.39, 61.65],
        [32.74, 36.07, 45.07, 51.27, 53.6, 55.2, 56.78, 58.54, 59.96, 61.27, 62.26, 63.21, 63.43, 63.62],
        [34.66, 38.18, 46.77, 52.92, 55.0, 56.53, 59.03, 60.55, 61.94, 63.36, 64.15, 65.29, 65.47, 65.59],
        [30.82, 33.82, 40.98, 48.25, 50.25, 51.88, 53.12, 54.68, 56.22, 57.61, 58.7, 59.36, 59.55, 59.96],
        [32.74, 36.3, 44.25, 50.48, 52.41, 53.89, 55.13, 56.74, 58.25, 59.47, 60.67, 61.62, 61.84, 62.16],
        [34.66, 38.78, 47.52, 52.71, 54.57, 55.9, 57.14, 58.80, 60.28, 61.33, 62.64, 63.88, 64.13, 64.36],
    ], methods=[
        'Margin Sampling',
        'Least Confidence',
        'Ratio Sampling',
        'Entropy Based',
        'Density Weighted',
        'Random Sampling'
    ])