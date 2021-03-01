import pandas as pd
from copy import deepcopy
import numpy as np

"""
root = '../results.csv'

results = pd.read_csv(root)
datasets_rep = ['White blood cells', 'Skin Lesions', 'Jurkat cells cycle', 'Retina']
metrics_rep = ['Recall', 'Precision', 'F1-score', 'Accuracy']

for metric in metrics_rep:
    results[f'{metric} Avg.'] = 0
    results[f'{metric} Avg. Rank'] = 0
    results[f'{metric} AUC'] = 0

for dataset in datasets_rep:
    result = deepcopy(results[results['Dataset'] == dataset])

    # start, end = result.index[0], result.index[-1]

    for metric in metrics_rep:
        metric_cols = [f'{metric} {i}' for i in range(9)]

        result[f'{metric} Avg.'] = result[metric_cols].mean(axis=1)
        result[f'{metric} Avg. Rank'] = result[metric_cols].rank(method='min', axis=0, ascending=False).mean(axis=1)
        result[f'{metric} AUC'] = np.trapz(result[metric_cols].values, axis=1)

        results.loc[results['Dataset'] == dataset, f'{metric} Avg.'] = result[f'{metric} Avg.']
        results.loc[results['Dataset'] == dataset, f'{metric} Avg. Rank'] = result[f'{metric} Avg. Rank']
        results.loc[results['Dataset'] == dataset, f'{metric} AUC'] = result[f'{metric} AUC']

results.to_excel('../results_with_metrics.xlsx')
"""
import os

df = pd.read_excel('../results/results_with_metrics.xlsx')

methods = df['Method'].unique()

for method in methods:
    df.loc[df['Method'] == method, 'F1-score Avg. Avg. Rank'] = df[df['Method'] == method]['F1-score Avg. Rank'].median()


df.to_excel('../results/results_with_metrics_avg.xlsx')
