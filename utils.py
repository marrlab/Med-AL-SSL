import os
import torch.nn as nn
import shutil
from torch.utils.data import DataLoader
from numpy.random import default_rng
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
import json
from datetime import datetime
import torchvision


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar', best_model_filename='model_best.pth.tar'):
    directory = os.path.join(args.checkpoint_path, args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, best_model_filename))


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, *self.shape)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset, labeled_indices, unlabeled_indices, kwargs):
    labeled_dataset.indices = labeled_indices
    unlabeled_dataset.indices = unlabeled_indices

    labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    return labeled_loader, unlabeled_loader, val_loader


def create_base_loader(base_dataset, kwargs, batch_size):
    return DataLoader(dataset=base_dataset, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)


def stratified_random_sampling(unlabeled_indices, number):
    rng = default_rng()
    samples_indices = rng.choice(unlabeled_indices.shape[0], size=number, replace=False)

    return samples_indices


def postprocess_indices(labeled_indices, unlabeled_indices, samples_indices):
    unlabeled_mask = torch.ones(size=(len(unlabeled_indices),), dtype=torch.bool)
    unlabeled_mask[samples_indices] = 0
    labeled_indices = np.hstack([labeled_indices, unlabeled_indices[~unlabeled_mask]])
    unlabeled_indices = unlabeled_indices[unlabeled_mask]

    return labeled_indices, unlabeled_indices


class Metrics:
    def __init__(self):
        self.targets = []
        self.outputs = []
        self.outputs_probs = None

    def add_mini_batch(self, mini_targets, mini_outputs):
        self.targets.extend(mini_targets.tolist())
        self.outputs.extend(torch.argmax(mini_outputs, dim=1).tolist())
        self.outputs_probs = mini_outputs \
            if self.outputs_probs is None else torch.cat([self.outputs_probs, mini_outputs], dim=0)

    def get_metrics(self):
        return precision_recall_fscore_support(self.targets, self.outputs, average='macro', zero_division=1)

    def get_report(self):
        return classification_report(self.targets, self.outputs, zero_division=1)

    def get_confusion_matrix(self):
        return confusion_matrix(self.targets, self.outputs)

    def get_roc_auc_curve(self):
        self.outputs_probs = torch.softmax(self.outputs_probs, dim=1)
        return roc_auc_score(self.targets, self.outputs_probs.cpu().numpy(), multi_class='ovr')


import torch
import torch.nn as nn


class NTXent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXent, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples()

    def mask_correlated_samples(self):
        # noinspection PyTypeChecker
        mask = torch.ones((self.batch_size * 2, self.batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )

        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size

        return loss


class TransformsSimCLR:
    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


def print_args(args):
    print('Arguments:\n'
          f'Model name: {args.name}\t'
          f'Epochs: {args.epochs}\t'
          f'Batch Size: {args.batch_size}\n'
          f'Architecture: {args.arch}\t'
          f'Weak Supervision Strategy: {args.weak_supervision_strategy}\n'
          f'Uncertainty Sampling Method: {args.uncertainty_sampling_method}\t'
          f'Semi Supervised Method: {args.semi_supervised_method}\n'
          f'Dataset root: {args.root}')


def store_logs(args, acc_ratio):
    filename = '{0}-{1}-seed:{2}'.format(datetime.now().strftime("%d.%m.%Y"), args.name, args.seed)
    file = dict()
    file.update({'name': args.name})
    file.update({'time': str(datetime.now())})
    file.update({'seed': args.seed})
    file.update({'dataset': args.dataset})
    file.update({'metrics': acc_ratio})
    file.update({'args': args})

    with open(os.path.join(args.log_path, filename), 'w') as fp:
        json.dump(file, fp, indent=4, sort_keys=True)


# noinspection PyTypeChecker,PyUnresolvedReferences
def print_metrics(name, log_path):
    filenames = os.listdir(log_path)
    metrics = {'acc1': [[], []], 'acc5': [[], []], 'prec': [[], []], 'recall': [[], []],
               'acc1_std': [], 'acc5_std': [], 'prec_std': [], 'recall_std': []}
    ratios = []

    for filename in filenames:
        with open(os.path.join(log_path, filename), 'r') as fp:
            file = json.load(fp)
        if file['name'] == name:
            for k, v in file['metrics'].items():
                v = np.round(v, decimals=2)
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
    metrics['prec'][0] = np.array(metrics['prec'][0])
    metrics['prec'][1] = np.array(metrics['prec'][1])
    metrics['recall'][0] = np.array(metrics['recall'][0])
    metrics['recall'][1] = np.array(metrics['recall'][1])

    print(f'* Name: {name}\n\n'
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
