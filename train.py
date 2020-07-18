import argparse
import os
import time
from copy import deepcopy

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np

from model.wideresnet import WideResNet
from model.densenet import densenet121
from model.lenet import LeNet
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.cifar100_dataset import Cifar100Dataset
from utils import save_checkpoint, AverageMeter, accuracy, create_loaders, print_args, postprocess_indices
from utils import stratified_random_sampling, Metrics, store_logs
from active_learning.uncertainty_sampling import UncertaintySampling
from semi_supervised.pseudo_labeling import PseudoLabeling
from semi_supervised.auto_encoder import AutoEncoder

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--autoencoder-train-epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int,
                    help='widen factor (default: 2)')
parser.add_argument('--drop-rate', default=0.2, type=float,
                    help='dropout probability (default: 0.2)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', action='store_true', help='flag to be set if an existed model is to be loaded')
parser.add_argument('--name', default='densenet-least-confidence', type=str,
                    help='name of experiment')
parser.add_argument('--add-labeled-epochs', default=20, type=int,
                    help='if the test accuracy stays stable for add-labeled-epochs epochs then add new data')
parser.add_argument('--add-labeled-ratio', default=0.05, type=int,
                    help='what percentage of labeled data to be added')
parser.add_argument('--labeled-ratio-start', default=0.01, type=int,
                    help='what percentage of labeled data to start the training with')
parser.add_argument('--labeled-ratio-stop', default=0.7, type=int,
                    help='what percentage of labeled data to stop the training process at')
parser.add_argument('--labeled-warmup_epochs', default=80, type=int,
                    help='how many epochs to warmup for, without sampling or pseudo labeling')
parser.add_argument('--arch', default='lenet', type=str, choices=['wideresnet', 'densenet', 'lenet'],
                    help='arch name')
parser.add_argument('--uncertainty-sampling-method', default='least_confidence', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based',
                             'density_weighted'],
                    help='the uncertainty sampling method to use')
parser.add_argument('--root', default='/home/qasima/datasets/thesis/stratified/', type=str,
                    help='the root path for the datasets')
parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling'],
                    help='the weakly supervised strategy to use')
parser.add_argument('--semi-supervised-method', default='auto_encoder', type=str,
                    choices=['pseudo_labeling', 'auto_encoder'],
                    help='the semi supervised method to use')
parser.add_argument('--pseudo-labeling-threshold', default=0.3, type=int,
                    help='the threshold for considering the pseudo label as the actual label')
parser.add_argument('--weighted', action='store_true', help='to use weighted loss or not')
parser.add_argument('--eval', action='store_true', help='only perform evaluation and exit')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'matek', 'cifar100'],
                    help='the dataset to train on')
parser.add_argument('--checkpoint-path', default='/home/qasima/med_active_learning/runs/', type=str,
                    help='the directory root for saving/resuming checkpoints from')
parser.add_argument('--seed', default=9999, type=int, choices=[0, 9999, 2323, 5555], help='the random seed to set')
parser.add_argument('--log-path', default='/home/qasima/med_active_learning/logs/', type=str,
                    help='the directory root for storing/retrieving the logs')
parser.add_argument('--store_logs', action='store_false', help='store the logs after training')

parser.set_defaults(augment=True)

args = parser.parse_args()
datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset}

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(args.seed)


def main():
    global args
    if args.weak_supervision_strategy == 'semi_supervised':
        args.name = f"{args.dataset}@{args.arch}@{args.semi_supervised_method}"
    elif args.weak_supervision_strategy == 'active_learning':
        args.name = f"{args.dataset}@{args.arch}@{args.uncertainty_sampling_method}"
    else:
        args.name = f"{args.dataset}@{args.arch}@{args.weak_supervision_strategy}"

    if args.semi_supervised_method == 'auto_encoder':
        auto_encoder = AutoEncoder(args)
        auto_encoder.train()
        auto_encoder.train_validate_classifier()
        exit(1)

    dataset_class = datasets[args.dataset](root=args.root,
                                           labeled_ratio=args.labeled_ratio_start,
                                           add_labeled_ratio=args.add_labeled_ratio)

    labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = dataset_class.get_dataset()

    kwargs = {'num_workers': 2, 'pin_memory': False}
    train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset,
                                                                labeled_indices, unlabeled_indices, kwargs)

    uncertainty_sampler = UncertaintySampling(verbose=True,
                                              uncertainty_sampling_method=args.uncertainty_sampling_method)
    pseudo_labeler = PseudoLabeling()

    if args.arch == 'wideresnet':
        model = WideResNet(args.layers,
                           num_classes=dataset_class.num_classes,
                           widen_factor=args.widen_factor,
                           drop_rate=args.drop_rate,
                           input_size=dataset_class.input_size)
    elif args.arch == 'densenet':
        model = densenet121(num_classes=dataset_class.num_classes)
    elif args.arch == 'lenet':
        model = LeNet(num_channels=3, num_classes=dataset_class.num_classes, droprate=args.drop_rate)
    else:
        raise NotImplementedError

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # doc: for training on multiple GPUs.
    # doc: Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # doc: model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    if args.resume:
        file = os.path.join(args.checkpoint_path, args.name, 'model_best.pth.tar')
        if os.path.isfile(file):
            print("=> loading checkpoint '{}'".format(file))
            checkpoint = torch.load(file)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(file))

    if args.weighted:
        classes_targets = unlabeled_dataset.targets[unlabeled_indices]
        classes_samples = [torch.sum(classes_targets == i) for i in range(dataset_class.num_classes)]
        classes_weights = np.log(len(unlabeled_dataset)) - np.log(classes_samples)
        criterion = nn.CrossEntropyLoss(weight=classes_weights).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    last_best_epochs = 0
    current_labeled_ratio = args.labeled_ratio_start
    acc_ratio = {}
    best_model = deepcopy(model)

    print_args(args)

    if args.eval:
        print('Starting evaluation..')
        metrics, report = evaluate(val_loader, model)
        print('Evaluation:\t'
              f'Precision: {metrics[0]}\t'
              f'Recall: {metrics[1]}\t'
              f'F1-score: {metrics[2]}\t')
        print(report)
        exit(1)
    else:
        print('Starting training..')

    best_acc1 = 0
    best_acc5 = 0
    best_prec1 = 0
    best_recall1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, last_best_epochs)
        acc, acc5, (prec, recall, f1, _) = validate(val_loader, model, criterion, last_best_epochs)
        scheduler.step(epoch=epoch)

        is_best = acc > best_acc1
        last_best_epochs = 0 if is_best else last_best_epochs + 1
        best_model = deepcopy(model) if is_best else best_model

        if epoch > args.labeled_warmup_epochs and epoch % args.add_labeled_epochs == 0:
            acc_ratio.update({np.round(current_labeled_ratio, decimals=2):
                             [best_acc1, best_acc5, best_prec1, best_recall1]})
            if args.weak_supervision_strategy == 'active_learning':
                samples_indices = uncertainty_sampler.get_samples(epoch, args, model,
                                                                  train_loader,
                                                                  unlabeled_loader,
                                                                  number=dataset_class.add_labeled_num)

                labeled_indices, unlabeled_indices = postprocess_indices(labeled_indices, unlabeled_indices,
                                                                         samples_indices)

                train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset,
                                                                            test_dataset, labeled_indices,
                                                                            unlabeled_indices, kwargs)

                print(f'Uncertainty Sampling\t '
                      f'Current labeled ratio: {current_labeled_ratio}\t')
            elif args.weak_supervision_strategy == 'semi_supervised':
                samples_indices, samples_targets = pseudo_labeler.get_samples(epoch, args, best_model,
                                                                              unlabeled_loader,
                                                                              number=dataset_class.add_labeled_num)

                labeled_indices, unlabeled_indices = postprocess_indices(labeled_indices, unlabeled_indices,
                                                                         samples_indices)

                pseudo_labels_acc = np.zeros(samples_indices.shape[0])
                for i, j in enumerate(samples_indices):
                    if labeled_dataset.targets[j] == samples_targets[i]:
                        pseudo_labels_acc[i] = 1
                    else:
                        labeled_dataset.targets[j] = samples_targets[i]

                train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset,
                                                                            test_dataset, labeled_indices,
                                                                            unlabeled_indices, kwargs)

                print(f'Pseudo labeling\t '
                      f'Current labeled ratio: {current_labeled_ratio}\t'
                      f'Pseudo labeled accuracy: {np.sum(pseudo_labels_acc == 1) / samples_indices.shape[0]}')

            else:
                samples_indices = stratified_random_sampling(unlabeled_indices, number=dataset_class.add_labeled_num)

                labeled_indices, unlabeled_indices = postprocess_indices(labeled_indices, unlabeled_indices,
                                                                         samples_indices)

                train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset,
                                                                            test_dataset, labeled_indices,
                                                                            unlabeled_indices, kwargs)

                print(f'Random Sampling\t '
                      f'Current labeled ratio: {current_labeled_ratio}\t')

            current_labeled_ratio += args.add_labeled_ratio
            last_best_epochs = 0

        best_acc1 = max(acc, best_acc1)
        best_prec1 = max(prec, best_prec1)
        best_recall1 = max(recall, best_recall1)
        best_acc5 = max(acc5, best_acc5)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc1,
        }, is_best)

        if current_labeled_ratio > args.labeled_ratio_stop:
            break

    metrics, report = evaluate(val_loader, best_model)

    for k, v in acc_ratio.items():
        print(f'Ratio: {int(k*100)}%\t'
              f'Accuracy@1: {v[0]}\t'
              f'Accuracy@5: {v[1]}\t'
              f'Precision: {v[2]}\t'
              f'Recall: {v[3]}\t')
    print('Best acc@1: {0} \tacc@5: {1}'.format(best_acc1, best_acc5))
    print('Evaluation:\t'
          f'Precision: {metrics[0]}\t'
          f'Recall: {metrics[1]}\t'
          f'F1-score: {metrics[2]}\t')
    print(report)

    if args.store_logs:
        store_logs(args, acc_ratio)


def train(train_loader, model, criterion, optimizer, epoch, last_best_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (data_x, data_y) in enumerate(train_loader):
        data_y = data_y.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        optimizer.zero_grad()
        output, _ = model(data_x)
        loss = criterion(output, data_y)

        acc = accuracy(output.data, data_y, topk=(1,))[0]
        losses.update(loss.data.item(), data_x.size(0))
        top1.update(acc.item(), data_x.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Last best epoch {last_best_epoch}'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1,
                          last_best_epoch=last_best_epochs))


def validate(val_loader, model, criterion, last_best_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    metrics = Metrics()

    model.eval()

    end = time.time()
    for i, (data_x, data_y) in enumerate(val_loader):
        data_y = data_y.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output, _ = model(data_x)
        loss = criterion(output, data_y)

        acc = accuracy(output.data, data_y, topk=(1, 5, ))
        losses.update(loss.data.item(), data_x.size(0))
        top1.update(acc[0].item(), data_x.size(0))
        top5.update(acc[1].item(), data_x.size(0))
        metrics.add_mini_batch(data_y, output)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Last best epoch {last_best_epoch}'
                  .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1,
                          last_best_epoch=last_best_epochs))

    (prec, recall, f1, _) = metrics.get_metrics()
    print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1} * Acc@5 {top5.avg:.3f}\t'
          .format(prec, recall, top1=top1, top5=top5))

    return top1.avg, top5.avg, (prec, recall, f1, _)


def evaluate(val_loader, model):
    model.eval()

    metrics = Metrics()
    for i, (data_x, data_y) in enumerate(val_loader):
        data_y = data_y.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output, _ = model(data_x)
        metrics.add_mini_batch(data_y, output)

    return metrics.get_metrics(), metrics.get_report()


if __name__ == '__main__':
    main()
