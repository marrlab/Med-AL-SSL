import argparse
import os
import time
from copy import deepcopy

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np

manual_seed = 1

random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
cudnn.deterministic = True
cudnn.benchmark = False
torch.cuda.manual_seed(manual_seed)
# torch.cuda.set_rng_state(torch.manual_seed(0).get_state())

from model.wideresnet import WideResNet
from model.densenet import densenet121
from model.lenet import LeNet
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.cifar100_dataset import Cifar100Dataset
from utils import save_checkpoint, AverageMeter, accuracy, create_loaders, print_args, postprocess_indices, stratified_random_sampling, Metrics
from active_learning.uncertainty_sampling import UncertaintySampling
from semi_supervised.pseudo_labeling import PseudoLabeling

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--epochs', default=200, type=int,
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
parser.add_argument('--add-labeled-epochs', default=10, type=int,
                    help='if the test accuracy stays stable for add-labeled-epochs epochs then add new data')
parser.add_argument('--add-labeled-ratio', default=0.05, type=int,
                    help='what percentage of labeled data to be added')
parser.add_argument('--labeled-ratio-start', default=0.1, type=int,
                    help='what percentage of labeled data to start the training with')
parser.add_argument('--labeled-ratio-stop', default=0.35, type=int,
                    help='what percentage of labeled data to stop the training process at')
parser.add_argument('--arch', default='lenet', type=str, choices=['wideresnet', 'densenet', 'lenet'],
                    help='arch name')
parser.add_argument('--uncertainty-sampling-method', default='least_confidence', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based'],
                    help='the uncertainty sampling method to use')
parser.add_argument('--root', default='/home/qasima/datasets/thesis/stratified/', type=str,
                    help='the root path for the datasets')
parser.add_argument('--weak-supervision-strategy', default='random_sampling', type=str,
                    choices=['active_learning', 'semi_supervised', 'random_sampling'],
                    help='the weakly supervised strategy to use')
parser.add_argument('--semi-supervised-method', default='pseudo_labeling', type=str,
                    choices=['pseudo_labeling'],
                    help='the semi supervised method to use')
parser.add_argument('--pseudo-labeling-threshold', default=0.3, type=int,
                    help='the threshold for considering the pseudo label as the actual label')
parser.add_argument('--weighted', action='store_true', help='to use weighted loss or not')
parser.add_argument('--eval', action='store_true', help='only perform evaluation and exit')
parser.add_argument('--dataset', default='matek', type=str, choices=['cifar10', 'matek', 'cifar100'],
                    help='the dataset to train on')
parser.add_argument('--checkpoint-path', default='/home/qasima/med_active_learning/runs/', type=str,
                    help='the directory root for saving/resuming checkpoints from')

parser.set_defaults(augment=True)

best_acc1 = 0
best_prec1 = 0
best_recall1 = 0
args = parser.parse_args()
datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset}


def main():
    global args, best_acc1, best_prec1, best_recall1
    if args.weak_supervision_strategy == 'semi_supervised':
        args.name = f"{args.dataset}@{args.arch}@{args.semi_supervised_method}"
    elif args.weak_supervision_strategy == 'active_learning':
        args.name = f"{args.dataset}@{args.arch}@{args.uncertainty_sampling_method}"
    else:
        args.name = f"{args.dataset}@{args.arch}@{args.weak_supervision_strategy}"

    print(args)

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
            best_acc1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(file))

    if args.weighted:
        classes_targets = torch.FloatTensor(unlabeled_dataset.targets[unlabeled_indices])
        classes_samples = torch.FloatTensor([torch.sum(classes_targets == i) for i in range(dataset_class.num_classes)])
        classes_weights = np.log(len(unlabeled_dataset)) - torch.log(classes_samples)
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

    sampling_order = [80, 100, 120, 140, 160, 180, 200]

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, last_best_epochs)
        acc, (prec, recall, f1, _) = validate(val_loader, model, criterion, last_best_epochs)
        scheduler.step(epoch=epoch)

        is_best = acc > best_acc1
        last_best_epochs = 0 if is_best else last_best_epochs + 1
        best_model = deepcopy(model) if is_best else best_model

        # if last_best_epochs == args.add_labeled_epochs:
        if epoch == sampling_order[0]:
            acc_ratio.update({current_labeled_ratio: [best_acc1, best_prec1, best_recall1]})
            if args.weak_supervision_strategy == 'active_learning':
                samples_indices = uncertainty_sampler.get_samples(epoch, args, model,
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
            sampling_order.pop(0)

        best_acc1 = max(acc, best_acc1)
        best_prec1 = max(prec, best_prec1)
        best_recall1 = max(recall, best_recall1)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc1,
        }, is_best)

        if current_labeled_ratio == args.labeled_ratio_stop:
            break

    metrics, report = evaluate(val_loader, best_model)

    for k, v in acc_ratio.items():
        print(f'Ratio: {int(k*100)}%\t'
              f'Accuracy: {v[0]}\t'
              f'Precision: {v[1]}\t'
              f'Recall: {v[2]}\t')
    print('Best accuracy: ', best_acc1)
    print('Evaluation:\t'
          f'Precision: {metrics[0]}\t'
          f'Recall: {metrics[1]}\t'
          f'F1-score: {metrics[2]}\t')
    print(report)


def train(train_loader, model, criterion, optimizer, epoch, last_best_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (data_x, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        optimizer.zero_grad()
        output = model(data_x)
        loss = criterion(output, target)

        acc = accuracy(output.data, target, topk=(1,))[0]
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
    metrics = Metrics()

    model.eval()

    end = time.time()
    for i, (data_x, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(data_x)
        loss = criterion(output, target)

        acc = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), data_x.size(0))
        top1.update(acc.item(), data_x.size(0))
        metrics.add_mini_batch(target, output)

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
    print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1}'.format(prec, recall, top1=top1))

    return top1.avg, (prec, recall, f1, _)


def evaluate(val_loader, model):
    model.eval()

    metrics = Metrics()
    for i, (data_x, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(data_x)
        metrics.add_mini_batch(target, output)

    return metrics.get_metrics(), metrics.get_report()


if __name__ == '__main__':
    main()
