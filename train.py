import argparse
import os
import time

from torchvision.datasets import ImageFolder

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np
from sklearn.metrics import classification_report

from model.wideresnet import WideResNet
from model.densenet import DenseNet
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import CifarDataset
from utils import save_checkpoint, AverageMeter, accuracy, create_loaders, print_args, postprocess_indices
from active_learning.uncertainty_sampling import UncertaintySampling
from semi_supervised.pseudo_labeling import PseudoLabeling

parser = argparse.ArgumentParser(description='Active Learning Basic Medical Imaging')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int,
                    help='widen factor (default: 2)')
parser.add_argument('--drop-rate', default=0, type=float,
                    help='dropout probability (default: 0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='densenet-least-confidence', type=str,
                    help='name of experiment')
parser.add_argument('--add-labeled-epochs', default=10, type=int,
                    help='if the test accuracy stays stable for add-labeled-epochs epochs then add new data')
parser.add_argument('--add-labeled-ratio', default=0.05, type=int,
                    help='what percentage of labeled data to be added')
parser.add_argument('--labeled-ratio-start', default=0.05, type=int,
                    help='what percentage of labeled data to start the training with')
parser.add_argument('--labeled-ratio-stop', default=0.35, type=int,
                    help='what percentage of labeled data to stop the training process at')
parser.add_argument('--arch', default='densenet', type=str, choices=['wideresnet', 'densenet'],
                    help='arch name')
parser.add_argument('--uncertainty-sampling-method', default='least_confidence', type=str,
                    choices=['least_confidence', 'margin_confidence', 'ratio_confidence', 'entropy_based'],
                    help='the uncertainty sampling method to use')
parser.add_argument('--root', default='/home/qasima/datasets/thesis/stratified/', type=str,
                    help='the root path for the datasets')
parser.add_argument('--weak-supervision-strategy', default='semi_supervised', type=str,
                    choices=['active_learning', 'semi_supervised'],
                    help='the weakly supervised strategy to use')
parser.add_argument('--semi-supervised-method', default='pseudo_labeling', type=str,
                    choices=['pseudo_labeling'],
                    help='the semi supervised method to use')
parser.add_argument('--pseudo-labeling-threshold', default=0.3, type=int,
                    help='the threshold for considering the pseudo label as the actual label')
parser.add_argument('--weighted', action='store_true', help='to use weighted loss or not')
parser.add_argument('--eval', action='store_true', help='only perform  evaluation and exit')
parser.add_argument('--dataset', default='matek', type=str, choices=['cifar10', 'matek'],
                    help='the dataset to train on')

parser.set_defaults(augment=True)

best_prec1 = 0
args = parser.parse_args()
datasets = {'matek': MatekDataset, 'cifar10': CifarDataset}


def main():
    global args, best_prec1
    args.name = f"{args.dataset}@{args.arch}@{args.semi_supervised_method}" \
        if args.weak_supervision_strategy == 'semi_supervised' \
        else f"{args.dataset}@{args.arch}@{args.uncertainty_sampling_method}"

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
        model = DenseNet(num_classes=dataset_class.num_classes, growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         num_init_features=64,
                         drop_rate=args.drop_rate)
    else:
        raise NotImplementedError

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # doc: for training on multiple GPUs.
    # doc: Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # doc: model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.weighted:
        classes_targets = torch.FloatTensor(labeled_dataset.targets[labeled_indices])
        classes_samples = torch.FloatTensor([torch.sum(classes_targets == i) for i in range(dataset_class.num_classes)])
        classes_weights = np.log(len(labeled_dataset)) - torch.log(classes_samples)
        criterion = nn.CrossEntropyLoss(weight=classes_weights).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, eta_min=0)

    last_best_epochs = 0
    current_labeled_ratio = args.labeled_ratio_start
    acc_ratio = {}
    best_model = model

    print_args(args)

    if args.eval:
        print('Starting evaluation..')
        report = evaluate(val_loader, model)
        print(report)
        exit(1)
    else:
        print('Starting training..')

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, scheduler, epoch)
        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        last_best_epochs = 0 if is_best else last_best_epochs + 1
        best_model = model if is_best else best_model

        if last_best_epochs == args.add_labeled_epochs:
            acc_ratio.update({current_labeled_ratio: best_prec1})
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
            else:
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

            current_labeled_ratio += args.add_labeled_ratio
            last_best_epochs = 0

        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

        if current_labeled_ratio > args.labeled_ratio_stop:
            break

    report = evaluate(val_loader, best_model)

    for k, v in acc_ratio.items():
        print(f'Ratio: {int(k*100)}%\t'
              f'Accuracy: {v}')
    print('Best accuracy: ', best_prec1)
    print(report)


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (data_x, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        output = model(data_x)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), data_x.size(0))
        top1.update(prec1.item(), data_x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (data_x, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(data_x)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), data_x.size(0))
        top1.update(prec1.item(), data_x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def evaluate(val_loader, model):
    model.eval()

    outputs = []
    targets = []
    for i, (data_x, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(data_x)
        outputs.extend(torch.argmax(output, dim=1).tolist())
        targets.extend(target.tolist())

    return classification_report(targets, outputs, zero_division=1)


if __name__ == '__main__':
    main()
