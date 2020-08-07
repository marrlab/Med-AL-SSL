from options.train_options import get_arguments
import os
import time
from copy import deepcopy

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np

from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.cifar100_dataset import Cifar100Dataset
from utils import save_checkpoint, AverageMeter, accuracy, create_loaders, print_args, \
    create_model_optimizer_scheduler, get_loss, resume_model, set_model_name, perform_sampling
from utils import Metrics, store_logs
from active_learning.uncertainty_sampling import UncertaintySampling
from active_learning.mc_dropout import UncertaintySamplingMCDropout
from semi_supervised.pseudo_labeling import PseudoLabeling
from semi_supervised.auto_encoder import AutoEncoder
from semi_supervised.simclr import SimCLR

arguments = get_arguments()
datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset}


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(args.seed)

    args.name = set_model_name(args)

    if args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'auto_encoder':
        auto_encoder = AutoEncoder(args)
        auto_encoder.train()
        best_acc = auto_encoder.train_validate_classifier()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'simclr':
        simclr = SimCLR(args)
        simclr.train()
        best_acc = simclr.train_validate_classifier()
        return best_acc

    dataset_class = datasets[args.dataset](root=args.root,
                                           labeled_ratio=args.labeled_ratio_start,
                                           add_labeled_ratio=args.add_labeled_ratio,
                                           advanced_transforms=True)

    labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = dataset_class.get_dataset()

    kwargs = {'num_workers': 2, 'pin_memory': False}
    train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset,
                                                                labeled_indices, unlabeled_indices, kwargs)

    if args.uncertainty_sampling_method == 'mc_dropout':
        uncertainty_sampler = UncertaintySamplingMCDropout()
    else:
        uncertainty_sampler = UncertaintySampling(verbose=True,
                                                  uncertainty_sampling_method=args.uncertainty_sampling_method)
    pseudo_labeler = PseudoLabeling()

    model, optimizer, scheduler = create_model_optimizer_scheduler(args, dataset_class)

    if args.resume:
        model = resume_model(args, model)

    criterion = get_loss(args, unlabeled_dataset, unlabeled_indices, dataset_class)

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

    best_acc1, best_acc5, best_prec1, best_recall1 = 0, 0, 0, 0

    for epoch in range(args.start_epoch, args.epochs):
        model = train(train_loader, model, criterion, optimizer, epoch, last_best_epochs, args)
        acc, acc5, (prec, recall, f1, _), confusion_mat, roc_auc_curve = validate(val_loader, model,
                                                                                  criterion, last_best_epochs, args)
        is_best = acc > best_acc1
        last_best_epochs = 0 if is_best else last_best_epochs + 1
        best_model = deepcopy(model) if is_best else best_model

        if epoch > args.labeled_warmup_epochs and epoch % args.add_labeled_epochs == 0:
            acc_ratio.update({np.round(current_labeled_ratio, decimals=2):
                             [acc, acc5, prec, recall, f1, confusion_mat, roc_auc_curve]})

            train_loader, unlabeled_loader, val_loader = perform_sampling(args, uncertainty_sampler, pseudo_labeler,
                                                                          epoch, model, train_loader, unlabeled_loader,
                                                                          dataset_class, labeled_indices,
                                                                          unlabeled_indices, labeled_dataset,
                                                                          unlabeled_dataset,
                                                                          test_dataset, kwargs, current_labeled_ratio,
                                                                          best_model)
            current_labeled_ratio += args.add_labeled_ratio
            best_acc1, best_acc5, best_prec1, best_recall1 = 0, 0, 0, 0
            model, optimizer, scheduler = create_model_optimizer_scheduler(args, dataset_class)

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
        print(f'Ratio: {int(k * 100)}%\t'
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


def train(train_loader, model, criterion, optimizer, epoch, last_best_epochs, args):
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

    return model


def validate(val_loader, model, criterion, last_best_epochs, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    metrics = Metrics()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (data_x, data_y) in enumerate(val_loader):
            data_y = data_y.cuda(non_blocking=True)
            data_x = data_x.cuda(non_blocking=True)

            output, _ = model(data_x)
            loss = criterion(output, data_y)

            acc = accuracy(output.data, data_y, topk=(1, 5,))
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
    confusion_matrix = metrics.get_confusion_matrix()
    roc_auc_curve = metrics.get_roc_auc_curve()
    print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1} * Acc@5 {top5.avg:.3f}\t * Roc_Auc {2}\t'
          .format(prec, recall, roc_auc_curve, top1=top1, top5=top5))

    return top1.avg, top5.avg, (prec, recall, f1, _), confusion_matrix, roc_auc_curve


def evaluate(val_loader, model):
    model.eval()

    metrics = Metrics()
    with torch.no_grad():
        for i, (data_x, data_y) in enumerate(val_loader):
            data_y = data_y.cuda(non_blocking=True)
            data_x = data_x.cuda(non_blocking=True)

            output, _ = model(data_x)
            metrics.add_mini_batch(data_y, output)

    return metrics.get_metrics(), metrics.get_report()


if __name__ == '__main__':
    if arguments.run_batch:
        states = [
            ('active_learning', 'least_confidence', 'pseudo_labeling'),
            ('active_learning', 'margin_confidence', 'pseudo_labeling'),
            ('active_learning', 'ratio_confidence', 'pseudo_labeling'),
            ('active_learning', 'entropy_based', 'pseudo_labeling'),
            ('active_learning', 'density_weighted', 'pseudo_labeling'),
            ('semi_supervised', 'least_confidence', 'pseudo_labeling'),
            ('random_sampling', 'least_confidence', 'pseudo_labeling'),
            ('semi_supervised', 'least_confidence', 'simclr'),
            ('active_learning', 'mc_dropout', 'pseudo_labeling'),
            ('semi_supervised', 'least_confidence', 'auto_encoder'),
        ]

        for (m, u, s) in states:
            arguments.weak_supervision_strategy = m
            arguments.uncertainty_sampling_method = u
            arguments.semi_supervised_method = s
            random.seed(arguments.seed)
            torch.manual_seed(arguments.seed)
            np.random.seed(arguments.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
            torch.manual_seed(arguments.seed)
            main(args=arguments)
    else:
        main(args=arguments)
