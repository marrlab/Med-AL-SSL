from active_learning.augmentations_based import UncertaintySamplingAugmentationBased
from active_learning.learning_loss import LearningLoss
from data.config.cifar10_config import set_cifar_configs
from options.train_options import get_arguments
import os
import time
from copy import deepcopy

import random
import pandas as pd

from semi_supervised.fixmatch import FixMatch

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np

from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.plasmodium_dataset import PlasmodiumDataset
from data.config.matek_config import set_matek_configs
from data.config.jurkat_config import set_jurkat_configs
from data.config.plasmodium_config import set_plasmodium_configs

from utils import save_checkpoint, AverageMeter, accuracy, create_loaders, print_args, \
    create_model_optimizer_scheduler, get_loss, resume_model, set_model_name, perform_sampling, LossPerClassMeter, \
    load_pretrained, novel_class_detected
from utils import Metrics, store_logs
from active_learning.entropy_based import UncertaintySamplingEntropyBased
from active_learning.mc_dropout import UncertaintySamplingMCDropout
from semi_supervised.pseudo_labeling import PseudoLabeling
from semi_supervised.auto_encoder import AutoEncoder
from semi_supervised.simclr import SimCLR
from semi_supervised.auto_encoder_cl import AutoEncoderCl

arguments = get_arguments()
datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset, 'jurkat': JurkatDataset}
configs = {'matek': set_matek_configs, 'jurkat': set_jurkat_configs,
           'plasmodium': set_plasmodium_configs, 'cifar10': set_cifar_configs}


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(args.seed)

    args.name = set_model_name(args)
    args = configs[args.dataset](args)

    if args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'auto_encoder':
        auto_encoder = AutoEncoder(args)
        auto_encoder.train()
        best_acc = auto_encoder.train_validate_classifier()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'auto_encoder_cl':
        auto_encoder_cl = AutoEncoderCl(args)
        best_acc = auto_encoder_cl.main()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'simclr':
        simclr = SimCLR(args, train_feat=True, uncertainty_sampling_method='random_sampling')
        simclr.train()
        best_acc = simclr.train_validate_classifier()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'fixmatch':
        fixmatch = FixMatch(args)
        best_acc = fixmatch.main()
        return best_acc
    elif args.weak_supervision_strategy == 'active_learning' and args.uncertainty_sampling_method == 'learning_loss':
        learning_loss = LearningLoss(args)
        best_acc = learning_loss.main()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'simclr_with_al':
        simclr = SimCLR(args, train_feat=True, uncertainty_sampling_method='augmentations_based')
        simclr.train()
        best_acc = simclr.train_validate_classifier()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'auto_encoder_with_al':
        auto_encoder = AutoEncoder(args, uncertainty_sampling_method='augmentations_based')
        auto_encoder.train()
        best_acc = auto_encoder.train_validate_classifier()
        return best_acc
    elif args.weak_supervision_strategy == 'semi_supervised' and args.semi_supervised_method == 'fixmatch_with_al':
        fixmatch = FixMatch(args, uncertainty_sampling_method='augmentations_based')
        best_acc = fixmatch.main()
        return best_acc

    if args.uncertainty_sampling_method == 'mc_dropout':
        uncertainty_sampler = UncertaintySamplingMCDropout()
    elif args.uncertainty_sampling_method == 'augmentations_based':
        uncertainty_sampler = UncertaintySamplingAugmentationBased()
    else:
        uncertainty_sampler = UncertaintySamplingEntropyBased(verbose=True,
                                                              uncertainty_sampling_method=args.
                                                              uncertainty_sampling_method)

    dataset_class = datasets[args.dataset](root=args.root,
                                           add_labeled=args.add_labeled,
                                           advanced_transforms=True,
                                           unlabeled_subset_ratio=args.unlabeled_subset,
                                           oversampling=args.oversampling,
                                           merged=args.merged,
                                           remove_classes=args.remove_classes,
                                           unlabeled_augmentations=True if args.weak_supervision_strategy ==
                                           'active_learning' and args.
                                           uncertainty_sampling_method == 'augmentations_based' else False,
                                           seed=args.seed, start_labeled=args.start_labeled)

    base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
        dataset_class.get_dataset()

    kwargs = {'num_workers': 2, 'pin_memory': False}
    train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset,
                                                                labeled_indices, unlabeled_indices, kwargs,
                                                                dataset_class.unlabeled_subset_num)

    pseudo_labeler = PseudoLabeling()

    model, optimizer, scheduler = create_model_optimizer_scheduler(args, dataset_class)

    if args.load_pretrained:
        model = load_pretrained(model)

    if args.resume:
        model, _, _ = resume_model(args, model)

    criterion = get_loss(args, dataset_class.labeled_class_samples, reduction='none')

    current_labeled = dataset_class.start_labeled
    metrics_per_cycle = pd.DataFrame([])
    metrics_per_epoch = pd.DataFrame([])
    best_model = deepcopy(model)

    print_args(args)

    print('Starting training..')

    best_recall, best_report, last_best_epochs = 0, None, 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, last_best_epochs, args)
        val_loss, val_report = validate(val_loader, model, criterion, last_best_epochs, args)

        is_best = val_report['macro avg']['recall'] > best_recall
        last_best_epochs = 0 if is_best else last_best_epochs + 1

        val_report = pd.concat([val_report, train_loss, val_loss], axis=1)
        metrics_per_epoch = pd.concat([metrics_per_epoch, val_report])

        if epoch > args.labeled_warmup_epochs and last_best_epochs > args.add_labeled_epochs:
            metrics_per_cycle = pd.concat([metrics_per_cycle, best_report])

            train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                perform_sampling(args, uncertainty_sampler, pseudo_labeler,
                                 epoch, model, train_loader, unlabeled_loader,
                                 dataset_class, labeled_indices,
                                 unlabeled_indices, labeled_dataset,
                                 unlabeled_dataset,
                                 test_dataset, kwargs, current_labeled,
                                 model)
            current_labeled += args.add_labeled
            # best_recall, best_report, last_best_epochs = 0, None, 0
            last_best_epochs = 0

            if args.reset_model:
                model, optimizer, scheduler = create_model_optimizer_scheduler(args, dataset_class)

            if args.novel_class_detection:
                if novel_class_detected(train_loader, dataset_class, args):
                    break

            criterion = get_loss(args, dataset_class.labeled_class_samples, reduction='none')
        else:
            best_recall = val_report['macro avg']['recall'] if is_best else best_recall
            best_report = val_report if is_best else best_report
            best_model = deepcopy(model) if is_best else best_model

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': best_model.state_dict(),
            'best_recall': best_recall,
        }, is_best)

        if current_labeled > args.stop_labeled:
            break

    print(best_report)

    if args.store_logs:
        store_logs(args, metrics_per_cycle)
        store_logs(args, metrics_per_epoch, epoch_wise=True)


def train(train_loader, model, criterion, optimizer, epoch, last_best_epochs, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    losses_per_class = LossPerClassMeter(len(train_loader.dataset.dataset.classes))

    model.train()

    end = time.time()
    for i, (data_x, data_y) in enumerate(train_loader):
        data_y = data_y.cuda(non_blocking=True)
        data_x = data_x.cuda(non_blocking=True)

        optimizer.zero_grad()
        output, _, _ = model(data_x)
        loss = criterion(output, data_y)

        losses_per_class.update(loss.cpu().detach().numpy(), data_y.cpu().numpy())
        loss = torch.sum(loss) / loss.size(0)

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

    return pd.DataFrame.from_dict({f'{k}-train-loss': losses_per_class.avg[i]
                                   for i, k in enumerate(train_loader.dataset.dataset.classes)}, orient='index').T


def validate(val_loader, model, criterion, last_best_epochs, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    metrics = Metrics()
    losses_per_class = LossPerClassMeter(len(val_loader.dataset.dataset.classes))

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (data_x, data_y) in enumerate(val_loader):
            data_y = data_y.cuda(non_blocking=True)
            data_x = data_x.cuda(non_blocking=True)

            output, _, _ = model(data_x)
            loss = criterion(output, data_y)

            losses_per_class.update(loss.cpu().detach().numpy(), data_y.cpu().numpy())
            loss = torch.sum(loss) / loss.size(0)

            acc = accuracy(output.data, data_y, topk=(1, 2,))
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

    report = metrics.get_report(target_names=val_loader.dataset.dataset.classes)
    print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1} * Acc@5 {top5.avg:.3f}\t'
          .format(report['macro avg']['precision'], report['macro avg']['recall'], top1=top1, top5=top5))

    return pd.DataFrame.from_dict({f'{k}-val-loss': losses_per_class.avg[i]
                                   for i, k in enumerate(val_loader.dataset.dataset.classes)}, orient='index').T, \
        pd.DataFrame.from_dict(report)


if __name__ == '__main__':
    if arguments.run_batch:
        states = [
            # ('active_learning', 'least_confidence', 'pseudo_labeling'),
            # ('active_learning', 'margin_confidence', 'pseudo_labeling'),
            # ('active_learning', 'ratio_confidence', 'pseudo_labeling'),
            # ('active_learning', 'entropy_based', 'pseudo_labeling'),
            # ('active_learning', 'mc_dropout', 'pseudo_labeling'),
            # ('active_learning', 'learning_loss', 'pseudo_labeling'),
            # ('active_learning', 'augmentations_based', 'pseudo_labeling'),
            # ('random_sampling', 'least_confidence', 'pseudo_labeling'),
            # ('semi_supervised', 'least_confidence', 'pseudo_labeling'),
            ('semi_supervised', 'least_confidence', 'simclr'),
            # ('semi_supervised', 'least_confidence', 'auto_encoder'),
            # ('semi_supervised', 'least_confidence', 'auto_encoder_with_al'),
            # ('semi_supervised', 'least_confidence', 'auto_encoder_cl'),
            # ('semi_supervised', 'least_confidence', 'fixmatch'),
            # ('semi_supervised', 'least_confidence', 'fixmatch_with_al'),
            ('semi_supervised', 'least_confidence', 'simclr_with_al'),
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
