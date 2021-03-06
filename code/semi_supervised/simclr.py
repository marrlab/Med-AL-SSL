from active_learning.augmentations_based import UncertaintySamplingAugmentationBased
from active_learning.badge_sampling import UncertaintySamplingBadge
from active_learning.others import UncertaintySamplingOthers
from active_learning.mc_dropout import UncertaintySamplingMCDropout
from data.isic_dataset import ISICDataset
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.plasmodium_dataset import PlasmodiumDataset
from data.retinopathy_dataset import RetinopathyDataset

from utils import create_base_loader, AverageMeter, save_checkpoint, create_loaders, accuracy, Metrics, \
    store_logs, NTXent, get_loss, perform_sampling, \
    create_model_optimizer_simclr, LossPerClassMeter, print_args
import time
import torch
import numpy as np
import pandas as pd
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

'''
SimCLR implementation, adapted from:
Courtesy to: https://github.com/Spijkervet/SimCLR
LARS optimizer, courtesy to: https://github.com/kakaobrain/torchlars
'''


class SimCLR:
    def __init__(self, args, verbose=True, train_feat=False, uncertainty_sampling_method='random_sampling'):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset,
                         'jurkat': JurkatDataset, 'isic': ISICDataset, 'retinopathy': RetinopathyDataset}
        self.model = None
        self.kwargs = {'num_workers': 4, 'pin_memory': False}
        self.uncertainty_sampling_method = uncertainty_sampling_method
        self.train_feat = train_feat

    def train(self):
        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         add_labeled=self.args.add_labeled,
                                                         advanced_transforms=False,
                                                         merged=self.args.merged,
                                                         remove_classes=self.args.remove_classes,
                                                         oversampling=self.args.oversampling,
                                                         unlabeled_subset_ratio=self.args.unlabeled_subset,
                                                         seed=self.args.seed, start_labeled=self.args.start_labeled)

        base_dataset = dataset_class.get_base_dataset_simclr()

        train_loader = create_base_loader(base_dataset, self.kwargs, self.args.simclr_batch_size)

        criterion = NTXent(self.args.simclr_batch_size, self.args.simclr_temperature, torch.device("cuda"))

        self.args.lr = 3e-4
        model, optimizer, _, self.args = create_model_optimizer_simclr(self.args, dataset_class)

        best_loss = np.inf

        for epoch in range(self.args.start_epoch, self.args.simclr_train_epochs):
            model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for i, ((data_x_i, data_x_j), y) in enumerate(train_loader):
                data_x_i = data_x_i.cuda(non_blocking=True)
                data_x_j = data_x_j.cuda(non_blocking=True)

                optimizer.zero_grad()
                h_i, z_i = model(data_x_i)
                h_j, z_j = model(data_x_j)

                loss = criterion(z_i, z_j)

                losses.update(loss.data.item(), data_x_i.size(0))

                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

            is_best = best_loss > losses.avg
            best_loss = min(best_loss, losses.avg)

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_loss,
            }, is_best)

        self.model = model
        return model

    def train_validate_classifier(self):
        if self.uncertainty_sampling_method == 'mc_dropout':
            uncertainty_sampler = UncertaintySamplingMCDropout()
            self.args.weak_supervision_strategy = 'semi_supervised_active_learning'
        elif self.uncertainty_sampling_method == 'augmentations_based':
            uncertainty_sampler = UncertaintySamplingAugmentationBased()
            self.args.weak_supervision_strategy = 'semi_supervised_active_learning'
        elif self.uncertainty_sampling_method == 'random_sampling':
            uncertainty_sampler = None
            self.args.weak_supervision_strategy = "random_sampling"
        elif self.uncertainty_sampling_method == 'badge':
            uncertainty_sampler = UncertaintySamplingBadge()
            self.args.weak_supervision_strategy = 'semi_supervised_active_learning'
        else:
            uncertainty_sampler = UncertaintySamplingOthers(verbose=True,
                                                            uncertainty_sampling_method=
                                                            self.uncertainty_sampling_method)
            self.args.weak_supervision_strategy = 'semi_supervised_active_learning'

        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         add_labeled=self.args.add_labeled,
                                                         advanced_transforms=True,
                                                         merged=self.args.merged,
                                                         remove_classes=self.args.remove_classes,
                                                         oversampling=self.args.oversampling,
                                                         unlabeled_subset_ratio=self.args.unlabeled_subset,
                                                         unlabeled_augmentations=True if
                                                         self.uncertainty_sampling_method == 'augmentations_based'
                                                         else False,
                                                         seed=self.args.seed, k_medoids=self.args.k_medoids,
                                                         k_medoids_model=self.model,
                                                         k_medoids_n_clusters=self.args.k_medoids_n_clusters,
                                                         start_labeled=self.args.start_labeled)

        base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_class.get_dataset()

        train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset, unlabeled_dataset,
                                                                    test_dataset, labeled_indices, unlabeled_indices,
                                                                    self.kwargs, dataset_class.unlabeled_subset_num)

        model = self.model

        criterion = get_loss(self.args, dataset_class.labeled_class_samples, reduction='none')

        optimizer = torch.optim.Adam(model.parameters())

        metrics_per_cycle = pd.DataFrame([])
        metrics_per_epoch = pd.DataFrame([])
        num_class_per_cycle = pd.DataFrame([])

        best_recall, best_report, last_best_epochs = 0, None, 0
        best_model = deepcopy(model)

        print_args(self.args)

        self.args.start_epoch = 0
        current_labeled = dataset_class.start_labeled

        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_loss = self.train_classifier(train_loader, model, criterion, optimizer, last_best_epochs, epoch)
            val_loss, val_report = self.validate_classifier(val_loader, model, last_best_epochs, criterion)

            is_best = val_report['macro avg']['recall'] > best_recall
            last_best_epochs = 0 if is_best else last_best_epochs + 1

            val_report = pd.concat([val_report, train_loss, val_loss], axis=1)
            metrics_per_epoch = pd.concat([metrics_per_epoch, val_report])

            if epoch > self.args.labeled_warmup_epochs and last_best_epochs > self.args.add_labeled_epochs:
                metrics_per_cycle = pd.concat([metrics_per_cycle, best_report])

                train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, uncertainty_sampler, epoch, model, train_loader, unlabeled_loader,
                                     dataset_class, labeled_indices, unlabeled_indices, labeled_dataset,
                                     unlabeled_dataset, test_dataset, self.kwargs, current_labeled)

                current_labeled += self.args.add_labeled
                last_best_epochs = 0

                if self.args.reset_model:
                    model, optimizer, _, self.args = create_model_optimizer_simclr(self.args, dataset_class)
                    optimizer = torch.optim.Adam(model.parameters())

                if self.args.novel_class_detection:
                    num_classes = [np.sum(np.array(base_dataset.targets)[labeled_indices] == i)
                                   for i in range(len(base_dataset.classes))]
                    num_class_per_cycle = pd.concat([num_class_per_cycle,
                                                     pd.DataFrame.from_dict({cls: num_classes[i] for i, cls in
                                                                             enumerate(base_dataset.classes)},
                                                                            orient='index').T])

                criterion = get_loss(self.args, dataset_class.labeled_class_samples, reduction='none')
            else:
                best_recall = val_report['macro avg']['recall'] if is_best else best_recall
                best_report = val_report if is_best else best_report
                best_model = deepcopy(model) if is_best else best_model

            if current_labeled > self.args.stop_labeled:
                break

        if self.args.store_logs:
            store_logs(self.args, metrics_per_cycle)
            store_logs(self.args, metrics_per_epoch, log_type='epoch_wise')
            store_logs(self.args, num_class_per_cycle, log_type='novel_class')

        return best_recall

    def train_classifier(self, train_loader, model, criterion, optimizer, last_best_epochs, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        losses_per_class = LossPerClassMeter(len(train_loader.dataset.dataset.classes))

        end = time.time()

        model.train()

        for i, (data_x, data_y) in enumerate(train_loader):
            data_x = data_x.cuda(non_blocking=True)
            data_y = data_y.cuda(non_blocking=True)

            if self.train_feat:
                output = model.forward_encoder_classifier(data_x)
            else:
                model.eval()
                with torch.no_grad():
                    h = model.forward_encoder(data_x)
                model.train()
                output = model.forward_classifier(h)

            loss = criterion(output, data_y)

            losses_per_class.update(loss.cpu().detach().numpy(), data_y.cpu().numpy())
            loss = torch.sum(loss) / loss.size(0)

            acc = accuracy(output.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(acc.item(), data_x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch Classifier: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Last best epoch {last_best_epoch}'
                      .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses,
                              last_best_epoch=last_best_epochs))

        return pd.DataFrame.from_dict({f'{k}-train-loss': losses_per_class.avg[i]
                                       for i, k in enumerate(train_loader.dataset.dataset.classes)}, orient='index').T

    def validate_classifier(self, val_loader, model, last_best_epochs, criterion):
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
                data_x = data_x.cuda(non_blocking=True)
                data_y = data_y.cuda(non_blocking=True)

                if self.train_feat:
                    output = model.forward_encoder_classifier(data_x)
                else:
                    h = model.forward_encoder(data_x)
                    output = model.forward_classifier(h)

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

                if i % self.args.print_freq == 0:
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
