from torch.utils.data import DataLoader

from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.plasmodium_dataset import PlasmodiumDataset

import torch
import time

from utils import create_model_optimizer_scheduler, AverageMeter, accuracy, Metrics, perform_sampling, \
    store_logs, save_checkpoint, get_loss, LossPerClassMeter

import pandas as pd
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

'''
FixMatch implementation, adapted from:
Courtesy to: https://github.com/kekmodel/FixMatch-pytorch
'''


class FixMatch:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset,
                         'jurkat': JurkatDataset}
        self.model = None
        self.kwargs = {'num_workers': 16, 'pin_memory': False, 'drop_last': True}

    def main(self):
        dataset_cls = self.datasets[self.args.dataset](root=self.args.root,
                                                       labeled_ratio=self.args.labeled_ratio_start,
                                                       add_labeled_ratio=self.args.add_labeled_ratio,
                                                       advanced_transforms=True,
                                                       unlabeled_subset_ratio=self.args.unlabeled_subset,
                                                       expand_labeled=self.args.fixmatch_k_img,
                                                       expand_unlabeled=self.args.fixmatch_k_img*self.args.fixmatch_mu,
                                                       oversampling=self.args.oversampling)

        base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_cls.get_dataset()

        labeled_dataset, unlabeled_dataset = dataset_cls.get_datasets_fixmatch(base_dataset, labeled_indices,
                                                                               unlabeled_indices)

        model, optimizer, scheduler = create_model_optimizer_scheduler(self.args, dataset_cls, optimizer='sgd',
                                                                       scheduler='cosine_schedule_with_warmup',
                                                                       load_optimizer_scheduler=True)

        labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=self.args.batch_size,
                                    shuffle=True, **self.kwargs)
        unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, **self.kwargs)
        val_loader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size,
                                shuffle=True, **self.kwargs)

        criterion_labeled = get_loss(self.args, dataset_cls.labeled_class_samples, reduction='none')
        criterion_unlabeled = get_loss(self.args, dataset_cls.labeled_class_samples, reduction='none')

        criterions = {'labeled': criterion_labeled, 'unlabeled': criterion_unlabeled}

        model.zero_grad()

        best_recall, best_report, last_best_epochs = 0, None, 0
        best_model = deepcopy(model)

        metrics_per_ratio = pd.DataFrame([])
        metrics_per_epoch = pd.DataFrame([])
        self.args.start_epoch = 0
        self.args.weak_supervision_strategy = "random_sampling"
        current_labeled_ratio = self.args.labeled_ratio_start

        for epoch in range(self.args.start_epoch, self.args.fixmatch_epochs):
            train_loader = zip(labeled_loader, unlabeled_loader)
            train_loss = self.train(train_loader, model, optimizer, epoch, len(labeled_loader), criterions,
                                    base_dataset.classes, last_best_epochs)
            val_loss, val_report = self.validate(val_loader, model, last_best_epochs, criterions)

            is_best = val_report['macro avg']['recall'] > best_recall
            last_best_epochs = 0 if is_best else last_best_epochs + 1

            val_report = pd.concat([val_report, train_loss, val_loss], axis=1)
            metrics_per_epoch = pd.concat([metrics_per_epoch, val_report])

            if epoch > self.args.labeled_warmup_epochs and last_best_epochs > self.args.add_labeled_epochs:
                metrics_per_ratio = pd.concat([metrics_per_ratio, best_report])

                unlabeled_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, None, None,
                                     epoch, model, train_loader, unlabeled_loader,
                                     dataset_cls, labeled_indices,
                                     unlabeled_indices, labeled_dataset,
                                     unlabeled_dataset,
                                     test_dataset, self.kwargs, current_labeled_ratio,
                                     None)

                labeled_dataset, unlabeled_dataset = dataset_cls.get_datasets_fixmatch(base_dataset, labeled_indices,
                                                                                       unlabeled_indices)

                labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=self.args.batch_size,
                                            shuffle=True, **self.kwargs)
                unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=self.args.batch_size,
                                              shuffle=True, **self.kwargs)

                current_labeled_ratio += self.args.add_labeled_ratio
                best_recall, best_report, last_best_epochs = 0, None, 0

                if self.args.reset_model:
                    model, optimizer, scheduler = create_model_optimizer_scheduler(self.args, dataset_cls,
                                                                                   optimizer='sgd',
                                                                                   scheduler='cosine_schedule_with_'
                                                                                             'warmup',
                                                                                   load_optimizer_scheduler=True)

                criterion_labeled = get_loss(self.args, dataset_cls.labeled_class_samples, reduction='none')
                criterion_unlabeled = get_loss(self.args, dataset_cls.labeled_class_samples, reduction='none')
                criterions = {'labeled': criterion_labeled, 'unlabeled': criterion_unlabeled}
            else:
                best_recall = val_report['macro avg']['recall'] if is_best else best_recall
                best_report = val_report if is_best else best_report
                best_model = deepcopy(model) if is_best else best_model

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_prec1': best_recall,
            }, is_best)

            if current_labeled_ratio > self.args.labeled_ratio_stop:
                break

        if self.args.store_logs:
            store_logs(self.args, metrics_per_ratio)
            store_logs(self.args, metrics_per_epoch, epoch_wise=True)

        return best_recall

    def train(self, train_loader, model, optimizer, epoch, loaders_len, criterions, classes, last_best_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        losses_per_class = LossPerClassMeter(len(classes))

        end = time.time()

        model.train()

        for i, (data_labeled, data_unlabeled) in enumerate(train_loader):
            data_x, data_y = data_labeled
            data_x, data_y = data_x.cuda(non_blocking=True), data_y.cuda(non_blocking=True)

            (data_w, data_s), _ = data_unlabeled
            data_w, data_s = data_w.cuda(non_blocking=True), data_s.cuda(non_blocking=True)

            inputs = torch.cat((data_x, data_w, data_s))
            logits, _, _ = model(inputs)
            logits_labeled = logits[:self.args.batch_size]
            logits_unlabeled_w, logits_unlabeled_s = logits[self.args.batch_size:].chunk(2)
            del logits

            loss_labeled = criterions['labeled'](logits_labeled, data_y)

            losses_per_class.update(loss_labeled.cpu().detach().numpy(), data_y.cpu().numpy())
            loss_labeled = torch.sum(loss_labeled) / loss_labeled.size(0)

            pseudo_label = torch.softmax(logits_unlabeled_w.detach_(), dim=-1)
            max_probs, data_y_unlabeled = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.args.fixmatch_threshold).float()

            loss_unlabeled = (criterions['unlabeled'](logits_unlabeled_s, data_y_unlabeled) * mask).mean()

            loss = loss_labeled + self.args.fixmatch_lambda_u * loss_unlabeled

            acc = accuracy(logits_labeled.data, data_y, topk=(1,))[0]
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
                      .format(epoch, i, loaders_len,
                              batch_time=batch_time, loss=losses, last_best_epoch=last_best_epochs))

        return pd.DataFrame.from_dict({f'{k}-train-loss': losses_per_class.avg[i]
                                       for i, k in enumerate(classes)}, orient='index').T

    def validate(self, val_loader, model, last_best_epochs, criterions):
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

                output, _, _ = model(data_x)

                loss = criterions['labeled'](output, data_y)

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
