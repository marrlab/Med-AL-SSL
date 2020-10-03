import time
import torch

from active_learning.entropy_based import UncertaintySamplingEntropyBased
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.plasmodium_dataset import PlasmodiumDataset
from model.loss_net import LossNet
from utils import create_loaders, create_model_optimizer_scheduler, create_model_optimizer_loss_net, get_loss, \
    print_args, loss_module_objective_func, AverageMeter, accuracy, Metrics, store_logs, save_checkpoint, \
    perform_sampling, LossPerClassMeter

import pandas as pd
from copy import deepcopy

'''
Learning Loss for Active Learning (https://arxiv.org/pdf/1905.03677.pdf)
Code adapted from: https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''


class LearningLoss:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset,
                         'jurkat': JurkatDataset}
        self.model = None
        self.kwargs = {'num_workers': 2, 'pin_memory': False, 'drop_last': True}

    def main(self):
        dataset_cl = self.datasets[self.args.dataset](root=self.args.root,
                                                      labeled_ratio=self.args.labeled_ratio_start,
                                                      add_labeled_ratio=self.args.add_labeled_ratio,
                                                      advanced_transforms=True,
                                                      merged=self.args.merged,
                                                      remove_classes=self.args.remove_classes,
                                                      oversampling=self.args.oversampling,
                                                      unlabeled_subset_ratio=self.args.unlabeled_subset)

        base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_cl.get_dataset()

        train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset, unlabeled_dataset,
                                                                    test_dataset, labeled_indices, unlabeled_indices,
                                                                    self.kwargs, dataset_cl.unlabeled_subset_num)

        model_backbone, optimizer_backbone, _ = create_model_optimizer_scheduler(self.args, dataset_cl)
        model_module = LossNet().cuda()
        optimizer_module = torch.optim.Adam(model_module.parameters())

        models = {'backbone': model_backbone, 'module': model_module}
        optimizers = {'backbone': optimizer_backbone, 'module': optimizer_module}

        criterion_backbone = get_loss(self.args, dataset_cl.labeled_class_samples, reduction='none')

        criterions = {'backbone': criterion_backbone, 'module': loss_module_objective_func}

        uncertainty_sampler = UncertaintySamplingEntropyBased(verbose=True,
                                                              uncertainty_sampling_method=self.args.
                                                              uncertainty_sampling_method)

        current_labeled_ratio = self.args.labeled_ratio_start
        metrics_per_ratio = pd.DataFrame([])
        metrics_per_epoch = pd.DataFrame([])

        print_args(self.args)

        best_recall, best_report, last_best_epochs = 0, None, 0
        best_model = deepcopy(models['backbone'])

        for epoch in range(self.args.start_epoch, self.args.epochs):
            train_loss = self.train(train_loader, models, optimizers, criterions, epoch, last_best_epochs)
            val_loss, val_report = self.validate(val_loader, models, criterions, last_best_epochs)

            is_best = val_report['macro avg']['recall'] > best_recall
            last_best_epochs = 0 if is_best else last_best_epochs + 1

            val_report = pd.concat([val_report, train_loss, val_loss], axis=1)
            metrics_per_epoch = pd.concat([metrics_per_epoch, val_report])

            if epoch > self.args.labeled_warmup_epochs and last_best_epochs > self.args.add_labeled_epochs:
                metrics_per_ratio = pd.concat([metrics_per_ratio, best_report])

                train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, uncertainty_sampler, None,
                                     epoch, models, train_loader, unlabeled_loader,
                                     dataset_cl, labeled_indices,
                                     unlabeled_indices, labeled_dataset,
                                     unlabeled_dataset,
                                     test_dataset, self.kwargs, current_labeled_ratio,
                                     None)

                current_labeled_ratio += self.args.add_labeled_ratio
                last_best_epochs = 0

                if self.args.reset_model:
                    model_backbone, optimizer_backbone, scheduler_backbone = \
                        create_model_optimizer_scheduler(self.args, dataset_cl)
                    model_module, optimizer_module = create_model_optimizer_loss_net()
                    models = {'backbone': model_backbone, 'module': model_module}
                    optimizers = {'backbone': optimizer_backbone, 'module': optimizer_module}

                criterion_backbone = get_loss(self.args, dataset_cl.labeled_class_samples, reduction='none')
                criterions = {'backbone': criterion_backbone, 'module': loss_module_objective_func}
            else:
                best_recall = val_report['macro avg']['recall'] if is_best else best_recall
                best_report = val_report if is_best else best_report
                best_model = deepcopy(models['backbone']) if is_best else best_model

            if current_labeled_ratio > self.args.labeled_ratio_stop:
                break

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model_backbone.state_dict(),
                'best_prec1': best_recall,
            }, is_best)

        if self.args.store_logs:
            store_logs(self.args, metrics_per_ratio)
            store_logs(self.args, metrics_per_epoch, epoch_wise=True)

        return best_recall

    def train(self, train_loader, models, optimizers, criterions, epoch, last_best_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        losses_per_class = LossPerClassMeter(len(train_loader.dataset.dataset.classes))

        models['backbone'].train()
        models['module'].train()

        end = time.time()

        for i, (data_x, data_y) in enumerate(train_loader):
            data_y = data_y.cuda(non_blocking=True)
            data_x = data_x.cuda(non_blocking=True)

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            output, _, features = models['backbone'](data_x)
            target_loss = criterions['backbone'](output, data_y)

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            losses_per_class.update(target_loss.cpu().detach().numpy(), data_y.cpu().numpy())
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

            m_module_loss = criterions['module'](pred_loss, target_loss)
            loss = m_backbone_loss + self.args.learning_loss_weight * m_module_loss

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

            acc = accuracy(output.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(acc.item(), data_x.size(0))

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

    def validate(self, val_loader, models, criterions, last_best_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        metrics = Metrics()
        losses_per_class = LossPerClassMeter(len(val_loader.dataset.dataset.classes))

        models['backbone'].eval()
        models['module'].eval()

        end = time.time()

        with torch.no_grad():
            for i, (data_x, data_y) in enumerate(val_loader):
                data_y = data_y.cuda(non_blocking=True)
                data_x = data_x.cuda(non_blocking=True)

                output, _, _ = models['backbone'](data_x)
                loss = criterions['backbone'](output, data_y)

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
