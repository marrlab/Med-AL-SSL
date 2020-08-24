import time

from copy import deepcopy

import torch

from active_learning.uncertainty_sampling import UncertaintySampling
from data.cifar100_dataset import Cifar100Dataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.matek_dataset import MatekDataset
from model.loss_net import LossNet
from utils import create_loaders, create_model_optimizer_scheduler, create_model_optimizer_loss_net, get_loss, \
    print_args, loss_module_objective_func, AverageMeter, accuracy, Metrics, store_logs, save_checkpoint, \
    perform_sampling

import numpy as np

'''
Learning Loss for Active Learning (https://arxiv.org/pdf/1905.03677.pdf)
Code adapted from: https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''


class LearningLoss:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset,
                         'jurkat': JurkatDataset}
        self.model = None
        self.kwargs = {'num_workers': 2, 'pin_memory': False, 'drop_last': True}

    def main(self):
        dataset_cl = self.datasets[self.args.dataset](root=self.args.root,
                                                      labeled_ratio=self.args.labeled_ratio_start,
                                                      add_labeled_ratio=self.args.add_labeled_ratio,
                                                      advanced_transforms=True,
                                                      unlabeled_subset_ratio=self.args.unlabeled_subset,
                                                      oversampling=self.args.oversampling)

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

        criterion_backbone = get_loss(self.args, base_dataset, reduction='none')

        criterions = {'backbone': criterion_backbone, 'module': loss_module_objective_func}

        uncertainty_sampler = UncertaintySampling(verbose=True,
                                                  uncertainty_sampling_method=self.args.uncertainty_sampling_method)

        last_best_epochs = 0
        current_labeled_ratio = self.args.labeled_ratio_start
        acc_ratio = {}
        best_model = deepcopy(model_backbone)

        print_args(self.args)

        best_acc1, best_acc5, best_prec1, best_recall1, best_f1, best_confusion_mat = 0, 0, 0, 0, 0, None

        for epoch in range(self.args.start_epoch, self.args.epochs):
            models = self.train(train_loader, models, optimizers, criterions, epoch, last_best_epochs)
            acc, acc5, (prec, recall, f1, _), confusion_mat, roc_auc_curve = self.validate(val_loader, models,
                                                                                           criterions, last_best_epochs)

            is_best = recall > best_recall1
            last_best_epochs = 0 if is_best else last_best_epochs + 1
            best_model = deepcopy(model_backbone) if is_best else best_model

            if epoch > self.args.labeled_warmup_epochs and epoch % self.args.add_labeled_epochs == 0:
                acc_ratio.update({np.round(current_labeled_ratio, decimals=2):
                                 [best_acc1, best_acc5, best_prec1, best_recall1, best_f1,
                                 best_confusion_mat.tolist(), roc_auc_curve]})

                train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, uncertainty_sampler, None,
                                     epoch, models, train_loader, unlabeled_loader,
                                     dataset_cl, labeled_indices,
                                     unlabeled_indices, labeled_dataset,
                                     unlabeled_dataset,
                                     test_dataset, self.kwargs, current_labeled_ratio,
                                     None)

                current_labeled_ratio += self.args.add_labeled_ratio
                best_acc1, best_acc5, best_prec1, best_recall1, best_f1, best_confusion_mat = 0, 0, 0, 0, 0, None

                if self.args.reset_model:
                    model_backbone, optimizer_backbone, scheduler_backbone = \
                        create_model_optimizer_scheduler(self.args, dataset_cl)
                    model_module, optimizer_module = create_model_optimizer_loss_net()
                    models = {'backbone': model_backbone, 'module': model_module}
                    optimizers = {'backbone': optimizer_backbone, 'module': optimizer_module}
            else:
                best_acc1 = max(acc, best_acc1)
                best_prec1 = max(prec, best_prec1)
                best_recall1 = max(recall, best_recall1)
                best_acc5 = max(acc5, best_acc5)
                best_f1 = max(f1, best_f1)
                best_confusion_mat = confusion_mat if is_best else best_confusion_mat

            if current_labeled_ratio > self.args.labeled_ratio_stop:
                break

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model_backbone.state_dict(),
                'best_prec1': best_acc1,
            }, is_best)

        if self.args.store_logs:
            store_logs(self.args, acc_ratio)

        return best_acc1

    def train(self, train_loader, models, optimizers, criterions, epoch, last_best_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

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

        return models

    def validate(self, val_loader, models, criterions, last_best_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        metrics = Metrics()

        models['backbone'].eval()
        models['module'].eval()

        end = time.time()

        with torch.no_grad():
            for i, (data_x, data_y) in enumerate(val_loader):
                data_y = data_y.cuda(non_blocking=True)
                data_x = data_x.cuda(non_blocking=True)

                output, _, _ = models['backbone'](data_x)
                loss = criterions['backbone'](output, data_y)
                loss = torch.sum(loss) / loss.size(0)

                acc = accuracy(output.data, data_y, topk=(1, 5,))
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

        (prec, recall, f1, _) = metrics.get_metrics()
        confusion_matrix = metrics.get_confusion_matrix()
        roc_auc_curve = metrics.get_roc_auc_curve()
        print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1} * Acc@5 {top5.avg:.3f}\t * Roc_Auc {2}\t'
              .format(prec, recall, roc_auc_curve, top1=top1, top5=top5))

        return top1.avg, top5.avg, (prec, recall, f1, _), confusion_matrix, roc_auc_curve
