from data.isic_dataset import ISICDataset
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.plasmodium_dataset import PlasmodiumDataset
from utils import create_base_loader, AverageMeter, save_checkpoint, create_loaders, accuracy, Metrics, \
    store_logs, get_loss, perform_sampling, create_model_optimizer_autoencoder, LossPerClassMeter
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from pytorch_msssim import SSIM

torch.autograd.set_detect_anomaly(True)


'''
Based on the Convolutional Autoencoder (CAE):
https://www.nature.com/articles/s42256-020-0153-x
'''


class AutoEncoderCl:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset,
                         'jurkat': JurkatDataset, 'isic': ISICDataset}
        self.model = None
        self.kwargs = {'num_workers': 16, 'pin_memory': False}

    def main(self):
        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         add_labeled=self.args.add_labeled,
                                                         advanced_transforms=True,
                                                         merged=self.args.merged,
                                                         remove_classes=self.args.remove_classes,
                                                         oversampling=self.args.oversampling,
                                                         unlabeled_subset_ratio=self.args.unlabeled_subset,
                                                         seed=self.args.seed, start_labeled=self.args.start_labeled)

        _, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_class.get_dataset()

        labeled_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset, unlabeled_dataset,
                                                                      test_dataset, labeled_indices, unlabeled_indices,
                                                                      self.kwargs, dataset_class.unlabeled_subset_num)

        base_dataset = dataset_class.get_base_dataset_autoencoder()

        base_loader = create_base_loader(base_dataset, self.kwargs, self.args.batch_size)

        reconstruction_loss_log = []

        bce_loss = nn.BCELoss().cuda()
        l1_loss = nn.L1Loss()
        l2_loss = nn.MSELoss()
        ssim_loss = SSIM(size_average=True, data_range=1.0, nonnegative_ssim=True)

        criterions_reconstruction = {'bce': bce_loss, 'l1': l1_loss, 'l2': l2_loss, 'ssim': ssim_loss}
        criterion_cl = get_loss(self.args, dataset_class.labeled_class_samples, reduction='none')

        model, optimizer, self.args = create_model_optimizer_autoencoder(self.args, dataset_class)

        best_loss = np.inf

        metrics_per_cycle = pd.DataFrame([])
        metrics_per_epoch = pd.DataFrame([])
        num_class_per_cycle = pd.DataFrame([])

        best_recall, best_report, last_best_epochs = 0, None, 0
        best_model = deepcopy(model)

        self.args.start_epoch = 0
        self.args.weak_supervision_strategy = "random_sampling"
        current_labeled = dataset_class.start_labeled

        for epoch in range(self.args.start_epoch, self.args.epochs):
            cl_train_loss, losses_avg_reconstruction, losses_reconstruction = \
                self.train(labeled_loader, model, criterion_cl, optimizer, last_best_epochs, epoch,
                           criterions_reconstruction, base_loader)
            val_loss, val_report = self.validate(val_loader, model, last_best_epochs, criterion_cl)

            reconstruction_loss_log.append(losses_avg_reconstruction.tolist())
            best_loss = min(best_loss, losses_reconstruction.avg)

            is_best = val_report['macro avg']['recall'] > best_recall
            last_best_epochs = 0 if is_best else last_best_epochs + 1

            val_report = pd.concat([val_report, cl_train_loss, val_loss], axis=1)
            metrics_per_epoch = pd.concat([metrics_per_epoch, val_report])

            if epoch > self.args.labeled_warmup_epochs and last_best_epochs > self.args.add_labeled_epochs:
                metrics_per_cycle = pd.concat([metrics_per_cycle, best_report])

                labeled_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, None, None,
                                     epoch, model, labeled_loader, unlabeled_loader,
                                     dataset_class, labeled_indices,
                                     unlabeled_indices, labeled_dataset,
                                     unlabeled_dataset,
                                     test_dataset, self.kwargs, current_labeled,
                                     model)

                current_labeled += self.args.add_labeled
                last_best_epochs = 0

                if self.args.reset_model:
                    model, optimizer, self.args = create_model_optimizer_autoencoder(self.args, dataset_class)

                if self.args.novel_class_detection:
                    num_classes = [np.sum(np.array(base_dataset.targets)[labeled_indices] == i)
                                   for i in range(len(base_dataset.classes))]
                    num_class_per_cycle = pd.concat([num_class_per_cycle,
                                                     pd.DataFrame.from_dict({cls: num_classes[i] for i, cls in
                                                                             enumerate(base_dataset.classes)},
                                                                            orient='index').T])

                criterion_cl = get_loss(self.args, dataset_class.labeled_class_samples, reduction='none')
            else:
                best_recall = val_report['macro avg']['recall'] if is_best else best_recall
                best_report = val_report if is_best else best_report
                best_model = deepcopy(model) if is_best else best_model

            if current_labeled > self.args.stop_labeled:
                break

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_recall,
            }, is_best)

        if self.args.store_logs:
            store_logs(self.args, pd.DataFrame(reconstruction_loss_log, columns=['bce', 'l1', 'l2', 'ssim']),
                       log_type='ae_loss')
            store_logs(self.args, metrics_per_cycle)
            store_logs(self.args, metrics_per_epoch, log_type='epoch_wise')
            store_logs(self.args, num_class_per_cycle, log_type='novel_class')

        self.model = model
        return model

    def train(self, labeled_loader, model, criterion_cl, optimizer, last_best_epochs, epoch,
              criterions_reconstruction, base_loader):
        model.train()
        batch_time = AverageMeter()
        losses_reconstruction = AverageMeter()
        losses_sum_reconstruction = np.zeros(len(criterions_reconstruction.keys()))

        end = time.time()
        for i, (data_x, data_y) in enumerate(base_loader):
            data_x = data_x.cuda(non_blocking=True)

            output = model(data_x)

            losses_alt = np.array([v(output, data_x).cpu().detach().data.item() for v in
                                   criterions_reconstruction.values()])
            losses_alt[-1] = 1 - losses_alt[-1]
            losses_sum_reconstruction = losses_sum_reconstruction + losses_alt
            loss = criterions_reconstruction['l2'](output, data_x) + \
                (1 - criterions_reconstruction['ssim'](output, data_x))

            losses_reconstruction.update(loss.data.item(), data_x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(epoch, i, len(base_loader), batch_time=batch_time, loss=losses_reconstruction))

        losses_avg_reconstruction = losses_sum_reconstruction / len(base_loader)

        batch_time = AverageMeter()
        losses_cl = AverageMeter()
        top1 = AverageMeter()
        losses_per_class_cl = LossPerClassMeter(len(labeled_loader.dataset.dataset.classes))

        end = time.time()

        model.train()

        for i, (data_x, data_y) in enumerate(labeled_loader):
            data_x = data_x.cuda(non_blocking=True)
            data_y = data_y.cuda(non_blocking=True)

            output = model.forward_encoder_classifier(data_x)

            loss = criterion_cl(output, data_y)

            losses_per_class_cl.update(loss.cpu().detach().numpy(), data_y.cpu().numpy())
            loss = (torch.sum(loss) / loss.size(0)) * 100

            acc = accuracy(output.data, data_y, topk=(1,))[0]
            losses_cl.update(loss.data.item(), data_x.size(0))
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
                      .format(epoch, i, len(labeled_loader), batch_time=batch_time, loss=losses_cl,
                              last_best_epoch=last_best_epochs))

        return pd.DataFrame.from_dict({f'{k}-train-loss': losses_per_class_cl.avg[i]
                                       for i, k in enumerate(labeled_loader.dataset.dataset.classes)},
                                      orient='index').T, losses_avg_reconstruction, losses_reconstruction

    def validate(self, val_loader, model, last_best_epochs, criterion):
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

                output = model.forward_encoder_classifier(data_x)

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
