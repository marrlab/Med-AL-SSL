from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.matek_dataset import MatekDataset
from data.cifar100_dataset import Cifar100Dataset
from utils import create_base_loader, AverageMeter, save_checkpoint, create_loaders, accuracy, Metrics, \
    store_logs, get_loss, perform_sampling, create_model_optimizer_autoencoder
import time
import torch
import torch.nn as nn
import numpy as np

torch.autograd.set_detect_anomaly(True)


class AutoEncoder:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset,
                         'jurkat': JurkatDataset}
        self.model = None
        self.kwargs = {'num_workers': 2, 'pin_memory': False}

    def train(self):
        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         labeled_ratio=self.args.labeled_ratio_start,
                                                         add_labeled_ratio=self.args.add_labeled_ratio,
                                                         oversampling=self.args.oversampling)

        base_dataset = dataset_class.get_base_dataset_autoencoder()

        train_loader = create_base_loader(base_dataset, self.kwargs, self.args.batch_size)

        criterion = nn.BCELoss().cuda()
        model, optimizer, self.args = create_model_optimizer_autoencoder(self.args, dataset_class)

        best_loss = np.inf

        for epoch in range(self.args.start_epoch, self.args.autoencoder_train_epochs):
            model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for i, (data_x, data_y) in enumerate(train_loader):
                data_x = data_x.cuda(non_blocking=True)

                output = model(data_x)
                loss = criterion(output, data_x)

                losses.update(loss.data.item(), data_x.size(0))

                optimizer.zero_grad()
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
        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         labeled_ratio=self.args.labeled_ratio_start,
                                                         add_labeled_ratio=self.args.add_labeled_ratio,
                                                         oversampling=self.args.oversampling)

        base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_class.get_dataset()

        train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset, unlabeled_dataset,
                                                                    test_dataset, labeled_indices, unlabeled_indices,
                                                                    self.kwargs, dataset_class.unlabeled_subset_num)

        model = self.model

        criterion = get_loss(self.args, dataset_class.labeled_class_samples)

        optimizer = torch.optim.Adam(model.parameters())

        acc_ratio = {}
        best_acc1, best_acc5, best_prec1, best_recall1, best_f1, best_confusion_mat, best_micro = \
            0, 0, 0, 0, 0, None, None
        self.args.start_epoch = 0
        self.args.weak_supervision_strategy = "random_sampling"
        current_labeled_ratio = self.args.labeled_ratio_start

        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_classifier(train_loader, model, criterion, optimizer, epoch)
            acc, acc5, (prec, recall, f1, _), confusion_mat, roc_auc_curve, micro_metrics = \
                self.validate_classifier(val_loader, model, criterion)

            is_best = recall > best_recall1

            if epoch > self.args.labeled_warmup_epochs and epoch % self.args.add_labeled_epochs == 0:
                acc_ratio.update({np.round(current_labeled_ratio, decimals=2):
                                 [best_acc1, best_acc5, best_prec1, best_recall1, best_f1,
                                 best_confusion_mat.tolist(), roc_auc_curve, best_micro]})

                train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, None, None,
                                     epoch, model, train_loader, unlabeled_loader,
                                     dataset_class, labeled_indices,
                                     unlabeled_indices, labeled_dataset,
                                     unlabeled_dataset,
                                     test_dataset, self.kwargs, current_labeled_ratio,
                                     None)

                current_labeled_ratio += self.args.add_labeled_ratio
                best_acc1, best_acc5, best_prec1, best_recall1, best_f1, best_confusion_mat, best_micro = \
                    0, 0, 0, 0, 0, None, None

                if self.args.reset_model:
                    model, optimizer, self.args = create_model_optimizer_autoencoder(self.args, dataset_class)

                criterion = get_loss(self.args, dataset_class.labeled_class_samples)
            else:
                best_acc1 = max(acc, best_acc1)
                best_prec1 = max(prec, best_prec1)
                best_recall1 = max(recall, best_recall1)
                best_acc5 = max(acc5, best_acc5)
                best_f1 = max(f1, best_f1)
                best_confusion_mat = confusion_mat if is_best else best_confusion_mat
                best_micro = micro_metrics if is_best else best_micro

            if current_labeled_ratio > self.args.labeled_ratio_stop:
                break

        if self.args.store_logs:
            store_logs(self.args, acc_ratio)

        return best_acc1

    def train_classifier(self, train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()

        model.train()

        for i, (data_x, data_y) in enumerate(train_loader):
            data_x = data_x.cuda(non_blocking=True)
            data_y = data_y.cuda(non_blocking=True)

            output = model.forward_classifier(data_x)

            loss = criterion(output, data_y)

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
                      .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

    def validate_classifier(self, val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        metrics = Metrics()

        model.eval()

        end = time.time()

        with torch.no_grad():
            for i, (data_x, data_y) in enumerate(val_loader):
                data_x = data_x.cuda(non_blocking=True)
                data_y = data_y.cuda(non_blocking=True)

                output = model.forward_classifier(data_x)

                loss = criterion(output, data_y)

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
                          .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

        (prec, recall, f1, _) = metrics.get_metrics()
        micro_metrics = metrics.get_metrics(average='micro')
        confusion_matrix = metrics.get_confusion_matrix()
        roc_auc_curve = metrics.get_roc_auc_curve()
        print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1} * Acc@5 {top5.avg:.3f}\t * Roc_Auc {2}\t'
              .format(prec, recall, roc_auc_curve, top1=top1, top5=top5))

        return top1.avg, top5.avg, (prec, recall, f1, _), confusion_matrix, roc_auc_curve, micro_metrics
