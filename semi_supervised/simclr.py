from data.cifar10_dataset import Cifar10Dataset
from data.matek_dataset import MatekDataset
from data.cifar100_dataset import Cifar100Dataset
from model.lenet_simclr import LenetSimCLR
from utils import create_base_loader, AverageMeter, save_checkpoint, create_loaders, accuracy, Metrics, \
    stratified_random_sampling, postprocess_indices, store_logs, NTXent
import os
import time
import torch
import torch.nn as nn
import numpy as np

torch.autograd.set_detect_anomaly(True)


class SimCLR:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset}
        self.model = None

    def train(self):
        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         labeled_ratio=self.args.labeled_ratio_start,
                                                         add_labeled_ratio=self.args.add_labeled_ratio)

        base_dataset = dataset_class.get_base_dataset_simclr()

        kwargs = {'num_workers': 2, 'pin_memory': False}
        train_loader = create_base_loader(self.args, base_dataset, kwargs, self.args.simclr_batch_size)

        model = LenetSimCLR(num_channels=3,
                            num_classes=dataset_class.num_classes,
                            drop_rate=self.args.drop_rate, normalize=True)

        model = model.cuda()

        if self.args.resume:
            file = os.path.join(self.args.checkpoint_path, self.args.name, 'model_best.pth.tar')
            if os.path.isfile(file):
                print("=> loading checkpoint '{}'".format(file))
                checkpoint = torch.load(file)
                self.args.start_epoch = checkpoint['epoch']
                self.args.start_epoch = self.args.epochs
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(file))

        criterion = NTXent(self.args.simclr_temperature, torch.device("cuda"))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        best_loss = np.inf

        for epoch in range(self.args.start_epoch, self.args.simclr_train_epochs):
            model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for i, ((data_x_i, data_x_j), y) in enumerate(train_loader):
                data_x_i = data_x_i.cuda(non_blocking=True)
                data_x_j = data_x_j.cuda(non_blocking=True)

                h_i, z_i = model(data_x_i)
                h_j, z_j = model(data_x_j)

                loss = criterion(z_i, z_j, data_x_i.size(0))

                losses.update(loss.data.item(), data_x_i.size(0))

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
                                                         add_labeled_ratio=self.args.add_labeled_ratio)

        labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_class.get_dataset()

        kwargs = {'num_workers': 2, 'pin_memory': False}
        train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset, unlabeled_dataset,
                                                                    test_dataset,
                                                                    labeled_indices, unlabeled_indices, kwargs)

        model = self.model

        if self.args.weighted:
            classes_targets = unlabeled_dataset.targets[unlabeled_indices]
            classes_samples = [torch.sum(classes_targets == i) for i in range(dataset_class.num_classes)]
            classes_weights = np.log(len(unlabeled_dataset)) - np.log(classes_samples)
            criterion = nn.CrossEntropyLoss(weight=classes_weights).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.Adam(model.parameters())

        acc_ratio = {}
        best_acc1 = 0
        best_acc5 = 0
        best_prec1 = 0
        best_recall1 = 0
        self.args.start_epoch = 0
        current_labeled_ratio = self.args.labeled_ratio_start

        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train_classifier(train_loader, model, criterion, optimizer, epoch)
            acc, acc5, (prec, recall, f1, _) = self.validate_classifier(val_loader, model, criterion)

            if epoch > self.args.labeled_warmup_epochs and epoch % self.args.add_labeled_epochs == 0:
                acc_ratio.update({np.round(current_labeled_ratio, decimals=2):
                                 [best_acc1, best_acc5, best_prec1, best_recall1]})
                samples_indices = stratified_random_sampling(unlabeled_indices, number=dataset_class.add_labeled_num)

                labeled_indices, unlabeled_indices = postprocess_indices(labeled_indices, unlabeled_indices,
                                                                         samples_indices)

                train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset,
                                                                            unlabeled_dataset,
                                                                            test_dataset, labeled_indices,
                                                                            unlabeled_indices, kwargs)

                print(f'Random Sampling\t '
                      f'Current labeled ratio: {current_labeled_ratio}\t')

                current_labeled_ratio += self.args.add_labeled_ratio

            # is_best = best_acc1 > acc
            best_acc1 = max(best_acc1, acc)
            best_prec1 = max(prec, best_prec1)
            best_recall1 = max(recall, best_recall1)
            best_acc5 = max(acc5, best_acc5)

            '''
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_acc1,
            }, is_best)
            '''

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

            # model.eval()
            # with torch.no_grad():
            h = model.forward_encoder(data_x)
            # model.train()

            output = model.forward_classifier(h)

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
        for i, (data_x, data_y) in enumerate(val_loader):
            data_x = data_x.cuda(non_blocking=True)
            data_y = data_y.cuda(non_blocking=True)

            with torch.no_grad():
                h = model.forward_encoder(data_x)
                output = model.forward_classifier(h)

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
        print(' * Acc@1 {top1.avg:.3f}\t * Prec {0}\t * Recall {1} * Acc@5 {top5.avg:.3f}\t'
              .format(prec, recall, top1=top1, top5=top5))

        return top1.avg, top5.avg, (prec, recall, f1, _)
