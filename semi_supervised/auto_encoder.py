from data.cifar10_dataset import Cifar10Dataset
from data.matek_dataset import MatekDataset
from data.cifar100_dataset import Cifar100Dataset
from utils import create_base_loader, AverageMeter, save_checkpoint
from model.lenet_autoencoder import LenetAutoencoder
import os
import time
import torch
import torch.nn as nn
import numpy as np


class AutoEncoder:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'cifar100': Cifar100Dataset}

    def train(self):
        dataset_class = self.datasets[self.args.dataset](root=self.args.root,
                                                         labeled_ratio=self.args.labeled_ratio_start,
                                                         add_labeled_ratio=self.args.add_labeled_ratio)

        base_dataset = dataset_class.get_base_dataset()

        kwargs = {'num_workers': 2, 'pin_memory': False}
        train_loader = create_base_loader(self.args, base_dataset, kwargs)

        model = LenetAutoencoder(num_channels=3)

        # doc: for training on multiple GPUs.
        # doc: Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # doc: model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

        if self.args.resume:
            file = os.path.join(self.args.checkpoint_path, self.args.name, 'model_best.pth.tar')
            if os.path.isfile(file):
                print("=> loading checkpoint '{}'".format(file))
                checkpoint = torch.load(file)
                self.args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(file))

        criterion = nn.MSELoss().cuda()

        optimizer = torch.optim.SGD(model.parameters(), self.args.lr,
                                    momentum=self.args.momentum, nesterov=self.args.nesterov,
                                    weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        best_loss = np.inf

        for epoch in range(self.args.start_epoch, self.args.epochs):
            model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for i, (data_x, data_y) in enumerate(train_loader):
                data_x = data_x.cuda(non_blocking=True)

                optimizer.zero_grad()
                output = model(data_x)
                loss = criterion(output, data_x)

                losses.update(loss.data.item(), data_x.size(0))

                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

            scheduler.step(epoch=epoch)

            is_best = best_loss > losses.avg
            best_loss = min(best_loss, losses.avg)

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_loss,
            }, is_best)
