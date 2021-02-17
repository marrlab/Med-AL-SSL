import time
import torch
from torch.utils.data import DataLoader

from active_learning.others import UncertaintySamplingOthers
from data.isic_dataset import ISICDataset
from data.matek_dataset import MatekDataset
from data.cifar10_dataset import Cifar10Dataset
from data.jurkat_dataset import JurkatDataset
from data.plasmodium_dataset import PlasmodiumDataset
from data.retinopathy_dataset import RetinopathyDataset

from model.loss_net import LossNet
from utils import create_loaders, create_model_optimizer_scheduler, create_model_optimizer_loss_net, get_loss, \
    print_args, loss_module_objective_func, AverageMeter, accuracy, Metrics, store_logs, save_checkpoint, \
    perform_sampling, LossPerClassMeter, create_model_optimizer_autoencoder, load_pretrained, \
    create_model_optimizer_simclr, postprocess_indices

import pandas as pd
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


'''
Learning Loss for Active Learning (https://arxiv.org/pdf/1905.03677.pdf)
Code adapted from: https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''


class LearningLoss:
    def __init__(self, args, verbose=True):
        self.args = args
        self.verbose = verbose
        self.datasets = {'matek': MatekDataset, 'cifar10': Cifar10Dataset, 'plasmodium': PlasmodiumDataset,
                         'jurkat': JurkatDataset, 'isic': ISICDataset, 'retinopathy': RetinopathyDataset}
        self.model = None
        self.kwargs = {'num_workers': 16, 'pin_memory': False, 'drop_last': True}
        self.init = self.args.semi_supervised_init
        self.semi_supervised = self.args.semi_supervised_method

    def main(self):
        dataset_cl = self.datasets[self.args.dataset](root=self.args.root,
                                                      add_labeled=self.args.add_labeled,
                                                      advanced_transforms=True,
                                                      merged=self.args.merged,
                                                      remove_classes=self.args.remove_classes,
                                                      oversampling=self.args.oversampling,
                                                      unlabeled_subset_ratio=self.args.unlabeled_subset,
                                                      seed=self.args.seed, start_labeled=self.args.start_labeled)

        base_dataset, labeled_dataset, unlabeled_dataset, labeled_indices, unlabeled_indices, test_dataset = \
            dataset_cl.get_dataset()

        train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset, unlabeled_dataset,
                                                                    test_dataset, labeled_indices, unlabeled_indices,
                                                                    self.kwargs, dataset_cl.unlabeled_subset_num)

        model_backbone, optimizer_backbone, _ = create_model_optimizer_scheduler(self.args, dataset_cl)

        if self.init == 'pretrained' or (self.args.load_pretrained and self.init is None):
            print("Loading ImageNet pretrained model!")
            model_backbone = load_pretrained(model_backbone)
        elif self.init == 'autoencoder' or self.semi_supervised == 'auto_encoder_with_al':
            model_backbone, optimizer_backbone, _ = create_model_optimizer_autoencoder(self.args, dataset_cl)
        elif self.init == 'simclr' or self.semi_supervised == 'simclr_with_al':
            model_backbone, optimizer_backbone, _, _ = create_model_optimizer_simclr(self.args, dataset_cl)

        model_module = LossNet().cuda()
        optimizer_module = torch.optim.Adam(model_module.parameters())

        models = {'backbone': model_backbone, 'module': model_module}
        optimizers = {'backbone': optimizer_backbone, 'module': optimizer_module}

        criterion_backbone = get_loss(self.args, dataset_cl.labeled_class_samples, reduction='none')
        criterion_unlabeled = get_loss(self.args, dataset_cl.labeled_class_samples, reduction='none')

        criterions = {'backbone': criterion_backbone, 'module': loss_module_objective_func,
                      'unlabeled': criterion_unlabeled}

        uncertainty_sampler = UncertaintySamplingOthers(verbose=True,
                                                        uncertainty_sampling_method='learning_loss')

        train_loader_fix, unlabeled_loader_fix = None, None

        if 'fixmatch_with_al' == self.semi_supervised:
            labeled_dataset_fix, unlabeled_dataset_fix = dataset_cl.get_datasets_fixmatch(base_dataset,
                                                                                          labeled_indices,
                                                                                          unlabeled_indices)
            train_loader_fix = DataLoader(dataset=labeled_dataset_fix, batch_size=self.args.batch_size,
                                          shuffle=True, **self.kwargs)
            unlabeled_loader_fix = DataLoader(dataset=unlabeled_dataset_fix, batch_size=self.args.batch_size,
                                              shuffle=True, **self.kwargs)

        current_labeled = dataset_cl.start_labeled
        metrics_per_cycle = pd.DataFrame([])
        metrics_per_epoch = pd.DataFrame([])
        num_class_per_cycle = pd.DataFrame([])

        print_args(self.args)

        self.args.start_epoch, current_pseudo_labeled = 0, 0
        best_recall, best_report, last_best_epochs = 0, None, 0
        best_model = deepcopy(models['backbone'])

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if 'fixmatch_with_al' == self.semi_supervised:
                loaders_fix = zip(train_loader_fix, unlabeled_loader_fix)
                train_loss = self.train_fixmatch(loaders_fix, models, optimizers, criterions, epoch,
                                                 len(train_loader_fix), base_dataset.classes, last_best_epochs)
            else:
                train_loss = self.train(train_loader, models, optimizers, criterions, epoch, last_best_epochs)

            val_loss, val_report = self.validate(val_loader, models, criterions, last_best_epochs)

            if 'pseudo_label_with_al' == self.semi_supervised:
                samples_indices, samples_targets = self.get_pseudo_samples(best_model, unlabeled_loader,
                                                                           number=int(self.args.pseudo_labeling_num / 500))
                labeled_indices, unlabeled_indices = postprocess_indices(labeled_indices, unlabeled_indices,
                                                                         samples_indices)

                train_loader, unlabeled_loader, val_loader = create_loaders(self.args, labeled_dataset,
                                                                            unlabeled_dataset, test_dataset,
                                                                            labeled_indices, unlabeled_indices,
                                                                            self.kwargs,
                                                                            dataset_cl.unlabeled_subset_num)

                current_pseudo_labeled += len(samples_indices)

                print('Epoch Classifier: [{0}]\t'
                      'Pseudo Labels Added: [{1}]'.format(epoch, len(samples_indices)))

            is_best = val_report['macro avg']['recall'] > best_recall
            last_best_epochs = 0 if is_best else last_best_epochs + 1

            val_report = pd.concat([val_report, train_loss, val_loss], axis=1)
            metrics_per_epoch = pd.concat([metrics_per_epoch, val_report])

            if epoch > self.args.labeled_warmup_epochs and last_best_epochs > self.args.add_labeled_epochs:
                metrics_per_cycle = pd.concat([metrics_per_cycle, best_report])

                train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices = \
                    perform_sampling(self.args, uncertainty_sampler, epoch, models, train_loader, unlabeled_loader,
                                     dataset_cl, labeled_indices, unlabeled_indices, labeled_dataset, unlabeled_dataset,
                                     test_dataset, self.kwargs, current_labeled)

                if 'fixmatch_with_al' == self.semi_supervised:
                    labeled_dataset_fix, unlabeled_dataset_fix = dataset_cl.get_datasets_fixmatch(base_dataset,
                                                                                                  labeled_indices,
                                                                                                  unlabeled_indices)
                    train_loader_fix = DataLoader(dataset=labeled_dataset_fix, batch_size=self.args.batch_size,
                                                  shuffle=True, **self.kwargs)
                    unlabeled_loader_fix = DataLoader(dataset=unlabeled_dataset_fix, batch_size=self.args.batch_size,
                                                      shuffle=True, **self.kwargs)

                current_labeled += self.args.add_labeled
                last_best_epochs = 0

                if self.args.reset_model:
                    model_backbone, optimizer_backbone, scheduler_backbone = \
                        create_model_optimizer_scheduler(self.args, dataset_cl)

                    if self.init == 'pretrained':
                        model_backbone = load_pretrained(model_backbone)
                    elif self.init == 'autoencoder':
                        model_backbone, optimizer_backbone, _ = create_model_optimizer_autoencoder(self.args,
                                                                                                   dataset_cl)
                    elif self.init == 'simclr':
                        model_backbone, optimizer_backbone, _, _ = create_model_optimizer_simclr(self.args, dataset_cl)

                    model_module, optimizer_module = create_model_optimizer_loss_net()
                    models = {'backbone': model_backbone, 'module': model_module}
                    optimizers = {'backbone': optimizer_backbone, 'module': optimizer_module}

                if self.args.novel_class_detection:
                    num_classes = [np.sum(np.array(base_dataset.targets)[labeled_indices] == i)
                                   for i in range(len(base_dataset.classes))]
                    num_class_per_cycle = pd.concat([num_class_per_cycle,
                                                     pd.DataFrame.from_dict({cls: num_classes[i] for i, cls in
                                                                             enumerate(base_dataset.classes)},
                                                                            orient='index').T])

                criterion_backbone = get_loss(self.args, dataset_cl.labeled_class_samples, reduction='none')
                criterion_unlabeled = get_loss(self.args, dataset_cl.labeled_class_samples, reduction='none')
                criterions = {'backbone': criterion_backbone, 'module': loss_module_objective_func,
                              'unlabeled': criterion_unlabeled}
            else:
                best_recall = val_report['macro avg']['recall'] if is_best else best_recall
                best_report = val_report if is_best else best_report
                best_model = deepcopy(models['backbone']) if is_best else best_model

            if (current_labeled > self.args.stop_labeled) or (current_pseudo_labeled > self.args.pseudo_labeling_num):
                break

            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': model_backbone.state_dict(),
                'best_prec1': best_recall,
            }, is_best)

        if self.args.store_logs:
            store_logs(self.args, metrics_per_cycle)
            store_logs(self.args, metrics_per_epoch, log_type='epoch_wise')
            store_logs(self.args, num_class_per_cycle, log_type='novel_class')

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

            output, features = models['backbone'].forward_features(data_x)
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

                output = models['backbone'].forward_encoder_classifier(data_x)
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

    def train_fixmatch(self, train_loader, models, optimizers, criterions, epoch, loaders_len, cl, last_best_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        losses_per_class = LossPerClassMeter(len(cl))

        models['backbone'].train()
        models['module'].train()

        end = time.time()

        for i, (data_labeled, data_unlabeled) in enumerate(train_loader):
            data_x, data_y = data_labeled
            data_x, data_y = data_x.cuda(non_blocking=True), data_y.cuda(non_blocking=True)

            (data_w, data_s), _ = data_unlabeled
            data_w, data_s = data_w.cuda(non_blocking=True), data_s.cuda(non_blocking=True)

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            inputs = torch.cat((data_x, data_w, data_s))
            logits, features = models['backbone'].forward_features(inputs)
            logits_labeled = logits[:self.args.batch_size]
            logits_unlabeled_w, logits_unlabeled_s = logits[self.args.batch_size:].chunk(2)
            del logits

            features = [feat[:self.args.batch_size] for feat in features]
            target_loss = criterions['backbone'](logits_labeled, data_y)

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            losses_per_class.update(target_loss.cpu().detach().numpy(), data_y.cpu().numpy())
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

            pseudo_label = torch.softmax(logits_unlabeled_w.detach(), dim=-1)
            max_probs, data_y_unlabeled = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.args.fixmatch_threshold).float()
            loss_unlabeled = (criterions['unlabeled'](logits_unlabeled_s, data_y_unlabeled) * mask).mean()

            m_module_loss = criterions['module'](pred_loss, target_loss)

            loss = m_backbone_loss + \
                self.args.learning_loss_weight * m_module_loss + self.args.fixmatch_lambda_u * loss_unlabeled

            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

            acc = accuracy(logits_labeled.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(acc.item(), data_x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch Classifier: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Last best epoch {last_best_epoch}'
                      .format(epoch, i, loaders_len, batch_time=batch_time, loss=losses,
                              last_best_epoch=last_best_epochs))

        return pd.DataFrame.from_dict({f'{k}-train-loss': losses_per_class.avg[i]
                                       for i, k in enumerate(cl)}, orient='index').T

    def get_pseudo_samples(self, model, unlabeled_loader, number):
        batch_time = AverageMeter()
        samples = None
        samples_targets = None

        end = time.time()

        model.eval()

        for i, (data_x, _) in enumerate(unlabeled_loader):
            data_x = data_x.cuda(non_blocking=True)

            with torch.no_grad():
                output = model.forward_encoder_classifier(data_x)
            score = F.softmax(output, dim=1)
            score = torch.max(score, dim=1)

            samples = score[0] if samples is None else torch.cat([samples, score[0]])
            samples_targets = score[1] if samples_targets is None else torch.cat([samples_targets, score[1]])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                pass

        samples_targets = samples_targets[samples > self.args.pseudo_labeling_threshold]
        samples = samples[samples > self.args.pseudo_labeling_threshold]

        samples_indices = samples.argsort(descending=True)[:number]
        samples_targets = samples_targets[samples_indices]

        model.train()

        return samples_indices, samples_targets
