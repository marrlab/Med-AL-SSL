import torch
import torch.nn.functional as F
from utils import AverageMeter
import time
import numpy as np


class UncertaintySamplingOthers:
    """
    Active Learning methods to sample for uncertainty
    Credits to: https://github.com/rmunro/pytorch_active_learning
    """

    def __init__(self, uncertainty_sampling_method, verbose=False):
        self.uncertainty_sampling_method = uncertainty_sampling_method
        self.method = getattr(self, self.uncertainty_sampling_method)
        self.verbose = verbose

    @staticmethod
    def least_confidence(probs):
        simple_least_conf = torch.max(probs, dim=1)[0]  # most confident prediction

        return simple_least_conf

    @staticmethod
    def margin_confidence(probs):
        probs = torch.sort(probs, dim=1)[0]
        diff = probs[:, -1] - probs[:, -2]

        return diff

    @staticmethod
    def ratio_confidence(probs):
        probs = torch.sort(probs, dim=1)[0]
        ratio = probs[:, -1]/probs[:, -2]

        return ratio

    @staticmethod
    def entropy_based(probs):
        log_probs = torch.log(probs)
        entropy = torch.sum(-probs * log_probs, dim=1)

        return entropy

    @staticmethod
    def learning_loss(models, unlabeled_loader, args, epoch, uncertainty_sampling_method):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([]).cuda()
        targets = None

        with torch.no_grad():
            for i, (data_x, data_y) in enumerate(unlabeled_loader):
                data_x = data_x.cuda(non_blocking=True)
                data_y = data_y.cuda(non_blocking=True)

                output, features = models['backbone'].forward_features(data_x)
                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))

                targets = data_y.cpu().numpy() if targets is None \
                    else np.concatenate([targets, data_y.cpu().numpy().tolist()])

                uncertainty = torch.cat((uncertainty, pred_loss), 0)

                if i % args.print_freq == 0:
                    print('{0}\t'
                          'Epoch: [{1}][{2}/{3}]\t'
                          .format(uncertainty_sampling_method, epoch, i, len(unlabeled_loader)))

        return uncertainty

    def get_samples(self, epoch, args, model, _, unlabeled_loader, number):
        batch_time = AverageMeter()
        samples = None
        targets = None

        end = time.time()

        if args.uncertainty_sampling_method == 'learning_loss':
            scores = self.learning_loss(model, unlabeled_loader, args, epoch, self.uncertainty_sampling_method)
            return scores.argsort(descending=True)[:number]

        model.eval()

        for i, (data_x, data_y) in enumerate(unlabeled_loader):
            data_x = data_x.cuda(non_blocking=True)
            targets = data_y.cpu().numpy() if targets is None \
                else np.concatenate([targets, data_y.cpu().numpy().tolist()])

            with torch.no_grad():
                if args.weak_supervision_strategy == 'semi_supervised_active_learning':
                    output = model.forward_encoder_classifier(data_x)
                else:
                    output = model(data_x)

            score = self.method(F.softmax(output, dim=1))

            samples = score if samples is None else torch.cat([samples, score])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('{0}\t'
                      'Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      .format(self.uncertainty_sampling_method, epoch, i, len(unlabeled_loader), batch_time=batch_time))

        if self.uncertainty_sampling_method == 'entropy_based':
            return samples.argsort(descending=True)[:number]
        else:
            return samples.argsort()[:number]
