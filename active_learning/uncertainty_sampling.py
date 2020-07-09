import torch
import torch.nn.functional as F
from utils import AverageMeter
import time


class UncertaintySampling:
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

    def get_samples(self, epoch, args, model, unlabeled_loader, number):
        batch_time = AverageMeter()
        samples = None

        end = time.time()

        model.eval()

        for i, (data_x, data_y) in enumerate(unlabeled_loader):
            data_x = data_x.cuda(non_blocking=True)

            with torch.no_grad():
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
